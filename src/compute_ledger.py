"""Compute ledger: how many FLOPs (reported in PFLOP = 1e15) the project has spent and will have spent when the running study finishes.

    uv run python src/compute_ledger.py --out results/compute_ledger

FLOPs per sequence are MEASURED, not estimated from parameter counts: torch's FlopCounterMode counts every matmul and attention op (forward and
backward) of the real models on a 512-token sequence with 30% of the positions labelled, as in training. It runs on the CPU (the GPUs stay with the
training runs). It counts matrix-multiply work only (what runs on tensor cores); softmax, norms, the optimizer and data loading are not counted, which
understates the total by a few percent. A loop iteration costs the same each time, so the cost at depth d is a + b d: measured at depths 1 and 2.

The ledger multiplies those per-sequence costs by how many sequences each job processed. Counts come from the run configurations and logs; a few
older runs (before the checkpoints were deleted) are taken from their logs and marked 'approximate'. Not counted: tests, smoke runs, the teacher
benchmarks and unit-test GPU jobs, and the evaluations added after the ledger's first version (bits per byte across tokenizers, zero-shot NLI,
the hardware study, whose timed latency loops alone may add a few percent)."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertForMaskedLM

import distill
import halting
import pins
from arch import BortForMaskedLM, MakeBortConfig
from compare_arch import FlatForMaskedLM, flat_config
from targets import make_targets, target_kl, teacher_hidden

SEQ, MASK = 512, 0.3
VOTERS = int(SEQ * MASK)
D = 8
PFLOP = 1e15
HEAD_FLOPS = 2 * VOTERS * 1024 * 1024  # the teacher's prediction head (one 1024x1024 matmul) on the labelled positions


def batch_for(vocab, generator):
    ids = torch.randint(5, vocab - 5, (1, SEQ), generator=generator)
    labels = torch.full_like(ids, -100)
    labels[0, :VOTERS] = ids[0, :VOTERS]
    return {"input_ids": ids.masked_fill(labels != -100, 4), "attention_mask": torch.ones_like(ids), "labels": labels}


def flops(fn):
    with FlopCounterMode(display=False) as counter:
        fn()
    return counter.get_total_flops()


def train_flops(student, batch, stored, bias, args, depth):
    """Forward + backward of one distillation step (loss from stored teacher targets) on one sequence."""
    student.train()
    kwargs = {} if depth is None else {"recurrence_depth": depth}

    def step():
        out = student(**batch, **kwargs)
        kl = target_kl(out.logits, stored, student.decoder.weight, bias, args.temperature).mean()
        distill.combine(out.loss, kl, args).backward()

    return flops(step)


def forward_flops(student, batch, stored, bias, args, depth):
    """No-grad forward of one sequence including the teacher-side decode of the KL (what an evaluation pass costs, minus the teacher forward)."""
    student.eval()
    kwargs = {} if depth is None else {"recurrence_depth": depth}

    def step():
        with torch.no_grad():
            out = student(**batch, **kwargs)
            target_kl(out.logits, stored, student.decoder.weight, bias, args.temperature)

    return flops(step)


def teacher_flops(config, batch):
    """FLOPs of teacher_hidden (encoder + prediction head at the labelled positions) for one sequence, from copies with 1 and 2 layers (so the
    full model never has to be in memory): layer 0 is full attention, layer 1 sliding; the real teacher has 'full' at every third of its layers."""
    def run(types):
        c = AutoConfig.from_pretrained(config.name_or_path, **pins.kw(config.name_or_path))
        c.num_hidden_layers, c.layer_types = len(types), types
        model = ModernBertForMaskedLM(c).eval()
        return flops(lambda: teacher_hidden(model, batch))

    one, two = run(["full_attention"]), run(["full_attention", "sliding_attention"])
    sliding, full_and_head = two - one, one
    types = config.layer_types
    return full_and_head + (types.count("full_attention") - 1) * (one - HEAD_FLOPS) + types.count("sliding_attention") * sliding


def measure():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    args = distill.DistillArguments(output_dir="runs/_ledger", report_to="none", use_cpu=True)
    g = torch.Generator().manual_seed(0)
    teacher_config = AutoConfig.from_pretrained(args.teacher, **pins.kw(args.teacher))
    V, H = teacher_config.vocab_size, teacher_config.hidden_size
    bias = torch.zeros(V)
    batch = batch_for(V, g)
    stored = make_targets(torch.randn(VOTERS, H, generator=g).bfloat16(), torch.randn(V, H), bias, args.temperature, 0, 1.0)
    out = {"teacher_forward": teacher_flops(teacher_config, batch)}

    def bort(split, merge_base="state"):
        config = MakeBortConfig(args.teacher, *split)
        config.sparse_prediction = True
        config.merge_base = merge_base
        return BortForMaskedLM(config)

    models = {"bort8": (bort((2, 4, 2)), True), "bort4": (bort((1, 2, 1)), True), "bort4_anchor": (bort((1, 2, 1), "anchor"), True)}
    for name, split, types in (("flat8", (2, 4, 2), "bort"), ("flat4", (1, 2, 1), "bort"), ("flat10", (1, 8, 1), "modernbert")):
        models[name] = (FlatForMaskedLM(flat_config(args.teacher, layers=split, layer_types=types)), False)
    for name, (model, looped) in models.items():
        model.requires_grad_(True)
        model.decoder.weight.requires_grad_(False)  # tied, frozen teacher embedding (no weight gradient for it)
        depths = (1, 2) if looped else (None,)
        tr = {d or 1: train_flops(model, batch, stored, bias, args, d) for d in depths}
        fw = {d or 1: forward_flops(model, batch, stored, bias, args, d) for d in depths}
        if looped:
            out[name] = {"train": {d: tr[1] + (d - 1) * (tr[2] - tr[1]) for d in range(1, D + 1)},
                         "forward": {d: fw[1] + (d - 1) * (fw[2] - fw[1]) for d in range(1, D + 1)}}
            if name == "bort8":  # the linear model in depth is exact: check it against a direct measurement at depth 8
                direct = train_flops(model, batch, stored, bias, args, 8)
                out[name]["linear_check_depth8"] = [out[name]["train"][8], direct]
        else:
            out[name] = {"train": {1: tr[1]}, "forward": {1: fw[1]}}
        model.zero_grad(set_to_none=True)
    # the halting passes, measured directly on Bort (8 blocks), one sequence
    bortm = models["bort8"][0].eval()
    stored_t = stored
    out["table_slow"] = flops(lambda: halting.depth_table(bortm, batch, stored_t, bias, args, range(1, D + 1), signals=True))
    out["table_fast"] = flops(lambda: halting.depth_table_fast(bortm, batch, stored_t, bias, args, range(1, D + 1), signals=True, confidence=True))
    return out


def mean_train(m, arch):
    t = m[arch]["train"]
    return sum(t.values()) / len(t)


def sum_forward(m, arch):
    return sum(m[arch]["forward"].values())


def ledger(m):
    """-> list of (group, job, sequences, FLOPs per sequence, note). Sequences = training steps x 32 where relevant."""
    T = m["teacher_forward"]
    rows = []

    def add(group, job, seqs, per_seq, note=""):
        rows.append({"group": group, "job": job, "sequences": seqs, "gflop_per_sequence": per_seq / 1e9, "pflop": seqs * per_seq / PFLOP, "note": note})

    eval_extra = 0.02  # evaluations every 500 steps (3 depths x 256 sequences) are ~1.6% of a run's training FLOPs; allow 2%
    for job, steps, arch, teacher_steps, note in (
        ("bort1 (joint, local teacher)", 1000, "bort8", 1000, "approximate (log)"),
        ("bort100m (joint, local teacher)", 3200, "bort8", 3200, "approximate (log; part of it resumed from bort1)"),
        ("bort_shared", 6200, "bort8", 6200, "shared teacher stream: one teacher pass for both"),
        ("flat_shared (assumed same length)", 6200, "flat8", 0, "teacher pass counted once, under bort_shared; approximate"),
        ("flat100m", 6200, "flat8", 6200, "approximate (log)"),
    ):
        add("history before this study", job + ": student training", steps * 32, mean_train(m, arch) * (1 + eval_extra), note)
        if teacher_steps:
            add("history before this study", job + ": teacher", teacher_steps * 32, T, note)

    steps = 12208
    for arm, arch in (("A looped + merge", "bort8"), ("B looped, no merge", "bort8"), ("C flat", "flat8")):
        add("200M-token arms", f"{arm}: student training", steps * 32, mean_train(m, arch) * (1 + eval_extra), "12,208 steps x 32 seq")
    add("200M-token arms", "teacher streams (A+C shared one pass, B a second)", 2 * steps * 32, T, "2 passes x 390,656 seq")

    seqs = 4 * 300 * 32
    add("halting v1 (4 heads)", "frozen student at 8 depths (slow, per-depth runs)", seqs, m["table_slow"], "300 steps x 32 seq x 4 heads")
    add("halting v1 (4 heads)", "teacher", seqs, T, "")
    add("halting v1 (4 heads)", "head evals (6 x 256 seq per head)", 4 * 6 * 256, m["table_slow"] + T, "")

    ev = []  # (job, sequences, per-sequence cost)
    per_run_bench = 1024
    bench_models = 2 * sum_forward(m, "bort8") + m["flat8"]["forward"][1]
    ev.append(("bench_heldout, 4 runs (2 invalid) x (A, B at 8 depths, C) x 1024 seq", 4 * per_run_bench, bench_models + 2 * T))
    ev.append(("halting_stats (A, B; 1024 seq each, slow table)", 2 * 1024, m["table_slow"] + T))
    ev.append(("halting_free (A, 512 seq): slow table + 6 token-rule passes + flat C", 512, m["table_slow"] + 2 * T + 6 * m["bort8"]["forward"][8] + m["flat8"]["forward"][1]))
    ev.append(("halting_eval, final run (A, B; 256 seq: table + 3 change + 3 head passes) + the killed first run", int(256 * 2 * 1.3),
               m["table_slow"] + T + 6 * m["bort8"]["forward"][8]))
    ev.append(("flops_score + seq_max_exit (A, B and A; ~1024 seq slow tables)", 1024, m["table_slow"] + T))
    ev.append(("bench_fast + autotune pools (small)", 2000, m["table_slow"] + T))
    for job, seqs, per in ev:
        add("evaluations", job, seqs, per, "approximate counts")

    add("halting v2 features", "feature shards: 10,240 train + 512 check + 1,024 held-out seq (fast one-pass table)", 11776, m["table_fast"], "")
    add("halting v2 features", "teacher for the shards", 11776, T, "")

    add("small-model study", "teacher cache build (98,304 + 256 seq, local teacher)", 98560, T, "one pass, reused by all fifteen runs")
    add("small-model study", "A', B' training: 6 runs x 3072 steps x 32 seq", 6 * 3072 * 32, mean_train(m, "bort4") * (1 + eval_extra), "1+2+1 blocks, depth 1..8")
    add("small-model study", "C' training: 3 runs x 3072 steps x 32 seq", 3 * 3072 * 32, mean_train(m, "flat4") * (1 + eval_extra), "4 flat blocks")
    add("small-model study", "D' training: 3 runs x 3072 steps x 32 seq", 3 * 3072 * 32, mean_train(m, "flat10") * (1 + eval_extra),
        "10 flat blocks (iso-FLOP control)")
    add("small-model study", "E' training: 3 runs x 3072 steps x 32 seq", 3 * 3072 * 32, mean_train(m, "bort4_anchor") * (1 + eval_extra),
        "1+2+1 blocks, depth 1..8, input injection (merge_base anchor)")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/compute_ledger")
    a = p.parse_args(argv)
    m = measure()
    rows = ledger(m)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = {}
    for r in rows:
        groups[r["group"]] = groups.get(r["group"], 0) + r["pflop"]
    total = sum(groups.values())
    (out.with_suffix(".json")).write_text(json.dumps({"measured_gflop_per_sequence": {
        "teacher_forward": m["teacher_forward"] / 1e9, "table_slow": m["table_slow"] / 1e9, "table_fast": m["table_fast"] / 1e9,
        **{k: {"train_by_depth": {d: v / 1e9 for d, v in m[k]["train"].items()}, "forward_by_depth": {d: v / 1e9 for d, v in m[k]["forward"].items()}}
           for k in ("bort8", "bort4", "bort4_anchor", "flat8", "flat4", "flat10")}}, "rows": rows, "group_pflop": groups, "total_pflop": total}, indent=1))
    lines = ["# Compute ledger", "", f"Total: **{total:,.0f} PFLOP** ({total / 1e3:.2f} EFLOP).", "", "| group | PFLOP |", "|---|---:|"]
    lines += [f"| {g} | {v:,.1f} |" for g, v in groups.items()]
    lines += ["", "| group | job | sequences | GFLOP/seq | PFLOP | note |", "|---|---|---:|---:|---:|---|"]
    lines += [f"| {r['group']} | {r['job']} | {r['sequences']:,} | {r['gflop_per_sequence']:,.0f} | {r['pflop']:,.1f} | {r['note']} |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("measured GFLOP per 512-token sequence:")
    print(f"  teacher forward {m['teacher_forward'] / 1e9:.0f}; one-pass table {m['table_fast'] / 1e9:.0f}; per-depth table {m['table_slow'] / 1e9:.0f}")
    for k in ("bort8", "bort4", "bort4_anchor", "flat8", "flat4", "flat10"):
        t, f = m[k]["train"], m[k]["forward"]
        print(f"  {k}: train (fwd+bwd) depth1 {t[1] / 1e9:.0f}, mean over depth schedule {mean_train(m, k) / 1e9:.0f}; forward depth1 {f[1] / 1e9:.0f}")
    predicted, direct = (x / 1e9 for x in m["bort8"]["linear_check_depth8"])
    print(f"  linear-in-depth check (bort8 train, depth 8): predicted {predicted:.1f} vs measured {direct:.1f} GFLOP")
    for g, v in groups.items():
        print(f"{v:10,.1f} PFLOP  {g}")
    print(f"{total:10,.1f} PFLOP  TOTAL")


if __name__ == "__main__":
    main()

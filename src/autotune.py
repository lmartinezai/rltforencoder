"""Autotune the real trainer: micro-batch x gradient checkpointing x torch.compile for this GPU.

No reimplementation of the training step: every candidate runs `distill.train` itself (HF Trainer + accelerate, the depth
schedule, the loss, fused AdamW, the allocator cap and TF32 of `distill.configure_cuda`) on a store of real teacher targets for
a few optimizer steps, in a fresh process (an OOM or fragmentation never leaks into the next candidate). A Trainer callback
times aligned, complete depth cycles 1..max_depth. A separate run checks losses, gradients and parameters without
instrumenting the timed run. Candidates run in random order, `--repeats` times.

    uv run python src/autotune.py                                      # builds runs/bench/pool once, sweeps micro-batches
    uv run python src/autotune.py --mbs 3,4,5,6 --eff-batch 60 --compile 0,1 --ckpt 0,1
"""
import argparse
import itertools
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import HfArgumentParser, TrainerCallback

from distill import DistillArguments, DistillTrainer, configure_cuda, depth_for_step, run_extract, train
from store import TargetStore

TIE = 0.01  # only candidates within 1% of the fastest are treated as ties


class CheckedTrainer(DistillTrainer):
    """Accumulate loss validity on device without synchronizing every micro-batch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validity = next(c for c in self.callback_handler.callbacks if isinstance(c, Finiteness))

    def compute_loss(self, *args, **kwargs):
        loss = super().compute_loss(*args, **kwargs)
        if self.model.training:
            self.validity.check(loss)
        return loss


class Finiteness(TrainerCallback):
    def __init__(self):
        self.flags = []

    def check(self, tensor):
        self.flags.append(torch.isfinite(tensor.detach()).all())

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        for p in model.parameters():
            if p.grad is not None:
                self.check(p.grad)

    def validate(self, model):
        for p in model.parameters():
            self.check(p)
        if not self.flags or not torch.stack(self.flags).all().item():
            raise ValueError("non-finite loss, gradient, or final parameter")


class StepTimer(TrainerCallback):
    """Mean seconds per optimizer step after `warmup` steps (compile, allocator, cudnn autotune) and the peak VRAM since."""

    def __init__(self, warmup):
        self.warmup, self.t = warmup, []

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        if state.global_step == self.warmup:
            torch.cuda.reset_peak_memory_stats()
        self.t.append(time.perf_counter())  # t[i] = end of step i + 1

    def result(self):
        n = len(self.t) - self.warmup
        return dict(step_s=(self.t[-1] - self.t[self.warmup - 1]) / n, steps=n,
                    peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30)


# ---------------------------------------------------------------- one candidate, in its own process

def worker(spec):
    configure_cuda()
    (args,) = HfArgumentParser(DistillArguments).parse_dict(spec["args"])
    if spec["kind"] == "pool":
        run_extract(args)
        out = {}
    elif spec["kind"] == "validate":
        validity = Finiteness()
        trainer = train(args, callbacks=[validity], trainer_class=CheckedTrainer)
        validity.validate(trainer.model)
        out = {"valid": True}
    else:
        timer = StepTimer(spec["warmup"])
        train(args, callbacks=[timer])
        out = timer.result()
        out["depths"] = [depth_for_step(s, args.max_depth, args.seed)
                         for s in range(spec["warmup"], args.max_steps)]
        out["device"] = torch.cuda.get_device_name()
        out["torch"] = torch.__version__
    print("RESULT " + json.dumps(out), flush=True)


def launch(kind, args, timeout=3600, **spec):
    """Run one candidate in a fresh process -> its RESULT dict | {"oom": True} | {"error": stderr tail}."""
    cmd = [sys.executable, __file__, "--worker", json.dumps(dict(kind=kind, args=args, **spec))]
    try:
        p = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s"}
    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("RESULT ")), None)
    if line and p.returncode == 0:
        return json.loads(line[7:])
    return {"oom": True} if "OutOfMemoryError" in p.stderr else {"error": "\n".join(p.stderr.splitlines()[-6:])}


# ---------------------------------------------------------------- the sweep

def check_ladder(mbs, eff_batch):
    """Every micro-batch must divide the effective batch (accumulation steps are whole numbers): refuse, never skip."""
    bad = [mb for mb in mbs if mb < 1 or eff_batch % mb]
    if bad:
        raise ValueError(f"micro-batches {bad} do not divide --eff-batch {eff_batch}: pick sizes from its divisors "
                         f"{[d for d in range(1, eff_batch + 1) if eff_batch % d == 0]} or change --eff-batch")


def summarize(rows, eff_batch, seq_len):
    """rows (one per run) -> one line per candidate: median step time, tokens/s, spread, peak VRAM; any failed run disqualifies."""
    df = pd.DataFrame(rows).reindex(columns=["mb", "ckpt", "compile", "step_s", "peak_reserved_gb"])  # all-failed: columns still exist
    df["done"] = df["step_s"].notna()
    out = df.groupby(["mb", "ckpt", "compile"]).agg(
        runs=("done", "size"), ok=("done", "all"), step_s=("step_s", "median"), peak_gb=("peak_reserved_gb", "max"),
        spread=("step_s", lambda s: s.max() / s.min() - 1)).reset_index()
    out["tok_s"] = (eff_batch * seq_len / out["step_s"]).where(out["ok"])
    return out.sort_values("tok_s", ascending=False, na_position="last", ignore_index=True)


def pick(table):
    """Choose least reserved memory among candidates within 1% of the fastest."""
    ok = table[table["ok"]]
    if ok.empty:
        return None
    tied = ok[ok["tok_s"] >= ok["tok_s"].max() * (1 - TIE)]
    return tied.sort_values(["peak_gb", "tok_s"], ascending=[True, False]).iloc[0]


def cli(best, eff_batch, optim, max_depth=8, seed=0):
    return (f"--per_device_train_batch_size {best['mb']} --gradient_accumulation_steps {eff_batch // best['mb']} "
            + ("--gradient_checkpointing " if best["ckpt"] else "") + ("--compile_layers " if best["compile"] else "")
            + f"--optim {optim} --max_depth {max_depth} --seed {seed}")


def aligned_warmup(warmup, max_depth):
    """Complete the current shuffled depth bucket before starting the timer."""
    return ((warmup + max_depth - 1) // max_depth) * max_depth


def trainer_args(a, pool, mb, ckpt, compile_, out):
    return dict(output_dir=out, targets_dir=str(pool), per_device_train_batch_size=mb,
                gradient_accumulation_steps=a.eff_batch // mb, gradient_checkpointing=bool(ckpt), compile_layers=bool(compile_),
                optim=a.optim, max_depth=a.max_depth, max_steps=a.aligned_warmup + a.cycles * a.max_depth, eval_strategy="no",
                save_strategy="no", report_to="none", disable_tqdm=True, logging_steps=10**6, seed=a.seed,
                prelude_layers=a.split[0], looped_layers=a.split[1], coda_layers=a.split[2])


def run(a, log=print):
    check_ladder(a.mbs, a.eff_batch)  # before the pool or any worker
    if a.eff_batch < 1 or a.max_depth < 1 or a.warmup < 1 or a.cycles < 1 or a.repeats < 1 or a.seq_len < 3:
        raise ValueError("effective batch, depth, warmup, cycles, repeats must be positive and seq-len at least 3")
    if not a.mbs or not a.ckpt or not a.compile or any(x not in (0, 1) for x in a.ckpt + a.compile):
        raise ValueError("candidate lists must be nonempty; ckpt and compile must be 0 or 1")
    a.aligned_warmup = aligned_warmup(a.warmup, a.max_depth)
    if a.aligned_warmup != a.warmup:
        log(f"warmup extended to {a.aligned_warmup} steps to start at a depth-cycle boundary")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "recommended.json"
    report_path.write_text(json.dumps({"status": "incomplete", "recommendation": None}, indent=1))
    if a.repeats < 3:
        log(f"warning: --repeats {a.repeats}: run-to-run noise cannot be estimated from fewer than 3 runs")
    pool = Path(a.pool)
    if not pool.exists() or not any(pool.iterdir()):
        required = a.eff_batch * (a.aligned_warmup + a.cycles * a.max_depth)
        if a.pool_seqs < required:
            raise ValueError(f"--pool-seqs must be at least {required} for full measured updates")
        log(f"building the benchmark pool ({a.pool_seqs} real FineWeb sequences, teacher on this GPU) in {pool}")
        res = launch("pool", dict(output_dir=tempfile.mkdtemp(), stage="extract", targets_dir=str(pool), eval_samples=8,
                                  extract_tokens=a.pool_seqs * a.seq_len, seq_len=a.seq_len, seed=a.seed))
        if "error" in res or "oom" in res:
            raise SystemExit(f"pool build failed: {res}")
    try:
        stores = [TargetStore(pool / s) for s in ("train", "eval")]
        for ds in stores:
            if ds.meta["seq_len"] != a.seq_len or len(ds) != ds.meta["n_seq"] or not len(ds):
                raise ValueError("sequence length, count, or store completeness mismatch")
        needed = a.eff_batch * (a.aligned_warmup + a.cycles * a.max_depth)
        if len(stores[0]) < needed:
            raise ValueError(f"train pool needs at least {needed} sequences; found {len(stores[0])}")
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid benchmark pool {pool}: {exc}; resume extraction or choose a new pool") from exc
    jobs = list(itertools.product(a.mbs, a.ckpt, a.compile)) * a.repeats
    random.Random(a.seed).shuffle(jobs)  # no candidate is systematically first (cold GPU) or last (hot GPU)
    rows, validations = [], {}
    for mb, ckpt, compile_ in jobs:
        key = (mb, ckpt, compile_)
        with tempfile.TemporaryDirectory() as out:
            args = trainer_args(a, pool, mb, ckpt, compile_, out)
            if key not in validations:
                validations[key] = launch("validate", args)
            validation = validations[key]
            res = (launch("train", args, warmup=a.aligned_warmup) if validation.get("valid")
                   else {"error": f"validation failed: {validation}"})
            if "step_s" in res and (res.get("steps") != a.cycles * a.max_depth or
                    any(sorted(res.get("depths", [])[i:i + a.max_depth]) != list(range(1, a.max_depth + 1))
                        for i in range(0, a.cycles * a.max_depth, a.max_depth))):
                res = {"error": "incomplete or unbalanced timed depth cycles"}
        rows.append(dict(mb=mb, ckpt=ckpt, compile=compile_, **res))
        what = (f"{res['step_s']:6.3f} s/step  {a.eff_batch * a.seq_len / res['step_s']:7.0f} tok/s  {res['peak_reserved_gb']:5.2f} GB"
                if "step_s" in res else "OOM" if res.get("oom") else f"FAILED: {res['error']}")
        log(f"  mb={mb:<3} ckpt={ckpt} compile={compile_}  {what}")
    table = summarize(rows, a.eff_batch, a.seq_len)
    best = pick(table)
    report = dict(status="ok" if best is not None else "failed", recommendation=None,
                  settings=vars(a), mode="stored targets", pool_meta={"train": stores[0].meta, "eval": stores[1].meta},
                  validation=[dict(mb=k[0], ckpt=k[1], compile=k[2], result=v) for k, v in validations.items()],
                  runs=rows, tie_tolerance=TIE)
    table.to_csv(report_path.parent / "autotune.csv", index=False)
    if best is None:
        report_path.write_text(json.dumps(report, indent=1, default=str))
        raise SystemExit("no feasible candidate (every run failed or ran out of memory)")
    command = cli(best, a.eff_batch, a.optim, a.max_depth, a.seed)
    best_times = [r["step_s"] for r in rows if (r["mb"], r["ckpt"], r["compile"]) ==
                  (best["mb"], best["ckpt"], best["compile"]) and "step_s" in r]
    overlap = []
    for key in validations:
        times = [r["step_s"] for r in rows if (r["mb"], r["ckpt"], r["compile"]) == key and "step_s" in r]
        if key != (best["mb"], best["ckpt"], best["compile"]) and len(times) == a.repeats and \
                len(times) > 1 and min(times) <= max(best_times) and min(best_times) <= max(times):
            overlap.append(dict(mb=key[0], ckpt=key[1], compile=key[2]))
    report["recommendation"] = dict(cli=command, tok_s=float(best["tok_s"]), peak_gb=float(best["peak_gb"]),
                                    seq_len=a.seq_len, overlapping_candidates=overlap,
                                    comparison="inconclusive" if overlap or a.repeats < 3 else "observed fastest or within 1%")
    report_path.write_text(json.dumps(report, indent=1, default=str))
    log("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log(f"\nrecommended: {command}   ({best['tok_s']:,.0f} tok/s, {best['peak_gb']:.1f} GB reserved)")
    if overlap or a.repeats < 3:
        log("comparison is inconclusive: measurement ranges overlap or fewer than three repeats were run")
    return table, best


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mbs", type=int_list, default=[2, 4, 6], help="micro-batch sizes (must divide --eff-batch)")
    ap.add_argument("--eff-batch", type=int, default=60, help="sequences per optimizer step (held constant)")
    ap.add_argument("--ckpt", type=int_list, default=[0], help="gradient checkpointing off/on: 0 | 1 | 0,1")
    ap.add_argument("--compile", type=int_list, default=[1], help="torch.compile of the student layers off/on: 0 | 1 | 0,1")
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--max-depth", type=int, default=8, help="deepest recurrence sampled in training (distill.py --max_depth)")
    ap.add_argument("--warmup", type=int, default=2, help="steps before timing starts")
    ap.add_argument("--cycles", type=int, default=2, help="whole 1..max_depth depth cycles timed per run")
    ap.add_argument("--repeats", type=int, default=4, help="runs per candidate (>= 3 to estimate run-to-run noise)")
    ap.add_argument("--split", type=int_list, default=[2, 4, 2], help="prelude,looped,coda blocks of the Bort being timed (2,4,2 = the 8-block Bort)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default="runs/bench/pool", help="store of real teacher targets (built on first use)")
    ap.add_argument("--pool-seqs", type=int, default=1600)
    ap.add_argument("--out", default="runs/autotune")
    return ap


def int_list(s):
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    ap = parser()
    ap.add_argument("--worker", metavar="JSON", help="internal: run one candidate in this process (the driver spawns these)")
    args = ap.parse_args()
    worker(json.loads(args.worker)) if args.worker else run(args)

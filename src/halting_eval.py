"""Does adaptive halting pay for Bort? Held-out arXiv text and the FineWeb hold-out (src/bench_heldout.py's data), real distillation loss.

    uv run python src/halting_eval.py --models A=runs/arm_a_merge/checkpoint-12208,runs/halt_a_token/halting_head.pt,runs/halt_a_seq/halting_head.pt \
        B=runs/arm_b_nomerge/checkpoint-12208,runs/halt_b_token/halting_head.pt,runs/halt_b_seq/halting_head.pt \
        --flat C=runs/arm_c_noloop/checkpoint-12208

(NAME=checkpoint[,token_head.pt[,sequence_head.pt]]; omitted heads are skipped; the parameter-free baselines always run.)

Cost is the LOOP COUNT: the 8-block Bort (2 + 4 + 2, the only model this script was run on; `blocks` assumes it) at depth k costs 4 + 4k blocks and the flat model 8, and in Universal-Transformer halting the loop runs until the
last deciding position has halted. Every exit rule is therefore reported as (mean loop count, loss) and compared with the fixed-depth model of
the same loop count (loss interpolated between neighbouring depths = what picking a depth at random per sequence gives).

  sequence level  one decision per sequence; read exactly off the per-depth runs (a sequence stopped after k is the depth-k forward: tested)
                  rules: trained head on the sequence's mean state | state change | output KL between consecutive depths
  token level     the masked tokens halt individually (freeze.py: halted tokens keep their state while the rest iterate and attend to them;
                  the sequence is done when all of them have halted): a real forward, one per threshold
                  rules: trained head per token | state change
Parameter-free rules (CALM's state saturation, Huginn's KL) need no training; their thresholds are quantiles of the observed signal.
Also printed: the hindsight oracle (an upper bound: picks the best of 8 noisy losses), and the flat model's loss at its cost of 8 blocks."""
import argparse
import json
import sys

import torch

import distill
import freeze
from arch import BortForMaskedLM
from bench_heldout import heldout_batches, load_teacher
from compare_arch import FlatForMaskedLM
from halting import depth_table, first_true, interpolate, pool_by_sequence, token_losses
from targets import make_targets, teacher_hidden

DEPTHS = 8


def point(losses_sum, k, n_tokens, fixed):
    """Result of exit depths k [B] for sequences with per-depth summed losses [D, B]."""
    loss = losses_sum[k - 1, torch.arange(len(k))].sum().item() / n_tokens
    loops = k.float().mean().item()
    return {"loops": loops, "blocks": 4 + 4 * loops, "loss": loss, "fixed_loss": interpolate(fixed, loops), "gain": interpolate(fixed, loops) - loss}


@torch.no_grad()
def collect(student, teacher, bias, batches, args):
    """Per-depth runs over all batches -> per-token losses and signals, the token head's logits, pooled sequence states, plus the GPU-cached
    inputs for the token-level forwards."""
    out = {"loss": [], "kl": [], "change": [], "pooled": [], "rows": [], "cache": []}
    offset = 0
    for batch in batches:
        batch = {k: v.cuda() for k, v in batch.items()}
        stored = make_targets(teacher_hidden(teacher, batch), student.decoder.weight, bias, args.temperature, 0, 1.0)  # as the server: no autocast
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states, losses, kl, change = depth_table(student, batch, stored, bias, args, range(1, DEPTHS + 1), signals=True)
        rows = (batch["labels"] != -100).nonzero()[:, 0]
        n_seq = len(batch["labels"])
        out["loss"].append(losses.cpu()), out["kl"].append(kl.cpu()), out["change"].append(change.cpu())
        out["pooled"].append(pool_by_sequence(states, rows, n_seq).cpu())
        out["rows"].append((rows + offset).cpu())
        out["cache"].append((batch, stored))
        offset += n_seq
    for key in ("loss", "kl", "change"):
        out[key] = torch.cat(out[key], 1)
    out["rows"], out["pooled"] = torch.cat(out["rows"]), torch.cat(out["pooled"], 1)
    return out


def sequence_level(data, seq_head, head_thresholds, quantiles):
    """Offline sequence-level exits for the head and the two free signals."""
    rows, loss, n = data["rows"], data["loss"], data["loss"].shape[1]
    n_seq = int(rows.max()) + 1
    seq_sum = torch.zeros(DEPTHS, n_seq).index_add_(1, rows, loss)
    counts = torch.bincount(rows, minlength=n_seq).float()
    fixed = loss.mean(1).tolist()

    def mean_signal(x):
        return torch.zeros(DEPTHS, n_seq).index_add_(1, rows, torch.nan_to_num(x, posinf=0.0)) / counts

    result = {"fixed": fixed, "n_tokens": n, "n_sequences": n_seq, "rules": {}}
    result["oracle"] = point(seq_sum, (seq_sum / counts).argmin(0) + 1, n, fixed)
    for name in ("change", "kl"):
        sig = mean_signal(data[name])
        sig[0] = torch.inf  # no comparison available at depth 1
        thresholds = torch.quantile(sig[1:].flatten(), torch.tensor(quantiles)).tolist()
        result["rules"][name] = [{"threshold": t, **point(seq_sum, first_true(sig < t), n, fixed)} for t in thresholds]
    if seq_head is not None:
        logit = data["pooled"] @ seq_head["weight"].squeeze(0) + seq_head["bias"]
        result["rules"]["head"] = [{"threshold": t, **point(seq_sum, first_true(logit.sigmoid() >= t), n, fixed)} for t in head_thresholds]
    return result


@torch.no_grad()
def token_forward(student, data, args, rule_factory):
    """One real freeze-and-continue forward over the cached batches -> loss, mean loop count (per sequence), mean token exit iteration."""
    total = tokens = exits = 0.0
    loops = []
    for batch, stored in data["cache"]:
        voters = batch["labels"] != -100
        with torch.autocast("cuda", dtype=torch.bfloat16), freeze.FreezeLoop(student, voters, rule_factory(voters), "token") as halt:
            out = student(**batch, recurrence_depth=DEPTHS)
        loss = token_losses(out.logits, batch["labels"][voters], stored, student.decoder.weight, args.bias, args)
        total, tokens = total + loss.sum().item(), tokens + loss.numel()
        loops.append(halt.loops.float().cpu())
        exits += halt.exit[voters].float().sum().item()
    mean_loops = torch.cat(loops).mean().item()
    return {"loops": mean_loops, "blocks": 4 + 4 * mean_loops, "loss": total / tokens, "token_exit": exits / tokens}


def token_level(student, data, args, token_head, head_thresholds, quantiles):
    fixed = data["loss"].mean(1).tolist()
    finite = data["change"][1:][torch.isfinite(data["change"][1:])]
    taus = torch.quantile(finite[:2_000_000], torch.tensor(quantiles)).tolist()
    rules = {"change": [(tau, lambda voters, tau=tau: freeze.change_rule(tau, "token", voters)) for tau in taus]}
    if token_head is not None:
        rules["head"] = [(theta, lambda voters, theta=theta: freeze.head_rule(token_head["weight"], token_head["bias"], theta, "token", voters))
                         for theta in head_thresholds]
    result = {"rules": {}}
    for name, configs in rules.items():
        result["rules"][name] = []
        for threshold, factory in configs:
            r = token_forward(student, data, args, factory)
            result["rules"][name].append({"threshold": threshold, **r, "fixed_loss": interpolate(fixed, r["loops"]),
                                          "gain": interpolate(fixed, r["loops"]) - r["loss"]})
    return result


@torch.no_grad()
def flat_loss(flat, cache, args):
    total = n = 0.0
    for batch, stored in cache:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = flat(**batch)
        loss = token_losses(out.logits, batch["labels"][batch["labels"] != -100], stored, flat.decoder.weight, args.bias, args)
        total, n = total + loss.sum().item(), n + loss.numel()
    return total / n


def show(dataset, name, level, rules):
    for rule, points in rules.items():
        for p in points:
            extra = f"  token exit {p['token_exit']:.2f}" if "token_exit" in p else ""
            print(f"[{dataset}] {name} {level:8s} {rule:6s} thr {p['threshold']:9.4g}  loops {p['loops']:.2f} ({p['blocks']:.1f} blocks)  loss {p['loss']:.3f}"
                  f"  fixed {p['fixed_loss']:.3f}  gain {p['gain']:+.3f}{extra}")


def parse_model(spec):
    name, _, rest = spec.partition("=")
    parts = (rest.split(",") + ["", ""])[:3]
    return name, parts[0], parts[1] or None, parts[2] or None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="NAME=checkpoint[,token_head.pt[,sequence_head.pt]]")
    p.add_argument("--flat", nargs="*", default=[], help="NAME=checkpoint of a flat (no-loop) model, for its loss at 8 blocks")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--head_thresholds", type=float, nargs="+", default=[0.5, 0.8, 0.95], help="stop when the head's probability reaches this")
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.4, 0.8], help="free signals stop below this quantile of the observed signal")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--out", default="runs/halting_eval.json")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    distill.configure_cuda()
    args = distill.DistillArguments(output_dir="runs/_halting_eval", report_to="none")

    batches = heldout_batches(a.samples, a.seq_len, a.batch_size)
    teacher = load_teacher()
    args.bias = teacher.decoder.bias.float()

    results = {k: {} for k in batches}
    for spec in a.models:
        name, path, token_head, seq_head = parse_model(spec)
        student = BortForMaskedLM.from_pretrained(path).cuda().eval()
        token_head = torch.load(token_head, map_location="cpu") if token_head else None
        seq_head = torch.load(seq_head, map_location="cpu") if seq_head else None
        for k, b in batches.items():
            data = collect(student, teacher, args.bias, b, args)
            print(f"[{k}] {name} fixed depth 1..8: " + " ".join(f"{x:.3f}" for x in data["loss"].mean(1).tolist()))
            seq = sequence_level(data, seq_head, a.head_thresholds, a.quantiles)
            tok_r = token_level(student, data, args, token_head, a.head_thresholds, a.quantiles)
            results[k][name] = {"sequence": seq, "token": tok_r}
            o = seq["oracle"]
            print(f"[{k}] {name} oracle(seq, hindsight) loops {o['loops']:.2f} loss {o['loss']:.3f} (fixed {o['fixed_loss']:.3f})")
            show(k, name, "sequence", seq["rules"])
            show(k, name, "token", tok_r["rules"])
            del data
        del student
        torch.cuda.empty_cache()
    for spec in a.flat:
        name, _, path = spec.partition("=")
        flat = FlatForMaskedLM.from_pretrained(path).cuda().eval()
        for k, b in batches.items():
            cache = []
            for batch in b:
                batch = {x: v.cuda() for x, v in batch.items()}
                cache.append((batch, make_targets(teacher_hidden(teacher, batch), flat.decoder.weight, args.bias, args.temperature, 0, 1.0)))
            results[k][name] = {"flat_loss_8_blocks": flat_loss(flat, cache, args)}
            print(f"[{k}] {name} (no loop, 8 blocks): loss {results[k][name]['flat_loss_8_blocks']:.3f}")
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()

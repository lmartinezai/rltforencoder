"""Static cost of every model: parameters, weight bytes and FLOPs per sequence at every loop depth and several sequence lengths. CPU only.

    uv run python src/model_cost.py --out results/hardware

FLOPs are counted by torch's FlopCounterMode on a real forward (matrix-multiply work: what tensor cores, NPUs and the CPU's matrix units execute; softmax,
norms and activations are not counted, a few percent). `encoder` = embeddings through the final norm, the part recurrence repeats; `total` adds the masked-LM
head (dense + decoder) at the masked 30% of the positions, as in every evaluation here (the head is the same width for every model, so it only shifts the totals).
Parameters: `body` = everything except the embeddings and the prediction head (the part that does the work and has to sit in fast memory); a looped model's
body counts each shared block once. Weight bytes are for the body at fp16 and int8; the embedding table is a lookup (a few rows per token), so it can live
in slow memory and is listed separately."""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from crossmodel_eval import load_model
from hw_models import available

SEQS = (128, 512)
MASK = 0.3


def parameter_counts(model):
    base = model.base_model
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    embedding = sum(p.numel() for n, p in base.named_parameters() if "embeddings" in n and "LayerNorm" not in n and "norm" not in n)
    body = sum(p.numel() for p in base.parameters()) - embedding
    return {"total_M": total / 1e6, "embedding_M": embedding / 1e6, "body_M": body / 1e6, "head_M": (total - sum(p.numel() for p in base.parameters())) / 1e6}


def encoder_flops(model, seq, depth=None):
    ids = torch.randint(5, model.config.vocab_size, (1, seq))
    kwargs = {} if depth is None else {"recurrence_depth": depth}
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)
    return counter.get_total_flops() / 1e9


def head_flops(model, seq):
    h, v = model.config.hidden_size, model.config.vocab_size
    return int(seq * MASK) * (2 * h * h + 2 * h * v) / 1e9


def rows_for(name, model, looped):
    counts = parameter_counts(model)
    model.requires_grad_(False)  # the module tracker of FlopCounterMode needs grad-free parameters under no_grad
    out = []
    for seq in SEQS:
        for depth in (range(1, 9) if looped else (None,)):
            enc = encoder_flops(model, seq, depth)
            out.append({"model": name, "depth": depth or "-", "seq": seq, "encoder_GFLOP": enc, "total_GFLOP": enc + head_flops(model, seq), **counts,
                        "body_MB_fp16": counts["body_M"] * 2, "body_MB_int8": counts["body_M"]})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/hardware")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    torch.set_num_threads(4)
    rows = []
    for name, (kind, path, _, looped) in available().items():
        model, _ = load_model(kind, path, "cpu")
        rows += rows_for(name, model, looped)
        r = rows[-1]
        print(f"[{name}] body {r['body_M']:.1f}M, embeddings {r['embedding_M']:.1f}M, total {r['total_M']:.1f}M; encoder {rows[-1]['encoder_GFLOP']:.0f} GFLOP at seq {r['seq']}", flush=True)
        del model
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "static.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'static.csv'}")


if __name__ == "__main__":
    main()

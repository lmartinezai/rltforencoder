"""Train the halting heads of the looped models, efficiently, and put them into the checkpoints.

    uv run python src/halting_train.py --models a_s1=runs/small/a_s1/checkpoint-3072 A8=runs/arm_a_merge/checkpoint-12208 ...

After `src/halting_data.py` has stored the teacher targets once, each model costs one pass of the frozen network over 4,096 + 2 x 512 stored sequences
(src/halting_features.py: the loop is run once to depth 8 and every iteration's state is decoded, not eight separate forwards) and a few seconds of fitting:

  sequence head   Linear(hidden, 1) on a sequence's mean state at each depth: "is the sequence's mean loss at this depth within 0.02 of its best depth".
                  This is exactly BortModel.halting_head (raw-state weights, the standardisation folded in): it is written into <checkpoint>-halting.
  token head      the same on single masked tokens ("within 0.1 of that token's best depth"), saved next to it as halting_head_token.pt.

The sequence head is scored exactly on both held-out sets (a sequence stopped after k iterations has the loss of the depth-k forward): the gain over the fixed
depth with the same mean loop count, best over thresholds and its mean over operating points with 2.5 <= loops <= 7.5, next to the hindsight ceiling no head
can beat. The token head is scored by accuracy only: what token-level halting does to a real forward pass is measured by src/halting_eval.py, and it hurt."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

import halting
import halting_features as hf
from arch import BortForMaskedLM
from halting_fit import (
    feature_matrix,
    first_depth,
    fit,
    gain_summary,
    oracle_points,
    policy_point,
    targets,
)

SEQ_TOLERANCE, TOKEN_TOLERANCE = 0.02, 0.1
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def extract_split(checkpoint, out, store, n, tokens, batch_size=16):
    """Features of one stored split for one model (skipped if already extracted)."""
    out = Path(out)
    if (out / "meta.json").exists() and any(out.glob("shard-*.pt")):
        return
    hf.main(["--student", str(checkpoint), "--out", str(out), "--source", "store", "--store", str(store), "--samples", str(n), "--tokens", str(tokens),
             "--batch_size", str(batch_size)])


def fit_sequence_head(train, epochs=10, seed=0, lr=3e-3):
    """-> (raw-state head dict, standardised head module, mean, std)."""
    x = feature_matrix(train, ("state",))
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    head = fit((x - mean) / std, targets(train["seq_loss"], "v1", SEQ_TOLERANCE), False, epochs, seed, lr)
    weight, bias = halting.fold_standardisation(head.net, mean, std)
    saved = {"weight": weight.unsqueeze(0), "bias": bias.reshape(1), "level": "sequence", "label": "v1", "tolerance": SEQ_TOLERANCE}
    return saved, head, mean, std


def token_rows(data):
    """tok_state [B, D, K, H], tok_loss [B, D, K] -> features [B*D*K, H] and the can-stop labels [B*D*K], rows ordered (sequence, depth, token)."""
    loss = data["tok_loss"]
    label = (loss <= loss.min(1, keepdim=True).values + TOKEN_TOLERANCE).float()
    return data["tok_state"].reshape(-1, data["tok_state"].shape[-1]), label.reshape(-1)


def standardise_half(x, mean, std, chunk=65536):
    """(x - mean) / std stored in half precision, built in chunks (a full float32 copy of the token features is several GB)."""
    out = torch.empty(x.shape, dtype=torch.float16)
    for i in range(0, len(x), chunk):
        out[i:i + chunk] = ((x[i:i + chunk].float() - mean) / std).half()
    return out


def fit_token_head(train, held_out, epochs=3, seed=0, lr=3e-3):
    """-> (raw-state head dict, held-out accuracy, the share of held-out rows labelled "can stop")."""
    x, y = token_rows(train)  # half-precision features: they are only ever converted a batch at a time
    sample = x[:: max(1, len(x) // 100_000)].float()
    mean, std = sample.mean(0), sample.std(0).clamp_min(1e-6)
    head = fit(standardise_half(x, mean, std), y, False, epochs, seed, lr, batch=1024, use_cpu=not torch.cuda.is_available()).cpu()  # on the GPU when there is one
    del x
    weight, bias = halting.fold_standardisation(head.net, mean, std)
    xh, yh = token_rows(held_out)
    with torch.no_grad():
        logits = head(standardise_half(xh, mean, std))["logits"]
    accuracy = ((logits > 0) == (yh > 0.5)).float().mean().item()
    return {"weight": weight.unsqueeze(0), "bias": bias.reshape(1), "level": "token", "label": "v1", "tolerance": TOKEN_TOLERANCE}, accuracy, yh.mean().item()


def score_sequence_head(saved, data):
    """Exact held-out scores of a raw-state sequence head: fixed-depth losses, the hindsight ceiling, the head's gain, and its operating points."""
    loss, n_voters = data["seq_loss"], data["n_voters"].float()
    fixed = ((loss * n_voters[:, None]).sum(0) / n_voters.sum()).tolist()
    ceiling = gain_summary(oracle_points(loss, n_voters), fixed)
    logits = data["seq_state"].float() @ saved["weight"].squeeze(0) + saved["bias"]
    probability = logits.sigmoid()
    points, operating = [], []
    for t in torch.linspace(0.05, 0.99, 30).tolist():
        value, loops = policy_point(loss, n_voters, first_depth(probability >= t))
        points.append((loops, value))
    for t in THRESHOLDS:
        value, loops = policy_point(loss, n_voters, first_depth(probability >= t))
        operating.append({"threshold": t, "loops": loops, "loss": value, "fixed_loss": halting.interpolate(fixed, loops), "gain": halting.interpolate(fixed, loops) - value})
    best, mean = gain_summary(points, fixed)
    return {"fixed": fixed, "ceiling_best": ceiling[0], "ceiling_mean": ceiling[1], "head_best_gain": best, "head_mean_gain": mean, "operating_points": operating}


def bake(checkpoint, head_path, out_dir):
    """Copy of the checkpoint with the trained halting head in place of the untrained one. Nothing else changes: the head is not used by the forward pass
    (checked: identical hidden states before and after)."""
    model = BortForMaskedLM.from_pretrained(checkpoint).eval()
    ids = torch.randint(3, model.config.vocab_size, (1, 32))
    with torch.no_grad():
        before = model.model(input_ids=ids, recurrence_depth=3).last_hidden_state
    halting.load_halting_head(model, head_path)
    model.save_pretrained(out_dir)
    again = BortForMaskedLM.from_pretrained(out_dir).eval()
    saved = torch.load(head_path, map_location="cpu")
    with torch.no_grad():
        after = again.model(input_ids=ids, recurrence_depth=3).last_hidden_state
    if not torch.equal(before, after) or not torch.equal(again.model.halting_head.weight.detach(), saved["weight"]):
        raise RuntimeError("baking changed the model or lost the head")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="NAME=CHECKPOINT_DIR (a looped Bort)")
    p.add_argument("--store", default="stores/halting")
    p.add_argument("--feats", default="feats/halting")
    p.add_argument("--out", default="results/halting_v2")
    p.add_argument("--train", type=int, default=4096)
    p.add_argument("--eval", type=int, default=512)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--inprocess", action="store_true", help="run every model in this process (default: one fresh process per model, memory is returned in between)")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(a.models) > 1 and not a.inprocess:
        for spec in a.models:
            subprocess.run([sys.executable, __file__, "--models", spec, "--store", a.store, "--feats", a.feats, "--out", a.out, "--train", str(a.train),
                            "--eval", str(a.eval), "--tokens", str(a.tokens)], check=True)
        return
    for spec in a.models:
        name, _, checkpoint = spec.partition("=")
        if (out / f"{name}.json").exists():
            print(f"[{name}] already trained, skipped")
            continue
        feats = Path(a.feats) / name
        extract_split(checkpoint, feats / "train", f"{a.store}/fineweb/train", a.train, a.tokens)
        extract_split(checkpoint, feats / "eval_fineweb", f"{a.store}/fineweb/eval", a.eval, a.tokens)
        extract_split(checkpoint, feats / "eval_arxiv", f"{a.store}/arxiv/eval", a.eval, a.tokens)
        train = hf.load_shards(feats / "train")[0]
        held = {"fineweb": hf.load_shards(feats / "eval_fineweb")[0], "arxiv": hf.load_shards(feats / "eval_arxiv")[0]}
        sequence_head, _, _, _ = fit_sequence_head(train)
        token_head, token_accuracy, stop_rate = fit_token_head(train, held["fineweb"])
        torch.save(sequence_head | {"student": str(checkpoint)}, feats / "halting_head.pt")
        torch.save(token_head | {"student": str(checkpoint)}, feats / "halting_head_token.pt")
        result = {"checkpoint": str(checkpoint), "train_sequences": len(train["seq_loss"]), "token_head": {"heldout_accuracy": token_accuracy, "can_stop_rate": stop_rate},
                  "sequence_head": {key: score_sequence_head(sequence_head, data) for key, data in held.items()}}
        baked = Path(str(checkpoint) + "-halting")
        bake(checkpoint, feats / "halting_head.pt", baked)
        torch.save(token_head | {"student": str(checkpoint)}, baked / "halting_head_token.pt")
        result["baked_checkpoint"] = str(baked)
        (out / f"{name}.json").write_text(json.dumps(result, indent=1))
        s = result["sequence_head"]
        print(f"[{name}] sequence head gain over fixed depth (best/mean): fineweb {s['fineweb']['head_best_gain']:+.4f}/{s['fineweb']['head_mean_gain']:+.4f}"
              f" (ceiling {s['fineweb']['ceiling_best']:+.4f}), arxiv {s['arxiv']['head_best_gain']:+.4f}/{s['arxiv']['head_mean_gain']:+.4f}"
              f" (ceiling {s['arxiv']['ceiling_best']:+.4f}); token head accuracy {token_accuracy:.3f} at {stop_rate:.2f} base rate; baked into {baked}", flush=True)


if __name__ == "__main__":
    main()

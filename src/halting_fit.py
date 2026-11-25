"""Fit sequence-level halting heads on stored feature shards (src/halting_features.py) and score them as exit policies. Seconds per head.

    uv run python src/halting_fit.py --train feats/a_train feats/a_train_220 --eval arxiv=feats/a_eval_arxiv fineweb=feats/a_eval_fineweb

A head sees one sequence at depth d (never later depths: a stopping rule cannot look ahead) and decides whether to run another iteration.
Policies are scored on held-out shards, exactly: a sequence stopped after k iterations has the loss of the depth-k forward (tested), so the
loss of a policy is read off the stored per-depth sequence losses (token-weighted by the number of masked tokens) and its cost is the mean loop count.

Labels
  v1      "depth d is within tol of the best depth of the sequence": the first head's label. It compares with EARLIER depths, which a stopping
          rule can no longer choose, so past the best depth it says "keep going" for a state that should stop.
  future  "no LATER depth is better by more than tol": stopping here is the best of what is still available. Causal.
  gain    regress the improvement still to come, max(0, L_d - min_{j>d} L_j), and stop when it is predicted to be below a price lambda per loop.
Features: state (the sequence's mean state), scalars (output KL to the previous depth, state change, entropy, top probability), depth (one-hot).
Reported per variant and held-out set: the best and mean GAIN over the fixed-depth model of the same mean loop count, over operating points with
2.5 <= loops <= 7.5, next to the ceiling: the same for the hindsight-optimal assignment (per-sequence argmin of loss + lambda x loops), which no
head can beat. The "best gain" is the maximum over thresholds on the held-out set itself (the threshold is not chosen on separate data), so it
is optimistic; the mean gain is not. Both favour the heads, which is the conservative direction for the finding that halting does not pay."""
import argparse
import itertools
import json
import sys

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from halting import first_true, interpolate
from halting_features import load_shards

D = 8
LABELS = {"v1": 0.02, "future0": 0.0, "future": 0.005, "gain": None}  # name -> tolerance
FEATURES = {"S": ("state",), "SD": ("state", "depth"), "CD": ("scalars", "depth"), "SCD": ("state", "scalars", "depth")}


def feature_matrix(data, parts):
    """[B*D, F] features, rows ordered (sequence, depth)."""
    B = data["seq_loss"].shape[0]
    cols = []
    if "state" in parts:
        cols.append(data["seq_state"].float().reshape(B * D, -1))
    if "scalars" in parts:
        cols.append(torch.stack([data[f"seq_{k}"] for k in ("kl", "change", "entropy", "maxp")], -1).reshape(B * D, 4))
    if "depth" in parts:
        cols.append(torch.eye(D).repeat(B, 1))
    return torch.cat(cols, 1)


def targets(loss, label, tolerance):
    """loss [B, D] (sequence mean losses) -> per (sequence, depth) target [B, D]: a stop label in {0,1} or the future gain."""
    future_min = torch.cat([loss[:, 1:].flip(1).cummin(1).values.flip(1), torch.full((len(loss), 1), torch.inf)], 1)  # min_{j>d} L_j (inf at D)
    if label == "gain":
        return (loss - future_min).clamp_min(0).nan_to_num(posinf=0.0)
    if label == "v1":
        return (loss <= loss.min(1, keepdim=True).values + tolerance).float()
    return (loss <= future_min + tolerance).float()


class Rows(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


class Head(nn.Module):
    """The probe: Linear(F, 1) (hidden=0, what arch.py has) or Linear-GELU-Linear, output layer zero-initialised. BCE for stop labels, Huber for the gain."""

    def __init__(self, features, regression, hidden=0):
        super().__init__()
        self.net = nn.Linear(features, 1) if not hidden else nn.Sequential(nn.Linear(features, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.regression = regression
        last = self.net if not hidden else self.net[-1]
        nn.init.zeros_(last.weight), nn.init.zeros_(last.bias)

    def forward(self, x, y=None):
        out = self.net(x.float()).squeeze(-1)  # features may be stored in half precision
        if y is None:
            return {"logits": out}
        loss = nn.functional.huber_loss(out, y * 100, delta=1.0) if self.regression else nn.functional.binary_cross_entropy_with_logits(out, y)
        return {"loss": loss, "logits": out}


def fit(x, y, regression, epochs, seed, lr, hidden=0, batch=256, use_cpu=True):
    args = TrainingArguments(output_dir="runs/_fit", num_train_epochs=epochs, per_device_train_batch_size=batch, learning_rate=lr, weight_decay=0.0,
                             lr_scheduler_type="cosine", warmup_steps=20, logging_strategy="no", save_strategy="no", eval_strategy="no",
                             report_to="none", disable_tqdm=True, seed=seed, use_cpu=use_cpu, optim="adamw_torch")
    head = Head(x.shape[1], regression, hidden)
    Trainer(model=head, args=args, train_dataset=Rows(x, y.reshape(-1))).train()
    return head.eval()


def first_depth(stop):
    """[B, D] bool -> 1-based first depth where True (the last if never)."""
    return first_true(stop, dim=1)


def policy_point(loss, n_voters, depth):
    """Token-weighted loss and mean loops of exit depths [B]."""
    chosen = loss[torch.arange(len(loss)), depth - 1]
    return (chosen * n_voters).sum().item() / n_voters.sum().item(), depth.float().mean().item()


def gain_summary(points, fixed):
    """[(loops, loss)] -> best and mean gain over the fixed-depth model at the same loops, restricted to 2.5 <= loops <= 7.5."""
    gains = [interpolate(fixed, loops) - loss for loops, loss in points if 2.5 <= loops <= 7.5]
    return (max(gains), sum(gains) / len(gains)) if gains else (float("nan"), float("nan"))


def oracle_points(loss, n_voters):
    """The hindsight frontier: per-sequence argmin of (weighted loss + lambda x loops) for a sweep of lambda."""
    w = n_voters / n_voters.sum()
    cost = loss * w[:, None]  # [B, D]
    pts = []
    for lam in torch.logspace(-6, -1, 60).tolist():
        depth = (cost + lam * torch.arange(1, D + 1) / len(loss)).argmin(1) + 1
        pts.append(policy_point(loss, n_voters, depth))
    return [(loops, loss_) for loss_, loops in pts]


def evaluate_head(head, data, parts, mean, std, regression):
    x = (feature_matrix(data, parts) - mean) / std
    with torch.no_grad():
        score = head(x)["logits"].reshape(-1, D)
    loss, n = data["seq_loss"], data["n_voters"].float()
    if regression:  # stop when the predicted remaining gain (in 1/100 loss units) is below lambda
        thresholds = torch.quantile(score.flatten(), torch.linspace(0.02, 0.98, 25)).tolist()
        pts = [policy_point(loss, n, first_depth(score <= t)) for t in thresholds]
    else:
        pts = [policy_point(loss, n, first_depth(score.sigmoid() >= t)) for t in torch.linspace(0.05, 0.99, 30).tolist()]
    return [(loops, loss_) for loss_, loops in pts]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="+", required=True, help="shard directories")
    p.add_argument("--eval", nargs="+", required=True, help="NAME=shard_directory")
    p.add_argument("--labels", nargs="+", default=list(LABELS))
    p.add_argument("--features", nargs="+", default=list(FEATURES))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden", type=int, default=0, help="0: a linear head (arch.py's); N: one hidden layer of N units")
    p.add_argument("--max_train", type=int, default=0, help="use only this many training sequences (learning curves)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/halting_fit.json")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(a.seed)

    train = {}
    for d in a.train:
        shard, _ = load_shards(d)
        for k, v in shard.items():
            train.setdefault(k, []).append(v)
    train = {k: torch.cat(v) for k, v in train.items()}
    if a.max_train:
        train = {k: v[:a.max_train] for k, v in train.items()}
    evals = {name: load_shards(path)[0] for name, _, path in (e.partition("=") for e in a.eval)}
    print(f"[fit] {len(train['seq_loss'])} training sequences; eval " + ", ".join(f"{n}: {len(d['seq_loss'])}" for n, d in evals.items()))

    results = {}
    for name, data in evals.items():
        fixed = ((data["seq_loss"] * data["n_voters"][:, None]).sum(0) / data["n_voters"].sum()).tolist()
        ceiling = oracle_points(data["seq_loss"], data["n_voters"].float())
        results[name] = {"fixed": fixed, "ceiling": gain_summary(ceiling, fixed)}
        print(f"[{name}] fixed depth 1..8: " + " ".join(f"{x:.4f}" for x in fixed) + f"   ceiling (hindsight) gain best {results[name]['ceiling'][0]:+.4f} mean {results[name]['ceiling'][1]:+.4f}")

    for label, feats in itertools.product(a.labels, a.features):
        parts, regression = FEATURES[feats], label == "gain"
        x_raw = feature_matrix(train, parts)
        mean, std = x_raw.mean(0), x_raw.std(0).clamp_min(1e-6)
        y = targets(train["seq_loss"], label, LABELS[label])
        head = fit((x_raw - mean) / std, y, regression, a.epochs, a.seed, a.lr, a.hidden)
        line = f"[{label:7s} {feats:3s}] "
        for name, data in evals.items():
            pts = evaluate_head(head, data, parts, mean, std, regression)
            best, avg = gain_summary(pts, results[name]["fixed"])
            results[name][f"{label}/{feats}"] = {"best_gain": best, "mean_gain": avg, "points": pts}
            line += f"{name}: gain best {best:+.4f} mean {avg:+.4f}   "
        print(line)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()

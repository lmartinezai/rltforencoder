"""Aggregate the small-model study (looped arms, flat arm, iso-FLOP flat control; one run per seed) into durable results: a table and curves.

    uv run python src/small_report.py --runs runs/small runs/small_c runs/iso_flop --out results/small_study

Each run directory name is `<arm>_s<seed>` (a = looped + gated merge, b = looped without merge, c = flat with the same 4 blocks, d = flat with 10 blocks:
the iso-FLOP control, as many block passes as the looped model at depth 4; e = looped + gated merge in RLT's form, which re-injects the
prelude output every iteration); its numbers are read from the
`trainer_state.json` of its last checkpoint (the Trainer's complete log history), so nothing is parsed from console text. Writes
  curves.csv     one row per (arm, seed, step, metric): every evaluation and every logged training loss
  summary.md     the same as a markdown table
  summary.json   final evaluation loss per arm and depth: mean, sample std over seeds, min, max, and the per-seed values
The held-out set is the stored eval split (256 FineWeb hold-out sequences, one fixed mask pattern). Every arm starts from copied teacher layers
(`--init teacher_layers`), which leaves nothing random in the flat arms and only the merge gate in arm a: the spread over seeds is the spread over data
order and depth schedule, not over initialisations."""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = {"a": "looped + gated merge", "b": "looped, no merge", "e": "looped + gated merge, input injection (RLT form)", "c": "flat (no loop)",
        "d": "flat, 10 blocks (iso-FLOP control)"}
FLAT = {"c": "--", "d": ":"}  # arms without a loop (their only evaluation is eval_R1_loss) and their line styles


def last_state(run):
    states = sorted(Path(run).glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    return json.loads(states[-1].read_text()) if states else None


def collect(run_dirs):
    """-> {(arm, seed): log_history} for every finished run directory under run_dirs."""
    runs = {}
    for root in run_dirs:
        for run in sorted(Path(root).glob("[abcde]_s[0-9]*")):
            m = re.fullmatch(r"([abcde])_s(\d+)", run.name)
            if m and run.is_dir() and (run / "DONE").exists() and (state := last_state(run)):
                runs[(m[1], int(m[2]))] = state["log_history"]
    return runs


def series(history, key):
    return [(h["step"], h[key]) for h in history if key in h]


def summarise(runs):
    final = defaultdict(lambda: defaultdict(dict))
    for (arm, seed), history in runs.items():
        last = {}  # the Trainer logs every depth's evaluation as its own entry: take each metric at its own last step
        for h in history:
            for key, value in h.items():
                if re.fullmatch(r"eval_R\d+_loss", key) and h["step"] >= last.get(key, (-1, 0))[0]:
                    last[key] = (h["step"], value)
        for key, (_, value) in last.items():
            final[arm][key][seed] = value
    summary = {}
    for arm, metrics in final.items():
        summary[arm] = {}
        for key, per_seed in sorted(metrics.items()):
            v = list(per_seed.values())
            summary[arm][key] = {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else None, "min": min(v), "max": max(v),
                                 "per_seed": dict(sorted(per_seed.items())), "n": len(v)}
    return summary


def write_csv(runs, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "step", "metric", "value"])
        for (arm, seed), history in sorted(runs.items()):
            for h in history:
                for key, value in h.items():
                    if re.fullmatch(r"eval_R\d+_loss", key) or key == "loss":
                        w.writerow([arm, seed, h["step"], key, value])


def smooth(points, window):
    """Mean over a trailing window of logged steps (training loss is noisy because the depth changes every step)."""
    out = []
    for i in range(len(points)):
        chunk = [v for _, v in points[max(0, i - window + 1): i + 1]]
        out.append((points[i][0], sum(chunk) / len(chunk)))
    return out


def table(summary):
    """Markdown table of the final held-out loss: mean +- sample std over seeds, per arm and depth."""
    depths = sorted({int(k.split("R")[1].split("_")[0]) for arm in summary for k in summary[arm]})
    lines = ["| arm | seeds | " + " | ".join(f"depth {d}" for d in depths) + " |", "|---|---:|" + "---:|" * len(depths)]
    for arm in sorted(summary):
        cells = []
        for d in depths:
            s = summary[arm].get(f"eval_R{d}_loss")
            cells.append("-" if not s else f"{s['mean']:.4f}" + (f" +- {s['std']:.4f}" if s["std"] is not None else ""))
        lines.append(f"| {ARMS[arm]} | {summary[arm]['eval_R1_loss']['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["runs/small"], help="directories holding <arm>_s<seed> run directories")
    p.add_argument("--out", default="results/small_study")
    a = p.parse_args(argv)
    runs = collect(a.runs)
    if not runs:
        raise SystemExit("no finished runs (a run is finished when its directory has a DONE file and a checkpoint)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(runs, out / "curves.csv")
    summary = summarise(runs)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.md").write_text("Final held-out distillation loss (256 FineWeb hold-out sequences), mean +- sample std over seeds. A flat arm has no loop: one "
                                     "value.\nThe 10-block flat arm is the iso-FLOP control: as many block passes as a looped arm at depth 4.\n\n" + table(summary))
    print(f"{len(runs)} runs: " + ", ".join(f"{arm}{seed}" for arm, seed in sorted(runs)))
    for arm in sorted(summary):
        for key, s in summary[arm].items():
            spread = f" +- {s['std']:.4f}" if s["std"] is not None else ""
            print(f"  {ARMS[arm]:24s} {key:13s} {s['mean']:.4f}{spread}  (n={s['n']}, range {s['min']:.4f}-{s['max']:.4f})")


if __name__ == "__main__":
    main()

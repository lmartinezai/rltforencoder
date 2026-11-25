"""Tables from src/zeroshot_nli.py's raw scores.

    uv run python src/nli_report.py --raw results/zeroshot_nli/raw.json --out results/zeroshot_nli

Accuracy of each model on the GLUE dev sets with the two fixed prompts, averaged over the two prompts (neither is picked on the dev set), raw and
calibrated, per task: MNLI (matched and mismatched pooled by their sizes), QNLI, RTE. The depth columns for a looped model are the pre-declared depths 1, 4
and 8 (not a best-on-dev depth); every depth is in `accuracy.csv`. The floor is the majority-class rate; `collapse` is the average share of
predictions on the single most predicted label (1/3 or 1/2 is balanced, 1.0 means the model says the same thing every time)."""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from names import REFERENCES

TASK_KEYS = {"MNLI": ("mnli/validation_matched", "mnli/validation_mismatched"), "QNLI": ("qnli/validation",), "RTE": ("rte/validation",)}
GROUPS = {r"a_s\d": "A' looped + gated merge (4 blocks)", r"b_s\d": "B' looped, no merge (4 blocks)", r"c_s\d": "C' flat (4 blocks)",
          r"d_s\d": "D' flat, 10 blocks (iso-FLOP control)", r"e_s\d": "E' looped + gated merge, input injection (4 blocks)",
          r"A8": "A looped + gated merge (8 blocks)", r"B8": "B looped, no merge (8 blocks)", r"C8": "C flat (8 blocks)"}
DEPTHS_SHOWN = (1, 4, 8)


def group_of(name):
    for pattern, label in GROUPS.items():
        if re.fullmatch(pattern, name):
            return label
    return REFERENCES.get(name)


def task_value(per_split, task, key, templates=("T1", "T2")):
    """Mean over the templates of the task's metric, pooling MNLI's two dev sets by size."""
    total = weight = 0.0
    for split in TASK_KEYS[task]:
        cells = per_split[split]
        n = cells[templates[0]]["n"]
        total += n * statistics.mean(cells[t][key] for t in templates)
        weight += n
    return total / weight


def majority(per_split, task):
    total = sum(per_split[s]["T1"]["n"] * per_split[s]["T1"]["majority"] for s in TASK_KEYS[task])
    return total / sum(per_split[s]["T1"]["n"] for s in TASK_KEYS[task])


def collect(raw):
    """-> {label: {depth: {task: {metric: [value per seed]}}}} for the models that belong to a group or are references."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for name, depths in raw.items():
        label = group_of(name)
        if label is None:
            continue
        for depth, per_split in depths.items():
            for task in TASK_KEYS:
                for key in ("raw", "calibrated", "raw_top_label_share", "calibrated_top_label_share"):
                    out[label][int(depth)][task][key].append(task_value(per_split, task, key))
    return out


def fmt(values):
    return f"{100 * statistics.mean(values):.1f}" + (f" +- {100 * statistics.stdev(values):.1f}" if len(values) > 1 else "")


def tables(raw):
    sample = next(iter(raw.values()))
    floor = {task: majority(sample["1"], task) for task in TASK_KEYS}
    data = collect(raw)
    lines = []
    for key, title in (("calibrated", "calibrated"), ("raw", "raw")):
        lines += [f"**Zero-shot accuracy (%), {title} label-word scores, mean of the two prompts; mean +- sample std over 3 seeds where there are 3.**", "",
                  "| model | depth | MNLI | QNLI | RTE | mean | collapse |", "|---|---|---:|---:|---:|---:|---:|",
                  "| majority class (the floor) | - | " + " | ".join(f"{100 * floor[t]:.1f}" for t in TASK_KEYS) + f" | {100 * statistics.mean(floor.values()):.1f} | - |"]
        for label in [*GROUPS.values(), *REFERENCES.values()]:
            if label not in data:
                continue
            depths = sorted(data[label])
            for d in (depths if len(depths) == 1 else [x for x in DEPTHS_SHOWN if x in depths]):
                row = data[label][d]
                accs = [statistics.mean(row[t][key]) for t in TASK_KEYS]
                collapse = statistics.mean(statistics.mean(row[t][f"{key}_top_label_share"]) for t in TASK_KEYS)
                lines.append(f"| {label} | {d if len(depths) > 1 else '-'} | " + " | ".join(fmt(row[t][key]) for t in TASK_KEYS)
                             + f" | {100 * statistics.mean(accs):.1f} | {collapse:.2f} |")
        lines.append("")
    return "\n".join(lines)


YLIM = {"MNLI": (30, 50), "QNLI": (44, 58), "RTE": (44, 62)}  # fixed per task so that noise-sized wiggles look like noise
DEV_SIZE = {"MNLI": 19647, "QNLI": 5463, "RTE": 277}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="results/zeroshot_nli/raw.json")
    p.add_argument("--out", default="results/zeroshot_nli")
    a = p.parse_args(argv)
    raw = json.loads(Path(a.raw).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "depth", "task_split", "template", "raw", "calibrated", "raw_top_label_share", "calibrated_top_label_share", "majority", "n"])
        for name, depths in raw.items():
            for depth, per_split in depths.items():
                for split, templates in per_split.items():
                    for template, c in templates.items():
                        w.writerow([name, depth, split, template, c["raw"], c["calibrated"], c["raw_top_label_share"], c["calibrated_top_label_share"], c["majority"], c["n"]])
    text = tables(raw)
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

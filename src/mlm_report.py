"""Tables from src/crossmodel_eval.py's raw scores (bits per byte of text, lower is better).

    uv run python src/mlm_report.py --raw results/mlm_crossmodel/raw.json --out results/mlm_crossmodel

Writes `bpb.csv` (every model, depth and dataset), `summary.md` (per model: depth 1, the best depth, depth 8, and the flat / reference models, mean +- sample
std over seeds). The "best" depth is the one with the lowest mean BPB on these same texts: it is picked on the data it is reported on, so it is the
looped model's most favourable depth, not a held-out choice (neighbouring depths differ by about 0.01 BPB)."""
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from names import REFERENCES

DATASETS = ("fineweb-holdout", "arxiv", "wikipedia", "python-code")
FAMILIES = {  # label -> (members, kind)
    "small a": "A' looped + gated merge", "small b": "B' looped, no merge", "small c": "C' flat (no loop)",
    "small d": "D' flat, 10 blocks (the FLOPs of B' at depth 4)", "small e": "E' looped + gated merge, input injection (RLT form)",
    "8 A": "A looped + gated merge (8 blocks, 200M tokens)", "8 B": "B looped, no merge (8 blocks)", "8 C": "C flat (8 blocks)",
}
GROUPS = {  # model name pattern -> group key
    r"a_s\d": "small a", r"b_s\d": "small b", r"c_s\d": "small c", r"d_s\d": "small d", r"e_s\d": "small e", r"A8": "8 A", r"B8": "8 B", r"C8": "8 C",
}


def bpb(cell):
    return cell["nats"] / math.log(2) / cell["bytes"]


def group_of(name):
    for pattern, key in GROUPS.items():
        if re.fullmatch(pattern, name):
            return key
    return None


def table_rows(raw):
    """-> {group or reference: {depth: {dataset: [bpb per seed]}}}."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name, depths in raw.items():
        key = group_of(name) or (name if name in REFERENCES else None)
        if key is None:
            continue
        for depth, datasets in depths.items():
            for dataset, cell in datasets.items():
                out[key][int(depth)][dataset].append(bpb(cell))
    return out


def mean_over_datasets(depth_cells):
    return statistics.mean(statistics.mean(depth_cells[d]) for d in DATASETS)


def fmt(values):
    return f"{statistics.mean(values):.4f}" + (f" +- {statistics.stdev(values):.4f}" if len(values) > 1 else "")


def summary(rows):
    lines = ["Masked-LM bits per byte of text (lower is better); mean +- sample std over seeds where there is more than one. Texts: 128 x 1000 characters",
             "per source, partition masking at a 12.5% rate (every token scored once). `best` = the depth with the lowest mean on these same texts.", "",
             "| model | seeds | depth | " + " | ".join(DATASETS) + " | mean |", "|---|---:|---|" + "---:|" * (len(DATASETS) + 1)]

    def line(label, depth_cells, depth):
        cells = [fmt(depth_cells[d]) for d in DATASETS]
        return f"| {label} | {len(depth_cells[DATASETS[0]])} | {depth} | " + " | ".join(cells) + f" | {mean_over_datasets(depth_cells):.4f} |"

    for key, label in {**{k: v for k, v in FAMILIES.items()}, **REFERENCES}.items():
        if key not in rows:
            continue
        depths = rows[key]
        if len(depths) == 1:
            lines.append(line(label, depths[1], "-"))
            continue
        best = min(depths, key=lambda d: mean_over_datasets(depths[d]))
        for d in sorted({1, best, 8}):
            lines.append(line(label, depths[d], f"{d}" + (" (best)" if d == best else "")))
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="results/mlm_crossmodel/raw.json")
    p.add_argument("--out", default="results/mlm_crossmodel")
    a = p.parse_args(argv)
    raw = json.loads(Path(a.raw).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "bpb.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "depth", "dataset", "bits_per_byte", "nats", "tokens", "bytes"])
        for name, depths in raw.items():
            for depth, datasets in depths.items():
                for dataset, cell in datasets.items():
                    w.writerow([name, depth, dataset, f"{bpb(cell):.6f}", f"{cell['nats']:.3f}", cell["tokens"], cell["bytes"]])
    rows = table_rows(raw)
    text = summary(rows)
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

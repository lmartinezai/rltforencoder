"""The 8-block, 200M-token arms (A looped + merge, B looped, C flat; one seed each) on held-out text: the table of src/bench_heldout.py's output.

    uv run python src/arms200m_report.py --bench runs/bench_heldout.json --out results/arms_200m

Input: the final held-out benchmark (512 arXiv articles and 512 FineWeb hold-out sequences x 512 tokens, 78,314 masked positions each), every loop depth.
`loss` is the training objective (0.6 CE + 0.4 T^2 KL at T = 2, against the teacher), `ce` the cross-entropy against the true token, `acc` top-1 accuracy at
the masked positions. Nothing is recomputed; this script only formats the stored numbers and copies them next to the table."""
import argparse
import json
import shutil
from pathlib import Path

ARMS = {"A": "A looped + gated merge", "B": "B looped, no merge", "C": "C flat (no loop)"}
DEPTHS = ("1", "2", "4", "5", "8")


def table(bench, source, metric, fmt):
    lines = ["| arm | " + " | ".join(f"depth {d}" for d in DEPTHS) + " | best |", "|---|" + "---:|" * (len(DEPTHS) + 1)]
    for arm, label in ARMS.items():
        by = bench[source][arm]
        cells = [fmt.format(by[d][metric]) if d in by else "-" for d in DEPTHS]
        pick = (max if metric == "acc" else min)(by, key=lambda d: by[d][metric])
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {fmt.format(by[pick][metric])} (depth {pick}) |")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", default="runs/bench_heldout.json")
    p.add_argument("--out", default="results/arms_200m")
    a = p.parse_args(argv)
    bench = json.loads(Path(a.bench).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.bench, out / "bench_heldout.json")
    lines = ["# 8-block arms, 200M tokens (one seed each): held-out text", "",
             "512 sequences x 512 tokens per source, 78,314 masked positions; a flat arm has no loop (depth 1 only). `best` is chosen on these same sequences.", ""]
    for source in bench:
        lines += [f"## {source} (teacher CE {bench[source]['teacher_ce']:.4f})", ""]
        for metric, title, fmt in (("loss", "distillation loss (lower is better)", "{:.4f}"), ("ce", "cross-entropy against the true token", "{:.4f}"),
                                   ("acc", "top-1 accuracy", "{:.4f}")):
            lines += [f"{title}:", ""] + table(bench, source, metric, fmt) + [""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

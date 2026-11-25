"""Figures for paper/main.tex, as vector PDFs in paper/figures/. Reads results/mlm_crossmodel/bpb.csv only.

    uv run python paper/figures.py

PM-BPB per model and depth is the unweighted mean of the four sources' bits per byte, as in the paper's tables. Every panel has its own
y-scale: the R = 1 values sit far above the rest, so a shared scale would flatten the differences the text discusses.
"""
import csv
import os
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))  # reproduce/reproduce.py points this at its own reproduced results
OUT = Path(os.environ.get("FIGURES_DIR", Path(__file__).resolve().parent / "figures"))
SEEDS = (1, 2, 3)
DEPTHS = list(range(1, 9))
SOURCES = (("fineweb-holdout", "FineWeb"), ("arxiv", "arXiv"), ("wikipedia", "Wikipedia"), ("python-code", "code"))

# categorical slots, fixed order (validated: dataviz validate_palette.js, light surface); text stays in ink colours
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
ARMS = (("a", BLUE, "o", "looped, merge"), ("b", ORANGE, "s", "looped, no merge"), ("e", AQUA, "^", "looped, anchor residual"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,  # Computer Modern, as the body text
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.titlesize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6, "pdf.fonttype": 42, "legend.frameon": False,
})


def load():
    """-> {(model, depth, source): bits per byte}"""
    return {(r["model"], int(r["depth"]), r["dataset"]): float(r["bits_per_byte"])
            for r in csv.DictReader((RESULTS / "mlm_crossmodel/bpb.csv").read_text().splitlines())}


def mean_bpb(raw, model, depth):
    return st.mean(raw[(model, depth, s)] for s, _ in SOURCES)


def depth_axis(ax, depths=DEPTHS):
    ax.set_xticks(depths)
    ax.set_xlim(depths[0] - 0.3, depths[-1] + 0.3)
    ax.set_xlabel("passes through the core, $R$")


def title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left")


def zero_line(ax):
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)


def four_block(raw):
    """(a) every model, full range; (b) looped models from R = 2, zoomed; (c) merge minus no merge, same seed; (d) anchor residual minus merge."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2), gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    (a, b), (c, d) = axes
    seed = {arm: [[mean_bpb(raw, f"{arm}_s{s}", k) for k in DEPTHS] for s in SEEDS] for arm in "abe"}
    mean = {arm: [st.mean(col) for col in zip(*rows)] for arm, rows in seed.items()}
    flat4 = st.mean(mean_bpb(raw, f"c_s{s}", 1) for s in SEEDS)
    flat10 = st.mean(mean_bpb(raw, f"d_s{s}", 1) for s in SEEDS)

    for arm, colour, marker, name in ARMS:
        a.plot(DEPTHS, mean[arm], color=colour, marker=marker, ms=3.5, mec="white", mew=0.6, label=name)
    a.axhline(flat4, color=YELLOW, ls=(0, (4, 2)), label="flat, same four blocks")
    a.axhline(flat10, color=MUTED, ls=(0, (1.5, 1.5)), label="flat, ten blocks")
    title(a, "a", "PM-BPB, all four-block models")
    a.set_ylabel("PM-BPB (lower is better)")
    depth_axis(a)

    zoom = DEPTHS[1:]
    for arm, colour, marker, name in ARMS:
        for rows in seed[arm]:
            b.plot(zoom, rows[1:], color=colour, lw=0.6, alpha=0.45)
        b.plot(zoom, mean[arm][1:], color=colour, marker=marker, ms=3.5, mec="white", mew=0.6)
    title(b, "b", "looped models from $R = 2$ (zoomed)")
    b.set_ylabel("PM-BPB")
    depth_axis(b, zoom)

    for ax, (x, y, colour, marker, text, letter) in ((c, ("a", "b", BLUE, "o", "merge $-$ no merge", "c")),
                                                    (d, ("e", "a", AQUA, "^", "anchor residual $-$ merge", "d"))):
        diffs = [[p - q for p, q in zip(seed[x][i], seed[y][i])] for i in range(len(SEEDS))]
        for row in diffs:
            ax.plot(DEPTHS, row, ls="none", marker=marker, ms=3.5, color=colour, alpha=0.55, mec="none")
        ax.plot(DEPTHS, [st.mean(col) for col in zip(*diffs)], color=colour)
        zero_line(ax)
        title(ax, letter, f"{text}, same seed")
        ax.set_ylabel("difference in PM-BPB")
        depth_axis(ax)
    c.set_ylim(top=0.0012)
    c.annotate("below zero: merge better", (0.98, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=7.5, color=MUTED)
    d.annotate("below zero: anchor residual better", (0.98, 0.05), xycoords="axes fraction", ha="right", va="bottom", fontsize=7.5, color=MUTED)

    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.02), handlelength=2.4)
    fig.savefig(OUT / "pmbpb_four_block.pdf", bbox_inches="tight")
    plt.close(fig)


def eight_block(raw):
    """(a) mean PM-BPB per depth, looped models and the flat model; (b) merge minus no merge per source and for the mean (one seed)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.7), gridspec_kw={"wspace": 0.32})
    for model, colour, marker, name in (("A8", BLUE, "o", "looped, merge"), ("B8", ORANGE, "s", "looped, no merge")):
        a.plot(DEPTHS, [mean_bpb(raw, model, k) for k in DEPTHS], color=colour, marker=marker, ms=3.5, mec="white", mew=0.6, label=name)
    a.axhline(mean_bpb(raw, "C8", 1), color=YELLOW, ls=(0, (4, 2)), label="flat, same eight blocks")
    a.legend(loc="upper right", fontsize=7.5)
    title(a, "a", "PM-BPB")
    a.set_ylabel("PM-BPB (lower is better)")
    depth_axis(a)

    for (source, name), marker in zip(SOURCES, ("o", "s", "D", "v")):
        b.plot(DEPTHS, [raw[("A8", k, source)] - raw[("B8", k, source)] for k in DEPTHS], color=MUTED, lw=0.7, marker=marker, ms=3.2,
               mfc="white", mec=MUTED, mew=0.8, label=name)
    b.plot(DEPTHS, [mean_bpb(raw, "A8", k) - mean_bpb(raw, "B8", k) for k in DEPTHS], color=BLUE, lw=1.8, label="mean of the four")
    zero_line(b)
    b.legend(loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=7, ncol=2, columnspacing=1.0)  # the empty band between the mean and code
    b.annotate("below zero: merge better", (0.98, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=7.5, color=MUTED)
    title(b, "b", "merge $-$ no merge, per source")
    b.set_ylabel("difference in PM-BPB")
    depth_axis(b)
    fig.savefig(OUT / "pmbpb_eight_block.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    raw = load()
    four_block(raw)
    eight_block(raw)
    print("wrote", *sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()

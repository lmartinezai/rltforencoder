"""Selected consistency check: recompute the paper's main numbers from the committed result files and check that main.tex states them, rounded
as printed. Table values are checked cell by cell in the row with the right label (arm and depth), not merely somewhere in the text.

    uv run python paper/check_numbers.py

Covered: every table row by row (models, paired differences, eight-block PM-BPB, latency), the four-block PM-BPB and loss values in the prose,
the compute figures, the int8 bound, the halting ranges and exploration maxima, the zero-shot NLI values in the prose, and the code listing (each
line must occur in src/arch.py). Not covered: every other number in the prose (quantisation per model, ...). Nothing is measured
here; every value is read from results/ (see paper/MANIFEST.md). A failure means main.tex and the results have drifted apart."""
import csv
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))  # reproduce/reproduce.py points this at its own reproduced results
TEX = re.sub(r"\\textbf\{([^}]*)\}", r"\1", (ROOT / "paper" / "main.tex").read_text(encoding="utf-8"))  # bold cells compare as plain
SOURCES = ("fineweb-holdout", "arxiv", "wikipedia", "python-code")
SEEDS = (1, 2, 3)
RUN_200M = 12208 * 32
failures, checked = [], 0


FLAT = re.sub(r"\s+", " ", TEX)


def expect(label, text):
    """A phrase from the prose, with its context words, so the number is bound to its claim (line breaks compare as spaces)."""
    global checked
    checked += 1
    if text not in FLAT:
        failures.append(f"{label}: '{text}' not in main.tex")


def table_text(key):
    """The body of the table environment that carries \\label{key}."""
    for block in re.findall(r"\\begin\{table\}.*?\\end\{table\}", TEX, flags=re.DOTALL):
        if f"\\label{{{key}}}" in block:
            return block
    failures.append(f"no table labelled {key}")
    return ""


def row(*leading, table):
    """The cells after `leading` of the one row of table `table` whose first cells are `leading` (None, and a failure, unless exactly one)."""
    hits = []
    for line in table_text(table).splitlines():
        if "&" not in line or not line.rstrip().endswith("\\\\"):
            continue
        cells = [c.strip() for c in line.rstrip()[:-2].split("&")]
        if tuple(cells[: len(leading)]) == leading:
            hits.append(cells[len(leading):])
    if len(hits) != 1:
        failures.append(f"row {leading}: found {len(hits)} rows")
        return None
    return hits[0]


def expect_row(leading, values, table):
    global checked
    checked += 1
    cells = row(*leading, table=table)
    if cells is not None and cells[: len(values)] != list(values):
        failures.append(f"row {leading}: main.tex has {cells[: len(values)]}, results give {list(values)}")


def f3(x):
    return f"{x:.3f}"


def sg(x):
    return f"${x:+.4f}$"


bpb_rows = list(csv.DictReader((RESULTS / "mlm_crossmodel/bpb.csv").read_text().splitlines()))


def source_bpb(model, depth):
    v = {r["dataset"]: float(r["bits_per_byte"]) for r in bpb_rows if r["model"] == model and r["depth"] == str(depth)}
    return [v[s] for s in SOURCES]


def mean_bpb(model, depth):
    return st.mean(source_bpb(model, depth))


def arm_bpb(arm, depth):
    return st.mean(mean_bpb(f"{arm}_s{s}", depth) for s in SEEDS)


# Table 2 (8-block PM-BPB): label, depth, four sources, mean
for label, model, depth in (("A looped, merge", "A8", 1), ("A looped, merge", "A8", 4), ("A looped, merge", "A8", 8), ("B looped, no merge", "B8", 1),
                            ("B looped, no merge", "B8", 4), ("B looped, no merge", "B8", 8), ("C flat", "C8", 1), ("ModernBERT-large (teacher)", "modernbert-large", 1),
                            ("RoBERTa-base", "roberta-base", 1), ("DeBERTa-base", "deberta-base", 1), ("DistilBERT-base-cased", "distilbert-cased", 1)):
    v = source_bpb(model, depth)
    expect_row((label, str(depth) if model in ("A8", "B8") else "--"), [f3(x) for x in v + [st.mean(v)]], "tab:main")

# Table 3 (four-block study): loss mean +- sd at depths 1, 2, 4, 8, then mean PM-BPB at R=4 and R=8
small = json.loads((RESULTS / "small_study/summary.json").read_text())


def loss_cell(arm, d):
    c = small[arm][f"eval_R{d}_loss"]
    return f"{c['mean']:.3f} $\\pm$ {c['std']:.3f}".replace("$\\pm$ 0.", "$\\pm$ .")


def loss_mean(arm, d):
    return f"{small[arm][f'eval_R{d}_loss']['mean']:.3f}"


expect("four-block R=4 merge vs none", f"PM-BPB at $R = 4$ is {f3(arm_bpb('a', 4))}, against {f3(arm_bpb('b', 4))} without it")
expect("four-block R=8", f"({f3(arm_bpb('a', 8))} and {f3(arm_bpb('b', 8))})")
expect("anchor residual R=1", f"({f3(arm_bpb('e', 1))}, against {f3(arm_bpb('a', 1))} with the merge)")
expect("anchor residual best", f"Its value at its best depth, {f3(min(arm_bpb('e', k) for k in range(1, 9)))}, is above the merged model's best, "
       f"{f3(min(arm_bpb('a', k) for k in range(1, 9)))}")
expect("anchor residual plateau", f"From $R = 4$ on it stays at {f3(arm_bpb('e', 4))}, while the merged model's PM-BPB rises to {f3(arm_bpb('a', 8))}")
expect("anchor residual R=2", f"$R = 2$ ({f3(arm_bpb('e', 2))} against {f3(arm_bpb('a', 2))})")
expect("anchor residual R=3", f"({f3(arm_bpb('e', 3))} against {f3(arm_bpb('a', 3))} at $R = 3$)")
for k in range(1, 9):  # lower than the merged model at R = 1, 2 and 8, higher at R = 3 to 7
    checked += 1
    if (arm_bpb("e", k) < arm_bpb("a", k)) != (k in (1, 2, 8)):
        failures.append(f"anchor residual vs merge ordering claim fails at R={k}")
checked += 1
if len({f3(arm_bpb("e", k)) for k in range(4, 9)}) != 1:
    failures.append("anchor residual is said to be constant from R=4 on")
checked += 1
if f3(arm_bpb("e", 4)) != f3(arm_bpb("e", 8)):
    failures.append("anchor residual is said to stay flat from R=4 to R=8")
expect("flat four-block", f"same four blocks has PM-BPB {f3(arm_bpb('c', 1))}")
expect("ten-block flat", f"flat model has PM-BPB {f3(arm_bpb('d', 1))}, against {f3(arm_bpb('b', 4))} for the looped model")
expect("anchor residual loss R=1", f"({loss_mean('e', 1)} against {loss_mean('a', 1)} with the merge)")
expect("anchor residual loss R=4", f"({loss_mean('e', 4)} against {loss_mean('a', 4)})")
for s in SEEDS:  # the anchor residual's loss still improves from R=4 to R=8 in every seed, the other two looped models' gets worse
    checked += 1
    e4, e8 = (small["e"][f"eval_R{d}_loss"]["per_seed"][str(s)] for d in (4, 8))
    if not e8 < e4 or any(small[x]["eval_R8_loss"]["per_seed"][str(s)] <= small[x]["eval_R4_loss"]["per_seed"][str(s)] for x in "ab"):
        failures.append(f"R=4 to R=8 loss trend claim fails for seed {s}")


# Table 4 (seed-paired differences at R=4)
def paired(x, y, dy):
    return [mean_bpb(f"{x}_s{s}", 4) - mean_bpb(f"{y}_s{s}", dy) for s in SEEDS]


def loss_pairs(d):
    return [small["a"][f"eval_R{d}_loss"]["per_seed"][str(s)] - small["b"][f"eval_R{d}_loss"]["per_seed"][str(s)] for s in SEEDS]


for leading, diffs in ((("merge (A$'$ $-$ B$'$)", "PM-BPB, $R = 4$"), paired("a", "b", 4)),
                       (("anchor residual (E$'$ $-$ A$'$)", "PM-BPB, $R = 4$"), paired("e", "a", 4)),
                       (("looping (B$'$ $-$ C$'$)", "PM-BPB, $R = 4$"), paired("b", "c", 1))) + tuple(
        (("merge (A$'$ $-$ B$'$)", f"loss, $R = {d}$"), loss_pairs(d)) for d in (1, 2, 4, 8)):
    expect_row(leading, [sg(v) for v in diffs + [st.mean(diffs)]], "tab:paired")
expect("anchor residual range", f"by {min(-v for v in paired('a', 'e', 4)):.3f} to {max(-v for v in paired('a', 'e', 4)):.3f} PM-BPB")
expect("looping range", f"PM-BPB is lower by {min(-v for v in paired('b', 'c', 1)):.3f} to {max(-v for v in paired('b', 'c', 1)):.3f}")
all_depths = [mean_bpb(f"b_s{s}", k) - mean_bpb(f"a_s{s}", k) for k in range(1, 9) for s in SEEDS]  # no merge minus merge, every depth and seed
checked += 1
if min(all_depths) <= 0:
    failures.append("PM-BPB with the merge is said to be lower in every seed at every depth")
expect("merge PM-BPB range, all depths", f"by {min(all_depths):.4f} to {max(all_depths):.4f}; at $R = 4$")
expect("merge PM-BPB range", f"the difference is {min(-v for v in paired('a', 'b', 4)):.4f} to {max(-v for v in paired('a', 'b', 4)):.4f}")
merge_range = [small["b"][f"eval_R{d}_loss"]["per_seed"][str(s)] - small["a"][f"eval_R{d}_loss"]["per_seed"][str(s)] for d in (1, 2, 4, 8) for s in SEEDS]
expect("merge loss range", f"by {min(merge_range):.4f} to {max(merge_range):.4f}")

# Table 1 (body and total parameters, weight storage, encoder GFLOP at R=4, 512 tokens)
by_model = {r["model"]: r for r in csv.DictReader((RESULTS / "hardware/static.csv").read_text().splitlines())
            if r["seq"] == "512" and r["depth"] in ("4", "-")}
for label, description, stored, applied, model in (
        ("A", "looped, merge", 8, 20, "A8"), ("B", "looped, no merge", 8, 20, "B8"), ("C", "flat", 8, 8, "C8"),
        ("A$'$", "looped, merge", 4, 10, "a_s1"), ("B$'$", "looped, no merge", 4, 10, "b_s1"), ("E$'$", "looped, anchor-residual merge", 4, 10, "e_s1"),
        ("C$'$", "flat", 4, 4, "c_s1"), ("D$'$", "flat, ten blocks", 10, 10, "d_s1"), ("", "ModernBERT-large (teacher)", 28, 28, "teacher")):
    r = by_model[model]
    expect_row((label, description), [str(stored), str(applied), f"{float(r['body_M']):.1f}", f"{float(r['total_M']):.1f}", f"{2 * float(r['total_M']):.0f}",
                                      f"{float(r['encoder_GFLOP']):.0f}"], "tab:arms")
expect("D'/B' total ratio", f"{float(by_model['d_s1']['total_M']) / float(by_model['b_s1']['total_M']):.1f} times its total")

# Zero-shot NLI table: accuracy (%) per task, calibrated and raw, MNLI's two dev sets pooled by size, mean of the two templates and of the seeds
nli = list(csv.DictReader((RESULTS / "zeroshot_nli/accuracy.csv").read_text().splitlines()))


def nli_value(model, depth, task, metric):
    cells = [r for r in nli if r["model"] == model and r["depth"] == str(depth) and r["task_split"].startswith(task + "/")]
    return 100 * sum(int(r["n"]) * float(r[metric]) for r in cells) / sum(int(r["n"]) for r in cells)


def nli_row(models, depth):
    return [f"{st.mean(nli_value(m, depth, task, metric) for m in models):.1f}" for task in ("mnli", "qnli", "rte") for metric in ("calibrated", "raw")]


def arms3(arm):
    return [f"{arm}_s{s}" for s in SEEDS]



def floor(task):
    cells = [r for r in nli if r["model"] == "modernbert-large" and r["template"] == "T1" and r["task_split"].startswith(task + "/")]
    return f"{100 * sum(int(r['n']) * float(r['majority']) for r in cells) / sum(int(r['n']) for r in cells):.1f}"


teacher_mnli, d_mnli = nli_row(["modernbert-large"], 1)[:2], nli_row(arms3("d"), 1)[:2]
four = [float(v) for arm, depth in (("a", 4), ("b", 4), ("e", 4), ("c", 1)) for v in nli_row(arms3(arm), depth)[:2]]
expect("NLI floors", f"({floor('qnli')}\\% and {floor('rte')}\\%)")
expect("MNLI floor", f"where that rate is {floor('mnli')}\\%")
expect("MNLI teacher", f"teacher reaches {teacher_mnli[0]}\\% ({teacher_mnli[1]}\\%)")
expect("MNLI ten-block", f"ten-block flat model reaches {d_mnli[0]}\\% ({d_mnli[1]}\\%)")
expect("MNLI four-block", f"are at {min(four):.1f} to {max(four):.1f}\\% with and without calibration")
eight_nli = [nli_row([m], d)[:2] for m, d in (("A8", 4), ("B8", 4), ("C8", 1))]
cal8, raw8 = [float(v[0]) for v in eight_nli], [float(v[1]) for v in eight_nli]
expect("MNLI eight-block", f"are at {min(cal8):.1f} to {max(cal8):.1f}\\% with calibration and {min(raw8):.1f} to {max(raw8):.1f}\\% without")

# Training compute (counted operations, merge profile for the looped arms)
m = json.loads((RESULTS / "compute_ledger.json").read_text())["measured_gflop_per_sequence"]


def mean_train(k):
    return st.mean(m[k]["train_by_depth"].values())


looped, flat, teacher = (x * RUN_200M / 1e6 for x in (mean_train("bort8"), mean_train("flat8"), m["teacher_forward"]))
ratio8, ratio4 = mean_train("bort8") / mean_train("flat8"), mean_train("bort4") / mean_train("flat4")
expect("looped PFLOP", f"counted at {looped:.0f} PFLOP for an eight-block looped model")
expect("looped PFLOP, seeds", f"run is counted at {looped:.0f} PFLOP ($10^{{15}}$ floating-point operations")
four_total = 3 * 3072 * 32 * (2 * mean_train("bort4") + mean_train("bort4_anchor") + mean_train("flat4") + mean_train("flat10")) / 1e6
checked += 1
if not four_total < 2 * looped + flat:
    failures.append(f"four-block study ({four_total:.0f} PFLOP) is said to cost less than the three eight-block runs ({2 * looped + flat:.0f})")
expect("flat PFLOP", f"{flat:.0f} for the flat model, a ratio of {ratio8:.2f}")
expect("ratio 4", f"the ratio is {ratio4:.2f}")
expect("teacher PFLOP", f"teacher targets for 200M tokens are counted at another {teacher:.0f} PFLOP")
expect("pipeline", f"about {ratio8 * (flat + teacher):.0f} PFLOP against {looped + teacher:.0f}")
expect("ledger total", f"{json.loads((RESULTS / 'compute_ledger.json').read_text())['total_pflop'] / 1e3:.2f} EFLOP")

# int8 bound: mean over the four sources, every evaluated student and depth
quant = json.loads((RESULTS / "hardware/quant.json").read_text())
students = ("A8", "B8", "C8", "a_s1", "b_s1", "c_s1", "d_s1", "e_s1")
worst = max(st.mean(quant[f"{s}/int8"]["bpb"][d].values()) - st.mean(quant[f"{s}/fp"]["bpb"][d].values()) for s in students for d in quant[f"{s}/int8"]["bpb"])
checked += 1
if not worst < 0.0012:
    failures.append(f"int8 bound: worst change {worst:.5f} is not below 0.0012")
expect("int8 bound", "by less than 0.0012")

# Halting: operating-point range of the reported heads; the appendix table and its hindsight gains
loops = [o["loops"] for f in (RESULTS / "halting_v2").glob("*.json") for s in json.loads(f.read_text())["sequence_head"].values() for o in s["operating_points"]]
expect("operating points", f"span {min(loops):.1f} to {max(loops):.1f} passes")
best = {}
for kind, fname in (("linear", "A8_linear.json"), ("hidden", "A8_mlp.json")):
    d = json.loads((RESULTS / "halting_v2/target_variants" / fname).read_text())
    best[kind] = max(v["best_gain"] for src in ("fineweb", "arxiv") for v in d[src].values() if isinstance(v, dict) and "best_gain" in v)
    expect(f"{kind} oracle", f"oracle gains of {d['fineweb']['ceiling'][0]:.4f} on FineWeb and {d['arxiv']['ceiling'][0]:.4f} on arXiv")
expect("exploration maxima", f"most {best['linear']:.4f} with linear heads and {best['hidden']:.4f} with hidden layers")

# Latency table (Table 4), from the R = 4 pass section of results/hardware/summary.md; columns RTX 5070, NPU, iGPU, OpenVINO CPU, PyTorch CPU
summary = (RESULTS / "hardware/summary.md").read_text(encoding="utf-8").split("## Latency at R = 4, every backend")[1].split("## ")[0]
lat = {}
for line in summary.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) == 7 and cells[0] != "model" and not cells[0].startswith("-"):
        lat[cells[0]] = cells[2:]
ROWS = ((("A", "looped, merge, $R = 4$"), "A looped+merge (8 blocks)"), (("B", "looped, no merge, $R = 4$"), "B looped (8 blocks)"),
        (("C", "flat"), "C flat (8 blocks)"), (("A$'$", "looped, merge, $R = 4$"), "A' looped+merge (4 blocks)"),
        (("B$'$", "looped, no merge, $R = 4$"), "B' looped (4 blocks)"), (("E$'$", "looped, anchor residual, $R = 4$"), "E' looped+merge, input injection (4 blocks)"),
        (("C$'$", "flat"), "C' flat (4 blocks)"), (("D$'$", "flat, ten blocks"), "D' flat, 10 blocks (iso-FLOP)"),
        (("", "ModernBERT-large (teacher)"), "ModernBERT-large (teacher)"))
for leading, name in ROWS:
    expect_row(leading, lat[name], "tab:latency")
ms = {name: [float(x) for x in lat[name]] for _, name in ROWS}
ratio = lambda a, b: [x / y for x, y in zip(ms[a], ms[b], strict=True)]
ratios = [r for lo, fl in (("A looped+merge (8 blocks)", "C flat (8 blocks)"), ("B looped (8 blocks)", "C flat (8 blocks)"),
                           ("A' looped+merge (4 blocks)", "C' flat (4 blocks)"), ("B' looped (4 blocks)", "C' flat (4 blocks)")) for r in ratio(lo, fl)]
expect("latency ratio range", f"by {min(ratios):.1f} times (four blocks without the merge, integrated GPU) to {max(ratios):.1f} times (four blocks with the merge, NPU)")
expect("latency ratio range, intro", f"{min(ratios):.1f} to {max(ratios):.1f} times the flat encoder's latency")
merge4, merge8 = ratio("A' looped+merge (4 blocks)", "B' looped (4 blocks)"), ratio("A looped+merge (8 blocks)", "B looped (8 blocks)")
expect("merge latency on the NPU", f"on the NPU by {100 * (merge4[1] - 1):.0f}\\% for the four-block models and {100 * (merge8[1] - 1):.0f}\\% for the eight-block models")
expect("merge latency elsewhere", f"on the other backends by at most {100 * (max(merge4[i] for i in (0, 2, 3, 4)) - 1):.0f}\\%")
d_b = ratio("D' flat, 10 blocks (iso-FLOP)", "B' looped (4 blocks)")
expect("ten-block vs no-merge", f"within {100 * max(abs(x - 1) for x in d_b):.0f}\\% of the four-block looped model without the merge")
# Agreement of the earlier sweep with the R = 4 pass, per backend, over the configurations both timed (largest relative difference)
def timings(path):
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "error" not in r and r["seq"] == 512 and r["batch"] == 1:
            out[(r["backend"], r["device"], r["model"], r["depth"])] = r["p50_ms"]
    return out


early, r4 = timings(RESULTS / "hardware/latency.jsonl"), timings(RESULTS / "hardware/depth4_pass/latency.jsonl")
agree = {dev: round(100 * max(abs(early[k] / r4[k] - 1) for k in r4 if k in early and k[:2] == dev)) for dev in
         (("openvino", "NPU"), ("openvino", "GPU.0"), ("torch", "cuda"), ("torch", "cpu"), ("openvino", "CPU"))}
expect("cross-session agreement", "its values are within {}\\% of the table's on the NPU, {}\\% on the integrated GPU, {}\\% on the RTX 5070, {}\\% on "
       "the PyTorch CPU path, and {}\\% on the OpenVINO CPU path".format(*agree.values()))
e_a = ratio("E' looped+merge, input injection (4 blocks)", "A' looped+merge (4 blocks)")
expect("anchor residual latency", f"within {100 * max(abs(x - 1) for x in e_a):.0f}\\% of the merged model's")

# Code listing: every line must occur verbatim in src/arch.py
arch = (ROOT / "src/arch.py").read_text(encoding="utf-8")
for block in re.findall(r"\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}", TEX, flags=re.DOTALL):
    for line in filter(None, (x.strip() for x in block.splitlines())):
        checked += 1
        if line not in arch:
            failures.append(f"listing line not in src/arch.py: {line}")

# Seeds paragraph: an eight-block looped run against a four-block one, in counted training operations
expect("run ratio", f"{mean_train('bort8') * RUN_200M / (mean_train('bort4') * 3072 * 32):.1f} times a four-block looped run")

# Equal inference FLOPs: the ten-block flat model against the four-block looped models at R = 4, unrounded counts
static = {(r["model"], r["depth"], r["seq"]): float(r["encoder_GFLOP"]) for r in csv.DictReader((RESULTS / "hardware/static.csv").read_text().splitlines())}
checked += 1
if static[("b_s1", "4", "512")] != static[("d_s1", "-", "512")]:
    failures.append("the ten-block flat model is said to have exactly the no-merge looped model's inference FLOPs")
expect("FLOP match GFLOP", f"({static[('d_s1', '-', '512')]:.0f} GFLOP per sequence")
expect("merge FLOP excess", f"the merge have about {round(100 * (static[('a_s1', '4', '512')] / static[('b_s1', '4', '512')] - 1), -1):.0f}\\% more")
expect("looping training ratio, four-block", f"at about {ratio4:.1f} times the student-training operations")

# Eight-block distillation loss: depth of the lowest loss (FineWeb hold-out block of results/arms_200m/summary.md)
best8 = re.findall(r"\| ([AB]) looped[^|]*\|.*\| (\d\.\d+) \(depth (\d)\) \|", (RESULTS / "arms_200m/summary.md").read_text(encoding="utf-8"))
fw = {arm: (float(v), d) for arm, v, d in best8 if 1.9 < float(v) < 2.0}  # the FineWeb hold-out rows (loss near 1.94)
checked += 1
if fw["A"][1] != fw["B"][1]:
    failures.append("the eight-block looped models are said to have their lowest loss at the same depth")
expect("eight-block loss best depth", f"both looped models are best at $R = {fw['A'][1]}$ ({fw['A'][0]:.3f} and {fw['B'][0]:.3f})")
# merge share of the model at each size
mp = 1e6 * (float(by_model["A8"]["total_M"]) - float(by_model["B8"]["total_M"]))  # merge parameters
expect("merge share", f"about {round(100 * mp / (1e6 * float(by_model['A8']['total_M']))):.0f}\\% of the eight-block model and "
       f"{round(100 * mp / (1e6 * float(by_model['a_s1']['total_M']))):.0f}\\% of the four-block")

# Training ratio with the teacher targets assigned to each eight-block run
expect("eight-block ratio with teacher", f"the eight-block ratio is {(looped + teacher) / (flat + teacher):.2f}")
expect("eight-block ratio with teacher, rounded", f"ratio is about {(looped + teacher) / (flat + teacher):.1f}")
# Eight-block merge minus no merge, PM-BPB, every depth (one seed)
gap8 = [mean_bpb("B8", k) - mean_bpb("A8", k) for k in range(1, 9)]
checked += 1
if min(gap8) <= 0:
    failures.append("the eight-block merged model is said to be better at every depth")
expect("eight-block merge range", f"by {min(gap8):.4f} to {max(gap8):.4f} ({f3(mean_bpb('A8', 4))} against {f3(mean_bpb('B8', 4))} at $R = 4$;")
# Conclusion trade-off: weight storage and PM-BPB at equal counted inference FLOPs
mb = {k: 2 * float(by_model[k]["total_M"]) for k in ("b_s1", "d_s1")}
expect("trade-off storage", f"({mb['b_s1']:.0f} MB for the looped model without the merge, against {mb['d_s1']:.0f} MB for")
expect("trade-off PM-BPB", f"substantially lower PM-BPB ({f3(arm_bpb('d', 1))} against {f3(arm_bpb('b', 4))})")
expect("anchor best vs merge best, conclusion", f"({f3(min(arm_bpb('e', k) for k in range(1, 9)))} against {f3(min(arm_bpb('a', k) for k in range(1, 9)))} PM-BPB)")

# Per-source merge differences (merge minus no merge, bits per byte)
def src(model, depth, source):
    return next(float(r["bits_per_byte"]) for r in bpb_rows if r["model"] == model and r["depth"] == str(depth) and r["dataset"] == source)


src4 = {so: sum(src(f"a_s{q}", k, so) < src(f"b_s{q}", k, so) for k in range(1, 9) for q in SEEDS) for so in SOURCES}
expect("four-block per source", f"in all {src4['fineweb-holdout']} depth--seed pairs on FineWeb, arXiv, and Wikipedia, and in {src4['python-code']} of 24 on code")
checked += 1
if not src4["fineweb-holdout"] == src4["arxiv"] == src4["wikipedia"] == 24:
    failures.append("four-block merge said to be better in all 24 pairs on FineWeb, arXiv, and Wikipedia")
d8 = {so: [src("A8", k, so) - src("B8", k, so) for k in range(1, 9)] for so in SOURCES}
expect("eight-block code range", f"largest on code, {-max(d8['python-code']):.4f} to {-min(d8['python-code']):.4f} at every depth")
expect("eight-block arXiv/Wikipedia worst", f"by at most {max(d8['arxiv'] + d8['wikipedia']):.4f}")
checked += 1
if max(d8["fineweb-holdout"]) > 0 or max(d8["python-code"]) >= 0 or max(d8["arxiv"][4:] + d8["wikipedia"][4:]) >= 0 or min(d8["arxiv"][0], d8["wikipedia"][0]) <= 0:
    failures.append("eight-block per-source sign pattern (FineWeb <= 0, code < 0, arXiv and Wikipedia worse at R=1 and better from R=5) fails")

if failures:
    print("\n".join(failures))
    sys.exit(f"{len(failures)} of {checked} checks failed")
print(f"{checked} checks: all match main.tex")

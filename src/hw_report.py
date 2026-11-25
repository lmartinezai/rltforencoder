"""Tables of the hardware study from the measured files in results/hardware (static.csv, latency.jsonl, quant.json) and the cross-tokenizer
quality scores (results/mlm_crossmodel/raw.json).

    uv run python src/hw_report.py --dir results/hardware

A small-model name stands for its three seeds (hardware cost does not depend on the trained values, so one seed is timed). A looped model's
latency at depths that were not timed is the straight line through the timed ones (latency is linear in depth: one more iteration costs the
same), marked * in the tables; on OpenVINO only two depths were timed, so that line is exact by construction there and not a check."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

LABELS = {"A8": "A looped+merge (8 blocks)", "B8": "B looped (8 blocks)", "C8": "C flat (8 blocks)", "a_s1": "A' looped+merge (4 blocks)", "b_s1": "B' looped (4 blocks)",
          "c_s1": "C' flat (4 blocks)", "d_s1": "D' flat, 10 blocks (iso-FLOP)", "e_s1": "E' looped+merge, input injection (4 blocks)", "teacher": "ModernBERT-large (teacher)", "roberta-base": "RoBERTa-base",
          "deberta-base": "DeBERTa-base", "distilbert-cased": "DistilBERT-cased"}
LOOPED = ("A8", "B8", "a_s1", "b_s1", "e_s1")
DEVICES = {("openvino", "NPU"): "Intel NPU (OpenVINO, fp16)", ("openvino", "GPU.0"): "Intel iGPU (OpenVINO, fp16)", ("openvino", "CPU"): "Core Ultra 7 265K CPU (OpenVINO, fp32)",
           ("torch", "cuda"): "RTX 5070 (PyTorch eager, bf16)", ("torch", "cpu"): "CPU (PyTorch, fp32)"}


def load_latency(path):
    last = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        last[(r["backend"], r["device"], r["model"], r["depth"], r["seq"], r["batch"])] = r  # a re-run replaces the earlier row of the same configuration
    rows = list(last.values())
    return [r for r in rows if "error" not in r], [r for r in rows if "error" in r]


def fit_depth(points):
    """[(depth, ms)] -> (intercept, slope, max relative residual)."""
    n = len(points)
    mx, my = sum(d for d, _ in points) / n, sum(v for _, v in points) / n
    slope = sum((d - mx) * (v - my) for d, v in points) / max(1e-12, sum((d - mx) ** 2 for d, _ in points)) if n > 1 else 0.0
    icpt = my - slope * mx
    return icpt, slope, max(abs(icpt + slope * d - v) / v for d, v in points)


def latency_table(rows):
    """-> {(backend, device, model, seq, batch): {depth: (ms, measured)}}, looped models filled to depths 1..8 by the fitted line."""
    grouped = defaultdict(dict)
    for r in rows:
        grouped[(r["backend"], r["device"], r["model"], r["seq"], r["batch"])][r["depth"]] = r["p50_ms"]
    out = {}
    for k, by_depth in grouped.items():
        if list(by_depth) == [None]:
            out[k] = {None: (by_depth[None], True)}
            continue
        points = sorted(by_depth.items())
        if len(points) >= 2:
            icpt, slope, _ = fit_depth(points)
            out[k] = {d: (by_depth[d], True) if d in by_depth else (icpt + slope * d, False) for d in range(1, 9)}
        else:
            out[k] = {d: (v, True) for d, v in points}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="results/hardware")
    a = p.parse_args(argv)
    d = Path(a.dir)
    static = list(csv.DictReader(open(d / "static.csv")))  # noqa: SIM115
    body = {r["model"]: float(r["body_M"]) for r in static}
    ok, failed = load_latency(d / "latency.jsonl") if (d / "latency.jsonl").exists() else ([], [])
    lat = latency_table(ok)
    models = [m for m in LABELS if m in body]

    quant = json.loads((d / "quant.json").read_text()) if (d / "quant.json").exists() else {}

    lines = ["# Hardware study", ""]
    lines += ["## Parameters and FLOPs (static)", "", "| model | body M | embeddings M | body MB fp16 | body MB int8 | encoder GFLOP, 128 tok | encoder GFLOP, 512 tok |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        rs = [r for r in static if r["model"] == m]
        top = rs[-1]
        shown = lambda seq: " / ".join(f"{float(r['encoder_GFLOP']):.0f}" for r in rs if r["seq"] == seq and r["depth"] in ("-", "1", "4", "8"))  # noqa: B023
        lines.append(f"| {LABELS[m]} | {float(top['body_M']):.1f} | {float(top['embedding_M']):.1f} | {float(top['body_MB_fp16']):.0f} | {float(top['body_MB_int8']):.0f} | "
                     f"{shown('128')} | {shown('512')} |")
    lines += ["", "(looped models: depths 1 / 4 / 8)", ""]
    for key, title in (((512, 1), "512 tokens, batch 1: p50 ms"), ((128, 1), "128 tokens, batch 1: p50 ms"), ((512, 8), "512 tokens, batch 8: sequences per second")):
        lines += [f"## Latency, {title}", ""]
        cols = [k for k in DEVICES if any((k[0], k[1], m, *key) in lat for m in models)]
        lines += ["| model | depth | " + " | ".join(DEVICES[c] for c in cols) + " |", "|---|---|" + "---:|" * len(cols)]
        for m in models:
            for dep in ((1, 4, 8) if m in LOOPED else (None,)):
                cells = []
                for c in cols:
                    v = lat.get((c[0], c[1], m, *key), {}).get(dep)
                    cells.append("-" if v is None else (f"{v[0]:.1f}" if key[1] == 1 else f"{key[1] * 1000 / v[0]:.1f}") + ("" if v[1] else "*"))
                if any(x != "-" for x in cells):
                    lines.append(f"| {LABELS[m]} | {dep or '-'} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["`*` = fitted line through the timed depths.", ""]
    r4 = [row for f in sorted((d / "depth4_pass").glob("*.jsonl")) for row in load_latency(f)[0]]
    if r4:  # the paper's Table 4: one dedicated pass that timed every model at R = 4 on every backend (E' in a pass of its own)
        by = {(r["backend"], r["device"], r["model"], r["depth"]): r["p50_ms"] for r in r4 if r["seq"] == 512 and r["batch"] == 1}
        cols = [("torch", "cuda"), ("openvino", "NPU"), ("openvino", "GPU.0"), ("openvino", "CPU"), ("torch", "cpu")]
        lines += ["## Latency at R = 4, every backend (the paper's Table 4): 512 tokens, batch 1, p50 ms", "",
                  "| model | depth | " + " | ".join(DEVICES[c] for c in cols) + " |", "|---|---|" + "---:|" * len(cols)]
        for m in models:
            dep = 4 if m in LOOPED else None
            cells = [f"{by[(*c, m, dep)]:.1f}" if (*c, m, dep) in by else "-" for c in cols]
            if any(x != "-" for x in cells):
                lines.append(f"| {LABELS[m]} | {dep or '-'} | " + " | ".join(cells) + " |")
        lines.append("")
    if quant:
        lines += ["## Weight quantisation (round-to-nearest, no calibration; mean BPB, change against unquantised)", "", "| model | depth | bf16 | int8 | int4 (group 64) |",
                  "|---|---|---:|---:|---:|"]
        names = sorted({k.split("/")[0] for k in quant}, key=lambda n: list(LABELS).index(n) if n in LABELS else 99)
        mean = lambda e, dep: statistics.fmean(e["bpb"][dep].values())
        for n in names:
            fp = quant.get(f"{n}/fp")
            if not fp:
                continue
            for dep in (("1", "4", "8") if len(fp["bpb"]) > 1 else ("1",)):
                cells = []
                for scheme in ("fp", "int8", "int4"):
                    e = quant.get(f"{n}/{scheme}")
                    cells.append("-" if not e else f"{mean(e, dep):.4f}" if scheme == "fp" else f"{mean(e, dep):.4f} ({mean(e, dep) - mean(fp, dep):+.4f})")
                lines.append(f"| {LABELS.get(n, n)} | {dep if len(fp['bpb']) > 1 else '-'} | " + " | ".join(cells) + " |")
        lines.append("")
    if failed:
        lines += ["## Configurations that failed", "", "| backend/device | model | depth | seq | batch | error |", "|---|---|---|---:|---:|---|"]
        lines += [f"| {r['backend']}/{r['device']} | {r['model']} | {r['depth']} | {r['seq']} | {r['batch']} | {r['error'][:100]} |" for r in failed]
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

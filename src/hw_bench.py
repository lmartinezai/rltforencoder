"""Latency and throughput of the masked-LM forward (src/ov_export.py: ids -> logits of the masked 30% of the positions) on this machine's devices.

    uv run python src/hw_bench.py --backend openvino --devices CPU GPU.0 NPU --out results/hardware
    uv run python src/hw_bench.py --backend torch --devices cuda cpu --out results/hardware

Every configuration (model, loop depth, sequence length, batch) is converted once, checked against the fp32 PyTorch output (relative max error, top-1
agreement), warmed up for WARMUP (3) runs, then timed run-by-run for at least MIN_SECONDS (3 s) and at least MIN_RUNS (8) runs: p50 / p95 / mean latency of
one call (with as few as 8 runs the p95 is the slowest or second-slowest run: report the p50). One row per
configuration is appended to latency.jsonl, with wall-clock start/end times so that src/hw_report.py can join the package-power (RAPL) and GPU-power
logs sampled during the sweep (src/hw_power.py) to get energy per sequence. A finished configuration is skipped on a re-run. Failures (a device that cannot
compile a graph) are recorded, not hidden. Run it alone: nothing else may use the CPU or the device while it times."""
import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from crossmodel_eval import load_model
from hw_models import available
from ov_export import MaskedLogits

MIN_SECONDS, MIN_RUNS, WARMUP = 3.0, 8, 3
OV_DEPTHS = (1, 8)  # latency is linear in depth (checked on the PyTorch sweep, which times all 8): two points fix the line
OV_SINGLE_DEPTH = {"B8": 8, "b_s1": 8}  # the no-merge arms were timed at one depth only on OpenVINO. They are NOT interchangeable with the merge arms there:
# on the NPU the merge arm was 28% (8 blocks) to 56% (4 blocks) slower at depth 8, far more than its 5-10% extra FLOPs
BATCH8_SEQ = 128  # throughput at batch 8 is timed at 128 tokens only
OV_PRECISION = {"CPU": "f32", "GPU.0": "f16", "NPU": "f16"}


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def timed(call, sync=lambda: None):
    for _ in range(WARMUP):
        call()
    sync()
    times, start = [], time.perf_counter()
    while len(times) < MIN_RUNS or time.perf_counter() - start < MIN_SECONDS:
        t = time.perf_counter()
        call()
        sync()
        times.append((time.perf_counter() - t) * 1e3)
    return {"runs": len(times), "p50_ms": statistics.median(times), "p95_ms": percentile(times, 0.95), "mean_ms": statistics.fmean(times), "min_ms": min(times)}


def agreement(out, ref):
    out = (out if isinstance(out, torch.Tensor) else torch.as_tensor(np.asarray(out))).float().cpu()  # (a bf16 tensor cannot go through numpy)
    ref = ref.float()
    return {"rel_max_error": ((out - ref).abs().max() / ref.abs().max()).item(), "top1_agreement": (out.argmax(-1) == ref.argmax(-1)).float().mean().item()}


def configs(models, seqs, batches, depths_for):
    for name, (kind, path, _, looped) in models.items():
        for depth in (depths_for(name) if looped else (None,)):
            for seq in seqs:
                for batch in batches:
                    if batch > 1 and seq != BATCH8_SEQ:
                        continue
                    yield name, kind, path, depth, seq, batch


def key(row):
    return (row["backend"], row["device"], row["model"], row["depth"], row["seq"], row["batch"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("openvino", "torch"), required=True)
    p.add_argument("--devices", nargs="+", required=True)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--seqs", type=int, nargs="+", default=[128, 512])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    p.add_argument("--depths", type=int, nargs="*", default=None, help="loop depths to time for looped models (default: 1 and 8 on OpenVINO, 1..8 on PyTorch)")
    p.add_argument("--retry_errors", action="store_true", help="re-run configurations that failed before")
    p.add_argument("--out", default="results/hardware")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "latency.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    done = {key(r) for r in rows if not (a.retry_errors and "error" in r)}
    models = {n: m for n, m in available().items() if a.models is None or n in a.models}
    torch.manual_seed(0)
    if a.backend == "openvino":
        import openvino as ov

        core = ov.Core()
        depths_for = lambda n: (OV_SINGLE_DEPTH[n],) if n in OV_SINGLE_DEPTH else OV_DEPTHS
    else:
        depths_for = lambda n: tuple(range(1, 9))
    if a.depths:
        depths_for = lambda n: tuple(a.depths)
    loaded, references, dead = {}, {}, set()
    for name, kind, path, depth, seq, batch in configs(models, a.seqs, a.batches, depths_for):
        todo = [d for d in a.devices if (a.backend, d, name, depth, seq, batch) not in done and d not in dead]
        if not todo:
            continue
        if name not in loaded:
            loaded.clear()
            loaded[name] = load_model(kind, path, "cpu")[0].requires_grad_(False)
        model = loaded[name]
        wrapper = MaskedLogits(model, seq, depth)
        ids = torch.randint(5, model.config.vocab_size, (batch, seq), generator=torch.Generator().manual_seed(seq))  # the same ids for every batch size
        if (name, depth, seq) not in references:
            with torch.no_grad():
                references.clear()
                references[(name, depth, seq)] = wrapper(ids[:1])
        ref = references[(name, depth, seq)]
        if a.backend == "openvino":
            t = time.perf_counter()
            graph = ov.convert_model(wrapper, example_input=ids)
            graph.reshape([batch, seq])
            convert_s = time.perf_counter() - t
        for device in todo:
            row = {"backend": a.backend, "device": device, "model": name, "depth": depth, "seq": seq, "batch": batch}
            try:
                if a.backend == "openvino":
                    row["precision"] = OV_PRECISION.get(device, "f32")
                    t = time.perf_counter()
                    compiled = core.compile_model(graph, device, {"INFERENCE_PRECISION_HINT": row["precision"], "PERFORMANCE_HINT": "LATENCY"})
                    row["compile_s"], row["convert_s"] = time.perf_counter() - t, convert_s
                    request = compiled.create_infer_request()
                    x = ids.numpy()
                    request.infer({0: x})
                    row |= agreement(request.get_output_tensor(0).data[:1], ref)
                    row["start"] = time.time()
                    row |= timed(lambda: request.infer({0: x}))  # noqa: B023 - called within the iteration
                else:
                    dtype = torch.float32 if device == "cpu" else torch.bfloat16
                    row["precision"] = "f32" if device == "cpu" else "bf16"
                    module = MaskedLogits(copy.deepcopy(model), seq, depth).to(device=device, dtype=dtype)  # a copy: the reference stays fp32 on the CPU
                    x = ids.to(device)
                    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

                    def call():
                        with torch.no_grad():
                            return module(x)  # noqa: B023

                    row |= agreement(call()[:1].cpu(), ref)
                    row["start"] = time.time()
                    row |= timed(call, sync)
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                        call()
                        row["peak_gpu_MB"] = torch.cuda.max_memory_allocated() / 2**20
                row["end"] = time.time()
                row["seq_per_s"] = 1e3 * batch / row["p50_ms"]
                print(f"[{a.backend}/{device}] {name} d={depth} S={seq} B={batch}: p50 {row['p50_ms']:.2f} ms, p95 {row['p95_ms']:.2f}, "
                      f"err {row['rel_max_error']:.3f}, top1 {row['top1_agreement']:.3f}", flush=True)
            except Exception as e:  # noqa: BLE001 - a device that cannot run a graph is a result
                row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"[{a.backend}/{device}] {name} d={depth} S={seq} B={batch}: FAILED {row['error'][:120]}", flush=True)
            if "DEVICE_LOST" in row.get("error", ""):
                dead.add(device)  # a hung device fails everything after it: stop using it in this run (reset it, re-run with --retry_errors)
            with open(log, "a") as f:
                f.write(json.dumps(row) + "\n")
            if a.backend == "torch" and device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

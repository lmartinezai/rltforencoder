"""Teacher-side GPU benchmark: is the machine that serves the teacher saturated, and on tensor cores?

Per (attention implementation, compile) x batch size it times the full serving step (teacher forward -> masked-position
hidden -> fp16 -> host arrays, exactly Teacher.__call__) with CUDA events after warm-up, counts FLOPs with FlopCounterMode,
reports achieved TFLOP/s against this card's measured bf16 matmul ceiling, and lists the kernels that dominate GPU time
(tensor-core GEMMs show up as cutlass/xmma/gemm names). Nothing else should use the GPU while it runs.

    uv run python src/bench_teacher.py [--batches 16,32,64,128] [--impls sdpa,eager] [--compile 0,1]
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoModelForMaskedLM

from store import batch_arrays
from targets import make_targets, teacher_hidden

REF = {}  # (batch, seq_len) -> reference hidden states
TC = ("gemm", "cutlass", "xmma", "wmma", "tensorop", "s16816", "s1688", "flash", "fmha", "sm80", "sm86", "sm89", "sm90", "sm120")


def matmul_ceiling(n=8192, iters=20):
    a, b = (torch.randn(n, n, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(iters):
        a @ b
    e.record()
    torch.cuda.synchronize()
    return 2 * n**3 * iters / (s.elapsed_time(e) / 1e3) / 1e12


def make_batch(B, L, vocab, mask_prob, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(5, vocab, (B, L), generator=g)
    labels = torch.where(torch.rand(B, L, generator=g) < mask_prob, ids, torch.full_like(ids, -100))
    ids = torch.where(labels != -100, torch.full_like(ids, 50284), ids)  # [MASK]
    return dict(input_ids=ids.to(device), attention_mask=torch.ones_like(ids).to(device), labels=labels.to(device))


@torch.no_grad()
def kl_at_T2(ref, z, W, b):
    """Mean KL(teacher(ref) || teacher(z)) of the distributions the student is trained on (decoder W z + b at T=2)."""
    lp = lambda h: torch.log_softmax((h.cuda() @ W.T + b) / 2.0, -1)
    return (lp(ref).exp() * (lp(ref) - lp(z))).sum(-1).mean()


@torch.no_grad()
def step(model, W, b, x):
    tg = make_targets(teacher_hidden(model, x), W, b, 2.0, 0, 1.0)
    return batch_arrays(x["input_ids"], x["labels"], tg)  # includes the device -> host copy


def bench(name, impl, compile_, B, args, ceiling):
    torch._dynamo.reset()  # a fresh compile per point: a stale cache (or the recompile limit) would silently fall back to eager
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForMaskedLM.from_pretrained(args.teacher, dtype=torch.bfloat16, attn_implementation=impl).cuda().eval().requires_grad_(False)
    if compile_:
        model.model = torch.compile(model.model)
    W, b = model.decoder.weight.detach().float(), model.decoder.bias.detach().float()
    x = make_batch(B, args.seq_len, model.config.vocab_size, 0.3, "cuda")
    t0 = time.time()
    for _ in range(args.warmup):
        step(model, W, b, x)
    warm = time.time() - t0
    times = []
    for _ in range(args.iters):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        step(model, W, b, x)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 1e3)
    sec = statistics.median(times)
    with torch.no_grad():
        z = teacher_hidden(model, x).float().cpu()
    ref = REF.setdefault((B, args.seq_len), z)  # the first variant at this batch size (uncompiled sdpa) is the reference
    rel = float((z - ref).norm() / ref.norm())
    kl = float(torch.stack([kl_at_T2(r_, z_, W, b) for r_, z_ in zip(ref.split(2048), z.split(2048))]).mean())
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        teacher_hidden(model, x)
    flops = fc.get_total_flops()
    with profile(activities=[ProfilerActivity.CUDA]) as prof, torch.no_grad():
        step(model, W, b, x)
        torch.cuda.synchronize()
    ev = [(k.key, k.device_time_total) for k in prof.key_averages() if k.device_time_total > 0]
    tot = sum(t for _, t in ev) or 1
    top = sorted(ev, key=lambda kv: -kv[1])[:6]
    tc_share = sum(t for n, t in ev if any(m in n.lower() for m in TC)) / tot
    r = dict(name=name, impl=impl, compile=compile_, batch=B, tok_per_s=B * args.seq_len / sec, tflops=flops / sec / 1e12,
             pct_ceiling=100 * flops / sec / 1e12 / ceiling, step_ms=sec * 1e3, spread_pct=100 * (max(times) / min(times) - 1),
             peak_gb=torch.cuda.max_memory_allocated() / 2**30, tensor_core_kernel_share=tc_share, warmup_s=warm, rel_l2_vs_ref=rel, kl_vs_ref=kl,
             top_kernels=[(n[:70], round(100 * t / tot, 1)) for n, t in top])
    del model, W, b, x
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="answerdotai/ModernBERT-large")
    ap.add_argument("--batches", type=lambda s: [int(v) for v in s.split(",")], default=[16, 32, 64, 128])
    ap.add_argument("--impls", type=lambda s: s.split(","), default=["sdpa", "eager"])
    ap.add_argument("--compile", type=lambda s: [bool(int(v)) for v in s.split(",")], default=[False, True])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.9)  # same loud-OOM rule as training: no silent WDDM spill
    ceiling = matmul_ceiling()
    print(f"{torch.cuda.get_device_name()}: bf16 matmul ceiling {ceiling:.1f} TFLOP/s", flush=True)
    rows = []
    for impl in args.impls:
        for comp in args.compile:
            for B in args.batches:
                try:
                    r = bench(f"{impl}{'+compile' if comp else ''}", impl, comp, B, args, ceiling)
                except torch.OutOfMemoryError:
                    print(f"  {impl} compile={comp} B={B}: OOM", flush=True)
                    continue
                rows.append(r)
                print(f"  {r['name']:<14} B={B:<4} {r['tok_per_s']:9,.0f} tok/s  {r['tflops']:5.1f} TFLOP/s ({r['pct_ceiling']:3.0f}% of ceiling)  "
                      f"{r['step_ms']:7.1f} ms  +-{r['spread_pct']:.0f}%  {r['peak_gb']:.1f} GB  tensor-core kernels {r['tensor_core_kernel_share']:.0%}  |dz|/|z| {r['rel_l2_vs_ref']:.1e}  KL vs ref {r['kl_vs_ref']:.1e}", flush=True)
    best = max(rows, key=lambda r: r["tok_per_s"])
    print(f"\nbest: {best['name']} B={best['batch']}  {best['tok_per_s']:,.0f} tok/s  {best['pct_ceiling']:.0f}% of the matmul ceiling")
    for n, p in best["top_kernels"]:
        print(f"    {p:5.1f}%  {n}")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(ceiling=ceiling, gpu=torch.cuda.get_device_name(), rows=rows), indent=1))


if __name__ == "__main__":
    main()

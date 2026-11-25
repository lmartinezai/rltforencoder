"""Halting-head features: extract once on any machine, fit heads as often as you like in seconds.

Running the frozen student at every depth is the only expensive part of training a halting head (the head itself is 1k parameters). This
script does that part and stores, per masked sequence and per depth 1..8, everything a head could use, so that fitting a head (src/halting_fit.py)
is a tiny job on the stored tensors and never touches the model again. Extraction is split into shards of `--shard_seqs` sequences, each a
pure function of (student, data seed, shard index): any number of machines can take disjoint shard ranges, and a re-run skips finished shards.

    # a worker (any GPU that can hold the 100M-parameter student; the teacher stream comes from the teacher server):
    uv run python src/halting_features.py --student runs/arm_a_merge/checkpoint-12208 --out feats/a_train --teacher_addr <teacher-host>:5555 \
        --data_seed 4242 --first_shard 0 --shards 40
    # held-out text, local teacher (no server needed):
    uv run python src/halting_features.py --student runs/arm_a_merge/checkpoint-12208 --out feats/a_arxiv --source arxiv --samples 512

Per sequence (B sequences per shard, D = 8 depths, K sampled masked tokens, H = 1024):
  seq_state [B,D,H] fp16   mean state over ALL the sequence's masked tokens          seq_loss, seq_kl, seq_change, seq_entropy, seq_maxp [B,D]
  tok_state [B,D,K,H] fp16 the state of K randomly chosen masked tokens              tok_loss, tok_kl, tok_change, tok_entropy, tok_maxp [B,D,K]
  n_voters  [B]            masked tokens in the sequence (weights for token-weighted losses)
loss = the distillation objective per token (0.6 CE + 1.6 KL_T2, what the runs trained); kl / change = distance to the previous depth's output
distribution / state (0 at depth 1); entropy / maxp = of the depth's output distribution. Everything except `loss` is available at inference
(but entropy, maxp and kl need the decoder at every depth, about 16 GFLOPs per depth and sequence against 58 per loop iteration)."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

import distill
from arch import BortForMaskedLM
from halting import depth_table_fast, pool_by_sequence
from store import TargetStore, collate_store, split_targets
from stream import RemoteTargets
from targets import make_targets, teacher_hidden

DEPTHS = 8
SCALARS = ("loss", "kl", "change", "entropy", "maxp")


def build_records(table, rows, n_seq, k, generator):
    """The per-depth tables of one batch ([D,N,...], N masked tokens of n_seq sequences, rows[N] = sequence of each) -> a dict of per-sequence tensors."""
    states, *scalars = table
    scalars = [torch.nan_to_num(x.float(), posinf=0.0) for x in scalars]  # kl and change are inf at depth 1 (nothing to compare with)
    out = {"n_voters": torch.bincount(rows, minlength=n_seq), "seq_state": pool_by_sequence(states, rows, n_seq).permute(1, 0, 2).half()}
    for name, x in zip(SCALARS, scalars, strict=True):
        out[f"seq_{name}"] = pool_by_sequence(x, rows, n_seq).permute(1, 0)
    pick = []
    for b in range(n_seq):
        idx = (rows == b).nonzero().squeeze(-1)
        take = torch.randperm(len(idx), generator=generator)[:k] if len(idx) >= k else torch.randint(len(idx), (k,), generator=generator)
        pick.append(idx[take.to(idx.device)])
    pick = torch.stack(pick)  # [B, K]
    out["tok_state"] = states[:, pick].permute(1, 0, 2, 3).half()
    for name, x in zip(SCALARS, scalars, strict=True):
        out[f"tok_{name}"] = x[:, pick].permute(1, 0, 2)
    return {key: value.cpu() for key, value in out.items()}


def merge(records):
    return {key: torch.cat([r[key] for r in records]) for key in records[0]}


def load_shards(directory):
    """All shards of a directory, concatenated -> dict of tensors (+ the meta)."""
    directory = Path(directory)
    shards = sorted(directory.glob("shard-*.pt"))
    if not shards:
        raise FileNotFoundError(f"no shards in {directory}")
    data = merge([torch.load(s, map_location="cpu") for s in shards])
    return data, json.loads((directory / "meta.json").read_text())


@torch.no_grad()
def extract_batches(student, batches, bias, args, k, seed, first_seq=0):
    """Iterate (batch, stored) pairs -> per-batch record dicts (run under bf16 autocast, as the evaluation does). The sampled tokens are a
    function of (seed, absolute index of the batch's first sequence), so a shard is the same whichever worker made it and wherever that worker
    started, as long as the workers use the same --batch_size."""
    seen = first_seq
    for batch, stored in batches:
        rows = (batch["labels"] != -100).nonzero()[:, 0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            table = depth_table_fast(student, batch, stored, bias, args, range(1, DEPTHS + 1), signals=True, confidence=True)
        yield build_records(table, rows, len(batch["labels"]), k, torch.Generator().manual_seed(seed * 1_000_003 + seen))
        seen += len(batch["labels"])


def stream_batches(addr, seed, start_seq, batch_size, buffer):
    """Teacher-server batches (the same stream a trainer would read) from sequence `start_seq` on."""
    dataset = RemoteTargets(addr, buffer, seed=seed, start_seq=start_seq)
    try:
        for rows in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_store, num_workers=0):
            yield split_targets({key: v.cuda() for key, v in rows.items()})
    finally:
        dataset.close()


def store_batches(store_dir, n, batch_size, device="cuda"):
    """Batches of a stored teacher-target split (store.py / distill.py --stage extract): the same corrupted sequences and teacher targets for every
    model, with no teacher on the GPU. n = 0: all of it."""
    store = TargetStore(store_dir)
    n = min(n, len(store)) if n else len(store)
    for start in range(0, n, batch_size):
        rows = collate_store([store[i] for i in range(start, min(start + batch_size, n))])
        yield split_targets({key: v.to(device) for key, v in rows.items()})


def local_batches(source, teacher, student, batch_size, samples, seq_len):
    """Held-out text with a local teacher: arxiv (ccdv/arxiv-summarization validation) or the FineWeb hold-out; the evaluation's masks."""
    from bench_heldout import heldout_batches

    for batch in heldout_batches(samples, seq_len, batch_size)["arxiv" if source == "arxiv" else "fineweb-holdout"]:
        batch = {key: v.cuda() for key, v in batch.items()}
        yield batch, make_targets(teacher_hidden(teacher, batch), student.decoder.weight, teacher.decoder.bias.float(), 2.0, 0, 1.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source", default="stream", choices=("stream", "arxiv", "fineweb", "store"))
    p.add_argument("--store", default="", help="--source store: a stored target split, e.g. stores/halting/train")
    p.add_argument("--teacher_addr", default="")
    p.add_argument("--data_seed", type=int, default=4242)
    p.add_argument("--first_shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1, help="stream source: shards to produce, counted from --first_shard")
    p.add_argument("--samples", type=int, default=512, help="local and store sources: sequences (store: 0 = all)")
    p.add_argument("--shard_seqs", type=int, default=256)
    p.add_argument("--tokens", type=int, default=16, help="K: masked tokens kept per sequence")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--stream_buffer", type=int, default=1024)
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    distill.configure_cuda()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    student = BortForMaskedLM.from_pretrained(a.student).cuda().eval()
    args = distill.DistillArguments(output_dir="runs/_feat", report_to="none")
    teacher = None

    if a.source == "stream":
        if not a.teacher_addr:
            raise SystemExit("--source stream needs --teacher_addr")
        done = 0
        while done < a.shards and (out / f"shard-{a.first_shard + done:06d}.pt").exists():
            done += 1  # finished shards are skipped: the stream restarts at the first missing one
        start = a.first_shard + done
        if done == a.shards:
            print(f"[features] all {a.shards} shards already exist in {out}")
            return
        bias = distill.teacher_decoder_params(SimpleNamespace(teacher=distill.TEACHER))[1].cuda()
        source = stream_batches(a.teacher_addr, a.data_seed, start * a.shard_seqs, a.batch_size, a.stream_buffer)
        first, last = start, a.first_shard + a.shards
    elif a.source == "store":
        if not a.store:
            raise SystemExit("--source store needs --store")
        bias = distill.teacher_decoder_params(SimpleNamespace(teacher=distill.TEACHER))[1].cuda()
        n = min(a.samples, len(TargetStore(a.store))) if a.samples else len(TargetStore(a.store))
        source = store_batches(a.store, a.samples, a.batch_size)
        first, last = 0, -(-n // a.shard_seqs)
    else:
        teacher = distill.build_teacher(SimpleNamespace(teacher=distill.TEACHER, device="cuda"))
        bias = teacher.decoder.bias.float()
        source = local_batches(a.source, teacher, student, a.batch_size, a.samples, 512)
        first, last = 0, -(-a.samples // a.shard_seqs)

    (out / "meta.json").write_text(json.dumps({"student": a.student, "source": a.source, "data_seed": a.data_seed, "shard_seqs": a.shard_seqs,
                                               "tokens": a.tokens, "depths": DEPTHS, "temperature": args.temperature,
                                               "weights": [args.ce_weight, args.kd_weight]}, indent=1))
    pending, shard, started, seqs = [], first, time.perf_counter(), 0
    if a.shard_seqs % a.batch_size:
        raise SystemExit("--shard_seqs must be a multiple of --batch_size")
    for records in extract_batches(student, source, bias, args, a.tokens, a.data_seed, first_seq=first * a.shard_seqs):
        pending.append(records)
        if sum(len(r["n_voters"]) for r in pending) >= a.shard_seqs:
            data = merge(pending)
            keep, rest = {k: v[:a.shard_seqs] for k, v in data.items()}, {k: v[a.shard_seqs:] for k, v in data.items()}
            torch.save(keep, out / f"shard-{shard:06d}.pt.tmp")
            (out / f"shard-{shard:06d}.pt.tmp").replace(out / f"shard-{shard:06d}.pt")
            seqs += a.shard_seqs
            print(f"[features] shard {shard} written; {seqs / (time.perf_counter() - started):.1f} sequences/s", flush=True)
            shard += 1
            pending = [rest] if len(rest["n_voters"]) else []
            if shard >= last:
                break
    if pending and a.source != "stream":  # the last, partial shard of a local source
        torch.save(merge(pending), out / f"shard-{shard:06d}.pt")
        print(f"[features] shard {shard} written (partial)")
    print("[features] done")


if __name__ == "__main__":
    main()

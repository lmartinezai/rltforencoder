"""Teacher targets as arrays: the wire/disk format shared by the network pipeline and the discrete (on-disk) pipeline.

`batch_arrays` / `rows_from_arrays` turn one teacher batch into numpy arrays and back into per-sequence rows (used by the
teacher server's frames and by the store); `collate_store` turns rows into a model batch. A store is a directory of shards
(one .npy per array, memory-mapped on read) plus meta.json. The masks are fixed at extraction time (static masking), so the
student sees exactly the corruption the teacher was asked about.
"""
import hashlib
import json
import os
import shutil
import tempfile
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from targets import Targets, make_targets, teacher_hidden

IGNORE = -100


def decoder_fingerprint(W, b):
    """Identify the exact fp32 decoder used to construct or decode targets."""
    h = hashlib.sha256()
    for tensor in (W, b):
        t = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def teacher_fingerprint(teacher):
    """Hash the actual model state, including encoder and prediction head, not just its decoder."""
    h = hashlib.sha256()
    for name, tensor in sorted(teacher.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(f"{name}:{t.dtype}:{tuple(t.shape)}".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _write_json(path, value):
    """Atomically replace a JSON file after flushing it; fsync the parent where supported."""
    path = Path(path)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            tmp = Path(f.name)
            json.dump(value, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if os.name != "nt":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _save(path, array):
    """np.save + fsync, so a DONE marker written afterwards never points at data still sitting in the page cache."""
    with open(path, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())


def batch_arrays(input_ids, labels, tg):
    """One extracted batch -> the numpy arrays a store shard (or a network frame) holds: corrupted ids [B,L], labels
    [B,L], per-sequence counts m (masked) / d (dense) and the Targets of the masked positions."""
    input_ids, labels = input_ids.cpu(), labels.cpu()
    tg = Targets(*(t.cpu() for t in tg))
    for name, t in (("input_ids", input_ids), ("sparse ids", tg.ids)):  # the arrays stored as uint16
        if t.numel() and (int(t.min()) < 0 or int(t.max()) > 0xFFFF):
            raise ValueError(f"{name} do not fit uint16 (range {int(t.min())}..{int(t.max())})")
    if not torch.isfinite(tg.hidden).all():  # bf16 -> fp16 overflows above 65504; never ship that to a trainer
        raise ValueError("non-finite teacher hidden state (fp16 overflow or a broken teacher forward)")
    m = (labels != IGNORE).sum(1)
    d = torch.stack([x.sum() for x in tg.dense.split(m.tolist())])
    return dict(input_ids=input_ids.numpy().astype(np.uint16), labels=labels.numpy().astype(np.int32),
                m=m.numpy(), d=d.numpy(), ids=tg.ids.numpy().astype(np.uint16),
                logp=tg.logp.half().numpy(), tail=tg.tail.numpy().astype(np.float32),
                dense=tg.dense.numpy(), hidden=(tg.hidden.view(torch.uint16).numpy() if tg.hidden.dtype == torch.bfloat16
                                                else tg.hidden.numpy()))


ARRAY_SPEC = {"input_ids": ("uint16", 2), "labels": ("int32", 2), "m": ("int64", 1), "d": ("int64", 1), "ids": ("uint16", 2),
              "logp": ("float16", 2), "tail": ("float32", 1), "dense": ("bool", 1), "hidden": ("float16", 2)}


def check_arrays(a):
    """Validate the arrays of one teacher batch (a frame off the wire) before they become rows: right names, dtypes and ranks,
    and counts that agree with each other. A mismatch would otherwise slip through as misaligned targets (wrong KL, no error)."""
    if set(a) != set(ARRAY_SPEC):
        raise ValueError(f"batch arrays {sorted(a)} != expected {sorted(ARRAY_SPEC)}")
    for k, (dtype, ndim) in ARRAY_SPEC.items():
        if (a[k].dtype != np.dtype(dtype) and not (k == "hidden" and a[k].dtype == np.dtype("uint16"))) or a[k].ndim != ndim:
            raise ValueError(f"array {k!r} is {a[k].dtype}/{a[k].ndim}-d, expected {dtype}/{ndim}-d")
    B = len(a["input_ids"])
    m, d, dense = a["m"].astype(np.int64), a["d"].astype(np.int64), a["dense"]
    if a["labels"].shape != a["input_ids"].shape or m.shape != (B,) or d.shape != (B,):
        raise ValueError("labels/m/d do not match input_ids")
    if not np.array_equal((a["labels"] != IGNORE).sum(1), m) or (d < 0).any() or (d > m).any():
        raise ValueError("per-sequence masked/dense counts do not match the labels")
    if len(dense) != m.sum():
        raise ValueError(f"{len(dense)} dense flags for {int(m.sum())} masked positions")
    cs = np.concatenate([[0], np.cumsum(dense, dtype=np.int64)])
    m_off = np.concatenate([[0], np.cumsum(m)])
    if not np.array_equal(cs[m_off[1:]] - cs[m_off[:-1]], d):
        raise ValueError("dense flags do not match the per-sequence dense counts")
    sparse = int((m - d).sum())
    if len(a["hidden"]) != int(d.sum()) or not (len(a["ids"]) == len(a["logp"]) == len(a["tail"]) == sparse):
        raise ValueError("target arrays do not match the dense/sparse counts")


def rows_from_arrays(a):
    """Inverse view of `batch_arrays`: one item per sequence, in the format `TargetStore.__getitem__` returns (views,
    no copies), so `collate_store` and the trainer cannot tell a network batch from a stored one."""
    off = lambda c: np.concatenate([[0], np.cumsum(c)]).astype(np.int64)
    m_off, d_off, s_off = off(a["m"]), off(a["d"]), off(a["m"] - a["d"])
    return [dict(input_ids=a["input_ids"][r], labels=a["labels"][r], kd_ids=a["ids"][s_off[r]:s_off[r + 1]],
                 kd_logp=a["logp"][s_off[r]:s_off[r + 1]], kd_tail=a["tail"][s_off[r]:s_off[r + 1]],
                 kd_dense=a["dense"][m_off[r]:m_off[r + 1]], kd_hidden=a["hidden"][d_off[r]:d_off[r + 1]])
            for r in range(len(a["m"]))]


class StoreWriter:
    """Writes shards as batches arrive. A shard is complete only once its DONE marker exists, so an interrupted
    extraction can resume: complete shards are kept, a half-written one is discarded."""

    def __init__(self, root, meta, shard_seqs=2048, resume=False):
        self.root, self.meta, self.shard_seqs = Path(root), dict(meta), shard_seqs
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "manifest.json"
        existing = list(self.root.glob("shard*"))
        if resume and existing:
            if not manifest.exists() or json.loads(manifest.read_text()) != self.meta:
                raise ValueError("cannot resume extraction: store manifest is missing or incompatible")
        elif existing:
            raise FileExistsError(f"target shards already exist in {self.root}")
        _write_json(manifest, self.meta)
        committed = sorted(d.name for d in existing if (d / "DONE").exists())
        if committed != [f"shard{i:05d}" for i in range(len(committed))]:
            raise ValueError("committed shard sequence has gaps")
        self.buf, self.n_seq, self.shards, self.totals = [], 0, 0, {"masked": 0, "sparse": 0, "dense": 0}
        for d in sorted(self.root.glob("shard*")):
            done = d / "DONE"
            if resume and done.exists():
                c = json.loads(done.read_text())
                self.n_seq, self.shards = self.n_seq + c.pop("n_seq"), self.shards + 1
                self.totals = {k: self.totals[k] + c[k] for k in self.totals}
            else:
                shutil.rmtree(d)  # incomplete shard on resume

    def add(self, input_ids, labels, tg):
        """One extracted batch: corrupted ids [B,L], labels [B,L] and the Targets of its masked positions."""
        self.buf.append(batch_arrays(input_ids, labels, tg))
        self.n_seq += len(input_ids)
        if sum(len(b["m"]) for b in self.buf) >= self.shard_seqs:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        cat = {k: np.concatenate([b[k] for b in self.buf]) for k in self.buf[0]}
        d = self.root / f"shard{self.shards:05d}"
        d.mkdir(exist_ok=True)
        off = lambda c: np.concatenate([[0], np.cumsum(c)]).astype(np.int64)
        for k in ("input_ids", "labels", "ids", "logp", "tail", "dense", "hidden"):
            _save(d / f"{k}.npy", cat[k])
        _save(d / "m_off.npy", off(cat["m"]))
        _save(d / "d_off.npy", off(cat["d"]))
        _save(d / "s_off.npy", off(cat["m"] - cat["d"]))
        counts = {"masked": int(cat["m"].sum()), "sparse": int((cat["m"] - cat["d"]).sum()), "dense": int(cat["d"].sum())}
        self.totals = {k: self.totals[k] + v for k, v in counts.items()}
        _write_json(d / "DONE", {"n_seq": len(cat["m"]), **counts})  # commit after data is fsynced
        self.shards, self.buf = self.shards + 1, []

    def close(self):
        self.flush()
        nbytes = sum(f.stat().st_size for f in self.root.rglob("*.npy"))
        meta = self.meta | dict(n_seq=self.n_seq, shards=self.shards, bytes=nbytes, **self.totals)
        _write_json(self.root / "meta.json", meta)
        return meta


class TargetStore(torch.utils.data.Dataset):
    """Map-style view of a store: one item per packed sequence (so Trainer shuffling and resume just work)."""

    FIELDS = ("input_ids", "labels", "ids", "logp", "tail", "dense", "hidden", "m_off", "d_off", "s_off")

    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        dirs = sorted(self.root.glob("shard*"))
        self.shards = [{k: np.load(d / f"{k}.npy", mmap_mode="r") for k in self.FIELDS} for d in dirs]
        self.starts = np.cumsum([0] + [len(s["input_ids"]) for s in self.shards])
        self._verify(dirs)

    def _verify(self, dirs):
        """Check shard boundaries and position counts against labels and target arrays."""
        for d, s in zip(dirs, self.shards, strict=True):
            done = d / "DONE"
            if not done.exists():
                raise ValueError(f"{d} was never committed (no DONE marker): the store is incomplete")
            n = len(s["input_ids"])
            ok = (json.loads(done.read_text())["n_seq"] == n == len(s["labels"]) and
                  all(len(s[k]) == n + 1 for k in ("m_off", "d_off", "s_off")) and
                  (len(s["dense"]), len(s["hidden"]), len(s["ids"])) == (int(s["m_off"][-1]), int(s["d_off"][-1]), int(s["s_off"][-1])))
            if not ok:
                raise ValueError(f"{d} is inconsistent (truncated or mixed-up arrays)")
            m, dense, sparse = (np.asarray(s[k]) for k in ("m_off", "d_off", "s_off"))
            if any(o[0] != 0 or np.any(np.diff(o) < 0) for o in (m, dense, sparse)):
                raise ValueError(f"{d} has inconsistent offsets")
            counts = np.diff(m)
            expected = np.count_nonzero(s["labels"] != IGNORE, axis=1)
            dense_prefix = np.concatenate(([0], np.cumsum(s["dense"], dtype=np.int64)))
            if (not np.array_equal(counts, expected) or
                    not np.array_equal(np.diff(dense), dense_prefix[m[1:]] - dense_prefix[m[:-1]]) or
                    not np.array_equal(np.diff(sparse), counts - np.diff(dense)) or
                    s["logp"].shape != s["ids"].shape or len(s["tail"]) != len(s["ids"])):
                raise ValueError(f"{d} has inconsistent per-sequence targets")
        if int(self.starts[-1]) != self.meta["n_seq"]:
            raise ValueError(f"{self.root}: shards hold {int(self.starts[-1])} sequences but meta.json says {self.meta['n_seq']}")

    def __getstate__(self):  # DataLoader workers: pickling memmaps would copy the whole store into every worker; reopen instead
        return {"root": str(self.root)}

    def __setstate__(self, state):
        self.__init__(state["root"])

    def __len__(self):
        return int(self.starts[-1])

    def __getitem__(self, i):
        s = int(np.searchsorted(self.starts, i, side="right") - 1)
        sh, r = self.shards[s], i - int(self.starts[s])
        m0, m1, d0, d1, s0, s1 = (int(x) for x in (sh["m_off"][r], sh["m_off"][r + 1], sh["d_off"][r],
                                                    sh["d_off"][r + 1], sh["s_off"][r], sh["s_off"][r + 1]))
        return dict(input_ids=np.array(sh["input_ids"][r]), labels=np.array(sh["labels"][r]),
                    kd_ids=np.array(sh["ids"][s0:s1]), kd_logp=np.array(sh["logp"][s0:s1]),
                    kd_tail=np.array(sh["tail"][s0:s1]), kd_dense=np.array(sh["dense"][m0:m1]),
                    kd_hidden=np.array(sh["hidden"][d0:d1]))


def collate_store(rows, targets=True):
    """Batch of store items -> model inputs (input_ids/attention_mask/labels) + kd_* target tensors.
    Targets are concatenated in row-major order of the labelled positions, matching the student's sparse logits.
    targets=False yields only the corrupted inputs (joint pipelines that recompute the teacher live)."""
    cat = lambda k, dt: torch.from_numpy(np.concatenate([r[k] for r in rows])).to(dt)
    ids = torch.from_numpy(np.stack([r["input_ids"] for r in rows]).astype(np.int64))
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                 labels=cat("labels", torch.int64).reshape(len(rows), -1))
    if targets:
        if len({r["kd_hidden"].dtype for r in rows}) != 1:
            raise ValueError("cannot mix BF16 and FP16 hidden state encodings in a batch")
        hidden = torch.from_numpy(np.concatenate([r["kd_hidden"] for r in rows]))
        if hidden.dtype == torch.uint16:
            hidden = hidden.view(torch.bfloat16)
        batch |= dict(kd_ids=cat("kd_ids", torch.int64), kd_logp=cat("kd_logp", torch.float32),
                      kd_tail=cat("kd_tail", torch.float32), kd_dense=cat("kd_dense", torch.bool),
                      kd_hidden=hidden)
    return batch


def split_targets(batch):
    """Pop the kd_* entries from a collated batch -> (model inputs, Targets)."""
    kd = {k: batch.pop(k) for k in ("kd_ids", "kd_logp", "kd_tail", "kd_dense", "kd_hidden")}
    return batch, Targets(kd["kd_ids"], kd["kd_logp"], kd["kd_tail"], kd["kd_dense"], kd["kd_hidden"])


def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


@torch.no_grad()
def keep_labels(labels, keep, seed, first_seq):
    """Keep a fraction `keep` of each sequence's supervised positions (at least one) and drop the rest from the loss: their inputs stay
    corrupted, only their targets are not stored. A pure function of (seed, absolute sequence index, labels), so a resumed extraction
    reproduces it. labels [B, L] -> a copy."""
    if not 0 < keep <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    out = labels.clone()
    if keep == 1:
        return out
    for r in range(len(labels)):
        pos = (labels[r] != IGNORE).nonzero().squeeze(-1)
        order = torch.randperm(len(pos), generator=torch.Generator().manual_seed(seed * 1_000_003 + first_seq + r))
        out[r, pos[order[max(1, round(keep * len(pos))):]]] = IGNORE
    return out


def extract(teacher, W, b, sequences, collate, out_dir, args, n_seq, batch_size=64, log=None, resume=False):
    """Teacher pass: corrupt `n_seq` packed sequences, run the teacher, write targets. Returns meta.

    sequences: iterable of {"input_ids": [...]} that is deterministic across runs (seeded stream), so that
    resume=True replays and verifies previously committed batches through the collator to restore its RNG state."""
    if n_seq < 1 or batch_size < 1:
        raise ValueError("n_seq and batch_size must be positive")
    tokenizer = getattr(collate, "tokenizer", None)
    tokenizer_state = tokenizer.backend_tokenizer.to_str() if tokenizer is not None and hasattr(tokenizer, "backend_tokenizer") else None
    meta = {"teacher": args.teacher, "T": args.temperature, "topk": args.topk, "topp": args.topp,
            "seq_len": args.seq_len, "mask_prob": args.mask_prob, "seed": args.seed, "hidden": int(W.shape[1]),
            "decoder_sha256": decoder_fingerprint(W, b), "teacher_sha256": teacher_fingerprint(teacher),
            "hidden_format": "bf16-bits-or-fp16-v1",
            "tokenizer_sha256": hashlib.sha256(tokenizer_state.encode()).hexdigest() if tokenizer_state else None,
            "batch_size": batch_size, "keep_fraction": getattr(args, "keep_fraction", 1.0),
            "data": {k: getattr(args, k) for k in ("dataset", "dataset_config", "dataset_revision", "train_split", "eval_split", "streaming",
                                                      "holdout_pct", "id_column", "shuffle_buffer")}}
    keep = meta["keep_fraction"]

    def corrupt(rows, first_seq):
        x = collate(rows)
        x["labels"] = keep_labels(x["labels"], keep, args.seed, first_seq)
        return x

    w = StoreWriter(out_dir, meta, resume=resume)
    if w.n_seq > n_seq:
        raise ValueError("existing store exceeds requested sequence count")
    it = iter(sequences)
    if w.n_seq:
        if w.n_seq % batch_size and w.n_seq < n_seq:
            raise ValueError("cannot resume a partial extraction batch with identical masking")
        stored = ((x, y) for d in sorted(Path(out_dir).glob("shard*")) if (d / "DONE").exists()
                  for x, y in zip(np.load(d / "input_ids.npy", mmap_mode="r"),
                                  np.load(d / "labels.npy", mmap_mode="r"), strict=True))
        replayed = 0
        for batch in batched(islice(it, w.n_seq), batch_size):
            x = corrupt(batch, replayed)
            for ids, labels in zip(x["input_ids"].numpy(), x["labels"].numpy(), strict=True):
                old_ids, old_labels = next(stored)
                if not np.array_equal(ids, old_ids) or not np.array_equal(labels, old_labels):
                    raise ValueError("cannot resume: source sequences or collator output changed")
                replayed += 1
        if replayed != w.n_seq:
            raise ValueError("cannot resume: source ended before committed sequences")
    for batch in batched(islice(it, n_seq - w.n_seq), batch_size):
        x = {k: v.to(teacher.device) for k, v in corrupt(batch, w.n_seq).items()}
        tg = make_targets(teacher_hidden(teacher, x), W, b, args.temperature, args.topk, args.topp)
        w.add(x["input_ids"], x["labels"], tg)
        if log:
            log(w.n_seq)
    if w.n_seq != n_seq:
        w.flush()  # retain complete work for a later resume, without publishing a completed meta.json
        raise ValueError(f"source ended after {w.n_seq} sequences; requested {n_seq}")
    return w.close()

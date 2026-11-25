import numpy as np
import pytest
import torch
from datasets import Dataset

from data import is_holdout
from store import StoreWriter, TargetStore, batch_arrays, collate_store, split_targets
from targets import decode, make_targets, target_kl

V, H, L, K = 200, 16, 24, 8


def fake_batch(B, gen):
    """Corrupted ids, labels (30% supervised), and the matching Targets, as extract() would produce them."""
    labels = torch.randint(0, V, (B, L), generator=gen)
    labels[torch.rand(B, L, generator=gen) > 0.3] = -100
    ids = torch.randint(0, V, (B, L), generator=gen)
    W, b = torch.randn(V, H, generator=gen), torch.randn(V, generator=gen) * 0.1
    z = (torch.randn(int((labels != -100).sum()), H, generator=gen) * 3).bfloat16().float()
    return ids, labels, make_targets(z, W, b, 2.0, K, 0.9), (z, W, b)


def build_store(tmp_path, n_batches=5, B=7, shard_seqs=16):
    gen = torch.Generator().manual_seed(0)
    w = StoreWriter(tmp_path, dict(K=K), shard_seqs=shard_seqs)
    batches = [fake_batch(B, gen) for _ in range(n_batches)]
    for ids, labels, tg, _ in batches:
        w.add(ids, labels, tg)
    return w.close(), batches


def test_roundtrip_alignment_across_shards(tmp_path):
    meta, batches = build_store(tmp_path)
    assert meta["n_seq"] == 35 and meta["shards"] > 1
    assert meta["sparse"] + meta["dense"] == meta["masked"] > 0 and meta["dense"] > 0 and meta["sparse"] > 0
    store = TargetStore(tmp_path)
    assert len(store) == 35
    ids = torch.cat([b[0] for b in batches]).numpy()
    for i in (0, 6, 7, 15, 16, 34):  # shard and batch boundaries
        assert np.array_equal(store[i]["input_ids"], ids[i])
    # every batch we read back, in any grouping, lines up with the labelled positions
    for rows in ([0, 1, 2], [5, 20, 33], list(range(35))):
        batch = collate_store([store[i] for i in rows])
        inputs, tg = split_targets(batch)
        labels = inputs["labels"]
        assert labels.shape == (len(rows), L) and inputs["input_ids"].dtype == torch.int64
        assert tg.dense.shape == (int((labels != -100).sum()),)
        assert tg.ids.shape[0] == int((~tg.dense).sum()) and tg.hidden.shape[0] == int(tg.dense.sum())


def test_stored_targets_reproduce_the_original_kl(tmp_path):
    """Same loss from the in-memory Targets and from the ones that went through fp16/uint16 disk storage."""
    _, batches = build_store(tmp_path, n_batches=3, B=8, shard_seqs=8)
    store = TargetStore(tmp_path)
    for j, (_, labels, tg, (z, W, b)) in enumerate(batches):
        _, back = split_targets(collate_store([store[8 * j + r] for r in range(8)]))
        assert torch.equal(back.dense, tg.dense) and torch.equal(back.ids, tg.ids)
        assert torch.equal(back.hidden, tg.hidden) and torch.equal(back.tail, tg.tail)
        s = decode(z, W, b) + torch.randn(len(z), V)
        assert torch.allclose(target_kl(s, back, W, b, 2.0), target_kl(s, tg, W, b, 2.0), rtol=5e-3, atol=5e-4)


def test_holdout_is_deterministic_and_about_pct():
    ids = [f"<urn:uuid:{i}>" for i in range(20_000)]
    flags = [is_holdout(i, 1) for i in ids]
    assert flags == [is_holdout(i, 1) for i in ids] and 0.005 < np.mean(flags) < 0.015
    assert not any(is_holdout(i, 0) for i in ids[:1000])


def test_stream_state_resume_never_repeats_consumed_data():
    mk = lambda: Dataset.from_dict({"x": list(range(2000))}).to_iterable_dataset(num_shards=4).shuffle(
        seed=0, buffer_size=32)
    ds = mk()
    it = iter(ds)
    consumed = [next(it)["x"] for _ in range(300)]
    state = ds.state_dict()
    resumed = mk()
    resumed.load_state_dict(state)
    after = [r["x"] for r in resumed]
    assert len(set(consumed)) == 300 and not set(consumed) & set(after)  # nothing consumed is replayed
    assert len(after) >= 2000 - 300 - 4 * 32  # lost at most the shuffle buffers


# ------------------------------------------------------------------ extract(): interrupted + resumed run (CPU, tiny)

def _tiny_teacher():
    from transformers import ModernBertConfig, ModernBertForMaskedLM

    torch.manual_seed(0)
    cfg = ModernBertConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           vocab_size=300, pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1,
                           sep_token_id=2, global_attn_every_n_layers=2, local_attention=8)
    return ModernBertForMaskedLM(cfg).eval()


class _Collate:
    """Stand-in for the HF MLM collator: 30% of interior positions supervised, deterministic per generator."""

    def __init__(self, seed):
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, rows):
        ids = torch.tensor([r["input_ids"] for r in rows])
        labels = ids.clone()
        keep = torch.rand(ids.shape, generator=self.gen) < 0.3
        keep[:, [0, -1]] = False
        labels[~keep] = -100
        ids = ids.masked_fill(keep, 3)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def test_extract_resumes_an_interrupted_run(tmp_path):
    from distill import DistillArguments
    from store import extract

    teacher = _tiny_teacher()
    W, b = teacher.decoder.weight.detach().float(), teacher.decoder.bias.detach().float()
    seqs = [{"input_ids": [1, *range(10 + i, 10 + i + 14), 2]} for i in range(40)]
    args = DistillArguments(output_dir="x", seq_len=16, topk=0, temperature=2.0)
    run = lambda out, n, **kw: extract(teacher, W, b, iter(seqs), _Collate(0), out, args, n, batch_size=4, **kw)

    full = run(tmp_path / "full", 40)
    assert full["n_seq"] == 40 and full["masked"] == full["dense"] > 0
    part = run(tmp_path / "cut", 20)  # "interrupted" after 20 sequences...
    (tmp_path / "cut" / "meta.json").unlink()
    (tmp_path / "cut" / "shard00099").mkdir()  # ...leaving a half-written shard behind
    (tmp_path / "cut" / "shard00099" / "ids.npy").write_bytes(b"junk")
    resumed = run(tmp_path / "cut", 40, resume=True)
    assert part["n_seq"] == 20 and resumed["n_seq"] == 40 and not (tmp_path / "cut" / "shard00099").exists()
    a, c = TargetStore(tmp_path / "full"), TargetStore(tmp_path / "cut")
    assert len(a) == len(c) == 40
    for i in range(40):  # every sequence present exactly once and in order; targets match the supervised positions
        r = c[i]
        assert np.array_equal(a[i]["input_ids"], r["input_ids"])
        assert np.array_equal(a[i]["labels"], r["labels"])
        assert np.array_equal(a[i]["kd_hidden"], r["kd_hidden"])
        original = np.where(r["labels"] != -100, r["labels"], r["input_ids"])
        assert original.tolist() == seqs[i]["input_ids"]
        assert r["kd_dense"].shape[0] == (r["labels"] != -100).sum() == r["kd_hidden"].shape[0]


def test_store_refuses_overwrite_and_incompatible_resume(tmp_path):
    w = StoreWriter(tmp_path, {"teacher": "a"}, shard_seqs=1)
    ids, labels, tg, _ = fake_batch(2, torch.Generator().manual_seed(0))
    w.add(ids, labels, tg)
    with pytest.raises(FileExistsError):
        StoreWriter(tmp_path, {"teacher": "a"})
    with pytest.raises(ValueError, match="manifest"):
        StoreWriter(tmp_path, {"teacher": "b"}, resume=True)
    assert (tmp_path / "shard00000" / "DONE").exists()


def test_store_rejects_ids_that_do_not_fit_uint16(tmp_path):
    import pytest

    w = StoreWriter(tmp_path, {})
    gen = torch.Generator().manual_seed(0)
    ids, labels, tg, _ = fake_batch(2, gen)
    ids[0, 0] = 70_000
    with pytest.raises(ValueError, match="uint16"):
        w.add(ids, labels, tg)


def test_non_finite_hidden_state_is_rejected_at_the_source():
    ids, labels, tg, _ = fake_batch(2, torch.Generator().manual_seed(0))
    sparse_free = tg._replace(hidden=torch.full((3, H), 7e4).half())  # 7e4 overflows fp16 -> inf
    assert not torch.isfinite(sparse_free.hidden).all()
    with pytest.raises(ValueError, match="non-finite"):
        batch_arrays(ids, labels, sparse_free)
    batch_arrays(ids, labels, tg)  # finite is fine


def test_sparse_token_ids_must_fit_uint16_too():
    ids, labels, tg, _ = fake_batch(2, torch.Generator().manual_seed(0))
    assert tg.ids.numel()
    big = tg._replace(ids=tg.ids.clone().index_fill_(0, torch.tensor([0]), 70_000))  # input ids are small: only these wrap
    with pytest.raises(ValueError, match="sparse ids"):
        batch_arrays(ids, labels, big)


def test_store_pickles_as_a_path_not_as_its_data(tmp_path):
    import pickle

    build_store(tmp_path)
    store = TargetStore(tmp_path)
    blob = pickle.dumps(store)  # what a spawned DataLoader worker receives
    assert len(blob) < 1000
    again = pickle.loads(blob)
    assert len(again) == len(store) and np.array_equal(again[20]["input_ids"], store[20]["input_ids"])
    assert np.array_equal(again[20]["kd_hidden"], store[20]["kd_hidden"])


def test_store_integrity_is_checked_on_open(tmp_path):
    import json

    build_store(tmp_path)
    TargetStore(tmp_path)  # a good store opens
    meta = json.loads((tmp_path / "meta.json").read_text())
    (tmp_path / "meta.json").write_text(json.dumps(meta | {"n_seq": meta["n_seq"] + 1}))
    with pytest.raises(ValueError, match="meta.json says"):
        TargetStore(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    shard = max(tmp_path.glob("shard*"))
    np.save(shard / "dense.npy", np.load(shard / "dense.npy")[:-1])
    with pytest.raises(ValueError, match="inconsistent"):
        TargetStore(tmp_path)
    shard2 = min(tmp_path.glob("shard*"))
    (shard2 / "DONE").unlink()
    with pytest.raises(ValueError, match="never committed"):
        TargetStore(tmp_path)


def test_negative_ids_are_rejected():
    ids, labels, tg, _ = fake_batch(2, torch.Generator().manual_seed(0))
    ids[0, 0] = -1
    with pytest.raises(ValueError, match="uint16"):
        batch_arrays(ids, labels, tg)
    ids[0, 0] = 0
    tg.ids[0, 0] = -1
    with pytest.raises(ValueError, match="uint16"):
        batch_arrays(ids, labels, tg)


def test_bf16_hidden_bit_patterns_roundtrip_through_store_and_wire_rows(tmp_path):
    import stream
    from store import check_arrays, rows_from_arrays

    W, b = torch.eye(4), torch.zeros(4)
    z = torch.tensor([[100000., 1e-9, -1e-6, 1.]], dtype=torch.bfloat16)
    tg = make_targets(z, W, b, 1.0, 0, 0.99)
    ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[-100, 2, -100]])
    arrays = batch_arrays(ids, labels, tg)
    assert arrays["hidden"].dtype == np.uint16
    check_arrays(arrays)
    payload = bytearray(b"".join(bytes(m.cast("B")) for m in map(memoryview, stream.encode_batch("train", arrays)) if m.nbytes))
    _, _, decoded = stream.decode_frame(payload)
    check_arrays(decoded)
    writer = StoreWriter(tmp_path, {})
    writer.add(ids, labels, tg)
    writer.close()
    for rows in (rows_from_arrays(decoded), [TargetStore(tmp_path)[0]]):
        hidden = collate_store(rows)["kd_hidden"]
        assert hidden.dtype == torch.bfloat16 and torch.equal(hidden, z)


def test_fp16_logp_negative_infinity_survives_store_and_kl(tmp_path):
    W, b = torch.eye(4), torch.zeros(4)
    z = torch.tensor([[0., -70000., -80000., -90000.]])
    tg = make_targets(z, W, b, 1., 4, 0.)
    ids, labels = torch.tensor([[1]]), torch.tensor([[1]])
    w = StoreWriter(tmp_path, {})
    w.add(ids, labels, tg)
    w.close()
    _, restored = split_targets(collate_store([TargetStore(tmp_path)[0]]))
    logits = torch.zeros(1, 4, requires_grad=True)
    loss = target_kl(logits, restored, W, b, 1.).sum()
    loss.backward()
    assert torch.allclose(loss, torch.tensor(4.).log()) and torch.isfinite(logits.grad).all()


def test_interior_offsets_are_checked(tmp_path):
    build_store(tmp_path)
    path = min(tmp_path.glob("shard*")) / "m_off.npy"
    offsets = np.load(path)
    offsets[1] = offsets[0]
    np.save(path, offsets)
    with pytest.raises(ValueError, match="inconsistent per-sequence"):
        TargetStore(tmp_path)


def test_short_extraction_is_not_published(tmp_path):
    from distill import DistillArguments
    from store import extract

    teacher = _tiny_teacher()
    args = DistillArguments(output_dir="x", seq_len=16)
    seqs = [{"input_ids": [1, *range(10, 24), 2]} for _ in range(4)]
    with pytest.raises(ValueError, match="source ended after 4"):
        extract(teacher, teacher.decoder.weight, teacher.decoder.bias, seqs, _Collate(0), tmp_path, args, 8, batch_size=4)
    assert not (tmp_path / "meta.json").exists()
    assert (tmp_path / "shard00000" / "DONE").exists()


def test_resume_rejects_changed_dataset_and_source(tmp_path):
    from distill import DistillArguments
    from store import extract

    teacher = _tiny_teacher()
    args = DistillArguments(output_dir="x", seq_len=16)
    seqs = [{"input_ids": [1, *range(10 + i, 24 + i), 2]} for i in range(8)]
    run = lambda rows, a, n, **kw: extract(teacher, teacher.decoder.weight, teacher.decoder.bias, rows, _Collate(0),
                                           tmp_path, a, n, batch_size=4, **kw)
    run(seqs, args, 4)
    args.dataset = "different"
    with pytest.raises(ValueError, match="manifest"):
        run(seqs, args, 8, resume=True)
    args.dataset = "HuggingFaceFW/fineweb"
    changed = list(seqs)
    changed[0] = {"input_ids": [1, *range(40, 54), 2]}
    with pytest.raises(ValueError, match="source sequences"):
        run(changed, args, 8, resume=True)
    with torch.no_grad():
        teacher.head.dense.weight[0, 0] += 1
    with pytest.raises(ValueError, match="manifest"):
        run(seqs, args, 8, resume=True)


# ---------------------------------------------------------------- crash recovery at every commit boundary

class Crash(Exception):
    pass


def _batches(n=5, B=7):
    gen = torch.Generator().manual_seed(0)
    return [fake_batch(B, gen)[:3] for _ in range(n)]


def _write_all(root, batches, crash_at=None, resume=False, crash_after=False):
    """Inject failure immediately before or after one array save or JSON commit."""
    import store

    count, real_save, real_json = [0], store._save, store._write_json

    def wrap(real):
        def inner(*a, **k):
            if crash_at is not None and count[0] == crash_at and not crash_after:
                raise Crash(count[0])
            result = real(*a, **k)
            if crash_at is not None and count[0] == crash_at and crash_after:
                raise Crash(count[0])
            count[0] += 1
            return result
        return inner

    store._save, store._write_json = wrap(real_save), wrap(real_json)
    try:
        w = StoreWriter(root, dict(K=K), shard_seqs=16, resume=resume)
        for ids, labels, tg in batches[w.n_seq // 7:]:
            w.add(ids, labels, tg)
        return w.close(), count[0]
    finally:
        store._save, store._write_json = real_save, real_json


def _contents(root):
    s = TargetStore(root)
    return {k: np.concatenate([sh[k] for sh in s.shards]) for k in ("input_ids", "labels", "ids", "logp", "tail", "dense", "hidden")}


@pytest.mark.parametrize("crash_after", [False, True], ids=["before", "after"])
def test_a_crash_at_any_commit_boundary_is_recoverable_and_never_publishes_a_partial_store(tmp_path, crash_after):
    batches = _batches()
    meta, n_ops = _write_all(tmp_path / "ref", batches)
    reference = _contents(tmp_path / "ref")
    assert n_ops > 20  # manifest, 10 array saves + DONE per shard, meta.json
    for crash_at in range(n_ops):
        root = tmp_path / f"crash{crash_at}"
        with pytest.raises(Crash):
            _write_all(root, batches, crash_at=crash_at, crash_after=crash_after)
        # the interrupted store is never readable as complete: no meta.json (or, if the crash was after it, a whole store)
        if (root / "meta.json").exists():
            visible = _contents(root)
            assert visible.keys() == reference.keys()
            assert all(np.array_equal(visible[k], reference[k]) for k in reference), crash_at
        else:
            with pytest.raises(FileNotFoundError):
                TargetStore(root)
        # recovery: keep committed shards, drop the half-written one, replay the rest -> identical to the clean run
        resumed, _ = _write_all(root, batches, resume=True)
        recovered = _contents(root)
        assert resumed["n_seq"] == meta["n_seq"] and resumed["shards"] == meta["shards"], crash_at
        assert all(np.array_equal(recovered[k], reference[k]) for k in reference), crash_at
        assert not list(root.glob("tmp*")) and not list(root.glob("*.tmp")), crash_at


def test_a_failed_atomic_replace_leaves_the_old_file_and_no_temp_files(tmp_path, monkeypatch):
    import os

    import store

    target = tmp_path / "meta.json"
    store._write_json(target, {"v": 1})
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        store._write_json(target, {"v": 2})
    monkeypatch.undo()
    assert target.read_text() == '{"v": 1}' and [p.name for p in tmp_path.iterdir()] == ["meta.json"]


# ---------------------------------------------------------------- validation of arrays and shards

def _good_arrays():
    ids, labels, tg, _ = fake_batch(4, torch.Generator().manual_seed(3))
    return batch_arrays(ids, labels, tg)


def test_check_arrays_names_each_kind_of_inconsistency():
    from store import check_arrays

    good = _good_arrays()
    check_arrays(good)
    bad = {
        "do not match input_ids": good | {"labels": good["labels"][:, :-1]},
        "do not match the labels": good | {"m": good["m"] + 1},
        "dense flags for": good | {"dense": good["dense"][:-1]},
        "dense flags do not match": good | {"d": np.where(good["d"] > 0, good["d"] - 1, good["d"] + 1).astype(good["d"].dtype)},
    }
    for message, arrays in bad.items():
        with pytest.raises(ValueError, match=message):
            check_arrays(arrays)


def test_resume_refuses_a_gap_in_committed_shards(tmp_path):
    w = StoreWriter(tmp_path, {"k": 1}, shard_seqs=1)
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        ids, labels, tg, _ = fake_batch(2, gen)
        w.add(ids, labels, tg)
    (tmp_path / "shard00001" / "DONE").unlink()  # shards 0 and 2 committed, 1 lost
    with pytest.raises(ValueError, match="gaps"):
        StoreWriter(tmp_path, {"k": 1}, resume=True)


def test_a_store_with_inconsistent_offsets_is_rejected_on_open(tmp_path):
    build_store(tmp_path)
    path = tmp_path / "shard00000" / "m_off.npy"
    off = np.load(path)
    off[0] = 1  # offsets must start at zero
    np.save(path, off)
    with pytest.raises(ValueError, match="inconsistent offsets"):
        TargetStore(tmp_path)


def test_extract_argument_and_partial_batch_resume_errors(tmp_path):
    from distill import DistillArguments
    from store import extract

    teacher = _tiny_teacher()
    W, b = teacher.decoder.weight.detach().float(), teacher.decoder.bias.detach().float()
    seqs = [{"input_ids": [1, *range(10 + i, 10 + i + 14), 2]} for i in range(40)]
    args = DistillArguments(output_dir="x", seq_len=16, topk=0, temperature=2.0)
    with pytest.raises(ValueError, match="must be positive"):
        extract(teacher, W, b, iter(seqs), _Collate(0), tmp_path / "a", args, 0)
    extract(teacher, W, b, iter(seqs), _Collate(0), tmp_path / "b", args, 10, batch_size=4)  # last batch is partial
    (tmp_path / "b" / "meta.json").unlink()
    with pytest.raises(ValueError, match="partial extraction batch"):  # 10 stored, batches of 4: masks cannot be replayed
        extract(teacher, W, b, iter(seqs), _Collate(0), tmp_path / "b", args, 40, batch_size=4, resume=True)


def test_extract_resume_refuses_a_shorter_request_and_a_shorter_source(tmp_path):
    from distill import DistillArguments
    from store import extract

    teacher = _tiny_teacher()
    W, b = teacher.decoder.weight.detach().float(), teacher.decoder.bias.detach().float()
    seqs = [{"input_ids": [1, *range(10 + i, 10 + i + 14), 2]} for i in range(40)]
    args = DistillArguments(output_dir="x", seq_len=16, topk=0, temperature=2.0)
    extract(teacher, W, b, iter(seqs), _Collate(0), tmp_path / "s", args, 8, batch_size=4)
    (tmp_path / "s" / "meta.json").unlink()
    with pytest.raises(ValueError, match="exceeds requested"):
        extract(teacher, W, b, iter(seqs), _Collate(0), tmp_path / "s", args, 4, batch_size=4, resume=True)
    with pytest.raises(ValueError, match="source ended before committed"):
        extract(teacher, W, b, iter(seqs[:5]), _Collate(0), tmp_path / "s", args, 12, batch_size=4, resume=True)


def test_keep_labels_keeps_a_deterministic_fraction_of_each_sequence_and_at_least_one():
    from store import keep_labels

    labels = torch.full((6, 40), -100)
    labels[:, 5:25] = 7  # 20 supervised positions each
    labels[3] = -100
    labels[3, 9] = 7  # one supervised position: it must survive
    a = keep_labels(labels, 0.25, 11, 100)
    assert ((a != -100).sum(1) == torch.tensor([5, 5, 5, 1, 5, 5])).all()
    assert ((a != -100) & (labels == -100)).sum() == 0  # only ever drops, never adds
    assert torch.equal(a, keep_labels(labels, 0.25, 11, 100)) and not torch.equal(a, keep_labels(labels, 0.25, 11, 101))
    assert torch.equal(a[2:], keep_labels(labels[2:], 0.25, 11, 102))  # a function of the absolute sequence index, not the batch it is in
    assert torch.equal(keep_labels(labels, 1.0, 0, 0), labels)
    with pytest.raises(ValueError, match="keep_fraction"):
        keep_labels(labels, 0.0, 0, 0)


def test_extract_with_a_keep_fraction_stores_fewer_targets_and_resumes_identically(tmp_path):
    from distill import DistillArguments
    from store import extract

    teacher = _tiny_teacher()
    W, b = teacher.decoder.weight.detach().float(), teacher.decoder.bias.detach().float()
    seqs = [{"input_ids": [1, *range(10 + i, 10 + i + 14), 2]} for i in range(40)]
    run = lambda out, n, keep, **kw: extract(teacher, W, b, iter(seqs), _Collate(0), out, DistillArguments(output_dir="x", seq_len=16, topk=0, keep_fraction=keep),
                                             n, batch_size=4, **kw)
    whole, half = run(tmp_path / "whole", 40, 1.0), run(tmp_path / "half", 40, 0.5)
    assert whole["keep_fraction"] == 1.0 and half["keep_fraction"] == 0.5 and 0.4 * whole["masked"] < half["masked"] < 0.7 * whole["masked"]
    run(tmp_path / "cut", 20, 0.5)
    (tmp_path / "cut" / "meta.json").unlink()
    resumed = run(tmp_path / "cut", 40, 0.5, resume=True)
    assert resumed["masked"] == half["masked"]
    a, c = TargetStore(tmp_path / "half"), TargetStore(tmp_path / "cut")
    for i in range(40):
        assert np.array_equal(a[i]["labels"], c[i]["labels"]) and np.array_equal(a[i]["kd_hidden"], c[i]["kd_hidden"])
        assert a[i]["kd_hidden"].shape[0] == (a[i]["labels"] != -100).sum()

"""Feature shards: records are per-sequence, deterministic in (seed, sequence index), and consistent with the per-depth tables. Tiny random models, CPU."""
import pytest
import torch

import distill
import halting
import halting_features as hf
from test_halting import batch_and_targets, models  # noqa: F401  (the fixture)


def table_for(models):  # noqa: F811
    student, teacher = models
    batch, stored = batch_and_targets(student, teacher)
    args = distill.DistillArguments(output_dir="runs/test")
    table = halting.depth_table_fast(student, batch, stored, teacher.decoder.bias, args, range(1, 9), signals=True, confidence=True)
    return batch, table


def test_records_shapes_pooling_and_finite_values(models):  # noqa: F811
    batch, table = table_for(models)
    rows = (batch["labels"] != -100).nonzero()[:, 0]
    rec = hf.build_records(table, rows, 3, 4, torch.Generator().manual_seed(0))
    D, H = 8, table[0].shape[-1]
    assert rec["seq_state"].shape == (3, D, H) and rec["tok_state"].shape == (3, D, 4, H) and rec["tok_loss"].shape == (3, D, 4)
    assert rec["seq_state"].dtype == torch.float16 and rec["n_voters"].tolist() == torch.bincount(rows).tolist()
    assert all(torch.isfinite(v.float()).all() for v in rec.values())  # inf at depth 1 became 0
    assert (rec["seq_kl"][:, 0] == 0).all() and (rec["seq_change"][:, 0] == 0).all()
    # the sequence mean loss is the mean of the per-token losses of that sequence
    want = table[1][:, rows == 1].mean(1)
    assert torch.allclose(rec["seq_loss"][1], want, atol=1e-6)
    # sampled tokens' losses are among that sequence's token losses
    for b in range(3):
        pool = table[1][:, rows == b]
        for k in range(4):
            assert any(torch.allclose(rec["tok_loss"][b, :, k], pool[:, j], atol=1e-6) for j in range(pool.shape[1]))


def test_token_sampling_is_a_function_of_the_seed_only(models):  # noqa: F811
    batch, table = table_for(models)
    rows = (batch["labels"] != -100).nonzero()[:, 0]
    a = hf.build_records(table, rows, 3, 4, torch.Generator().manual_seed(7))
    b = hf.build_records(table, rows, 3, 4, torch.Generator().manual_seed(7))
    c = hf.build_records(table, rows, 3, 4, torch.Generator().manual_seed(8))
    assert all(torch.equal(a[k], b[k]) for k in a) and not torch.equal(a["tok_loss"], c["tok_loss"])


def test_sequences_with_fewer_masked_tokens_than_k_are_sampled_with_replacement(models):  # noqa: F811
    batch, table = table_for(models)
    rows = (batch["labels"] != -100).nonzero()[:, 0]
    rec = hf.build_records(table, rows, 3, 40, torch.Generator().manual_seed(0))  # K larger than any sequence's voters
    assert rec["tok_state"].shape[2] == 40 and torch.isfinite(rec["tok_loss"]).all()


def test_shard_roundtrip_and_load(tmp_path, models):  # noqa: F811
    batch, table = table_for(models)
    rows = (batch["labels"] != -100).nonzero()[:, 0]
    rec = hf.build_records(table, rows, 3, 4, torch.Generator().manual_seed(0))
    for i in range(2):
        torch.save(rec, tmp_path / f"shard-{i:06d}.pt")
    (tmp_path / "meta.json").write_text('{"k": 4}')
    data, meta = hf.load_shards(tmp_path)
    assert data["seq_loss"].shape == (6, 8) and meta == {"k": 4}
    with pytest.raises(FileNotFoundError):
        hf.load_shards(tmp_path / "nothing")


def test_store_batches_replay_a_stored_split_in_order_with_targets_on_the_requested_device(tmp_path):
    from distill import DistillArguments
    from store import TargetStore, extract
    from test_store import _Collate, _tiny_teacher

    teacher = _tiny_teacher()
    seqs = [{"input_ids": [1, *range(10 + i, 10 + i + 14), 2]} for i in range(10)]
    args = DistillArguments(output_dir="x", seq_len=16, topk=0, temperature=2.0)
    extract(teacher, teacher.decoder.weight.detach().float(), teacher.decoder.bias.detach().float(), iter(seqs), _Collate(0), tmp_path, args, 10, batch_size=5)
    batches = list(hf.store_batches(tmp_path, 0, 4, device="cpu"))
    assert [len(b["input_ids"]) for b, _ in batches] == [4, 4, 2]
    store = TargetStore(tmp_path)
    flat = torch.cat([b["input_ids"] for b, _ in batches])
    assert torch.equal(flat, torch.stack([torch.tensor(store[i]["input_ids"].astype("int64")) for i in range(10)]))
    batch, stored = batches[0]
    assert stored.hidden.shape[0] == int((batch["labels"] != -100).sum()) and stored.ids.shape[0] == 0  # dense targets for every supervised position
    assert len(list(hf.store_batches(tmp_path, 3, 2, device="cpu"))) == 2  # n limits the sequences: 3 sequences in batches of 2 -> 2 batches

from itertools import islice
from types import SimpleNamespace

import pytest
from datasets import Dataset

import data
from data import is_holdout, make_eval_dataset, make_train_dataset, pack_dataset

DOCS, LEN, SEQ = 120, 40, 10  # each document is LEN copies of one unique token


class Tok:
    cls_token_id, sep_token_id = 1, 2

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[int(t) for t in text.split()] for text in texts]}


def corpus():
    return Dataset.from_dict({"id": [f"doc{k}" for k in range(DOCS)],
                              "text": [" ".join([str(1000 + k)] * LEN) for k in range(DOCS)]})


@pytest.fixture()
def args(monkeypatch):
    monkeypatch.setattr(data, "_load", lambda a, split: corpus().to_iterable_dataset() if a.streaming else corpus())
    return SimpleNamespace(dataset="x", dataset_config=None, dataset_revision="", train_split="train", eval_split="train",
                           streaming=True, holdout_pct=10, id_column="id", shuffle_buffer=50, seq_len=SEQ, seed=3,
                           eval_samples=20)


def doc_tokens(seqs):
    return {t for s in seqs for t in s["input_ids"] if t >= 1000}


def test_packing_cuts_fixed_sequences_with_special_tokens():
    seqs = list(pack_dataset(corpus(), Tok(), SEQ))
    assert all(len(s["input_ids"]) == SEQ and s["input_ids"][0] == 1 and s["input_ids"][-1] == 2 for s in seqs)
    assert len(seqs) == DOCS * LEN // (SEQ - 2)


def test_holdout_is_document_level_and_disjoint_from_training(args):
    held = {1000 + k for k in range(DOCS) if is_holdout(f"doc{k}", args.holdout_pct)}
    assert held and len(held) < DOCS
    train = doc_tokens(make_train_dataset(Tok(), args))
    evaluation = doc_tokens(make_eval_dataset(Tok(), args))
    assert evaluation <= held and not train & held and train | evaluation


def test_train_stream_is_a_pure_function_of_the_seed(args):
    def first(seed, n=60):
        return [tuple(s["input_ids"]) for s in islice(make_train_dataset(Tok(), SimpleNamespace(**{**vars(args), "seed": seed})), n)]

    assert first(3) == first(3)
    assert first(3) != first(4)  # a different seed is a different order


def test_eval_set_is_fixed_and_capped(args):
    a, b = list(make_eval_dataset(Tok(), args)), list(make_eval_dataset(Tok(), args))
    assert a == b and 0 < len(a) <= args.eval_samples
    assert len(make_eval_dataset(Tok(), args, n=2)) == 2


def test_eval_split_path_when_there_is_no_holdout(args):
    args.holdout_pct, args.streaming = 0, False
    assert len(make_eval_dataset(Tok(), args, n=7)) == 7
    args.streaming = True
    assert len(make_eval_dataset(Tok(), args, n=7)) == 7


def test_non_streaming_training_data_keeps_corpus_order_and_no_holdout_uses_every_document(args):
    args.streaming, args.holdout_pct = False, 0
    seqs = [tuple(s["input_ids"]) for s in make_train_dataset(Tok(), args)]
    assert seqs == [tuple(s["input_ids"]) for s in pack_dataset(corpus(), Tok(), SEQ)]  # no shuffle, no filtering


def test_load_passes_revision_and_streaming_to_the_hub(monkeypatch):
    seen = {}
    monkeypatch.setattr(data, "load_dataset", lambda *a, **k: seen.update(a=a, k=k) or "ds")
    args = SimpleNamespace(dataset="d", dataset_config="c", streaming=True, dataset_revision="")
    assert data._load(args, "train") == "ds" and seen["a"] == ("d", "c")
    assert seen["k"] == {"split": "train", "streaming": True, "revision": None}
    args.dataset_revision = "abc"
    data._load(args, "train")
    assert seen["k"]["revision"] == "abc"  # a pinned commit is passed through

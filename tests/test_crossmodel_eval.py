"""crossmodel_eval: partition masking scores every body token exactly once; bytes/BPB bookkeeping; the text cache. Tokenizer only (no model)."""
import json
import math

import pytest
import torch
from transformers import AutoTokenizer

import crossmodel_eval as ce

TEXT = "The quick brown fox jumps over the lazy dog, and then it naps under a very old oak tree near the river bank."


@pytest.fixture(scope="module", params=["answerdotai/ModernBERT-large", "FacebookAI/roberta-base", "distilbert/distilbert-base-cased"])
def tok(request):
    return AutoTokenizer.from_pretrained(request.param)


def test_every_body_token_is_masked_in_exactly_one_group_and_specials_never(tok):
    batch, mask, true, covered, n_body = ce.partition_batch(tok, TEXT, 8, 512)
    assert mask.sum().item() == n_body and (mask.sum(0) <= 1).all()  # each position is masked in at most one of the 8 copies, all body tokens at once
    assert (batch[mask] == tok.mask_token_id).all() and (batch[~mask] == true[~mask]).all()
    specials = torch.tensor(tok.get_special_tokens_mask(true[0].tolist(), already_has_special_tokens=True)).bool()
    assert not mask.any(0)[specials].any()
    assert covered <= len(TEXT.encode()) and covered >= len(TEXT.encode()) - 2  # the tokens cover (almost) all of the text


def test_group_assignment_is_interleaved(tok):
    _, mask, _, _, n_body = ce.partition_batch(tok, TEXT, 4, 512)
    assert [int(g.sum()) for g in mask] == [len(range(g, n_body, 4)) for g in range(4)]


def test_nats_is_the_summed_negative_log_probability_of_the_true_tokens():
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]))[None].repeat(2, 1, 1)  # [2 rows, 2 positions, 3 tokens]
    true, mask = torch.tensor([[0, 2], [1, 2]]), torch.tensor([[True, True], [False, True]])
    got = ce.nats(logits[mask], true, mask)
    assert got == pytest.approx(-(math.log(0.5) + math.log(0.8) + math.log(0.8)))


def test_truncation_counts_only_the_covered_bytes(tok):
    _, _, _, covered_short, n_short = ce.partition_batch(tok, TEXT, 8, 12)
    _, _, _, covered_long, n_long = ce.partition_batch(tok, TEXT, 8, 512)
    assert n_short < n_long and covered_short < covered_long


def test_text_cache_is_reused_and_a_changed_request_is_rebuilt(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ce, "_first", lambda stream, key, n, chars, keep=lambda r: True: calls.append(1) or [f"{key}{i}".ljust(chars, "x") for i in range(n)])
    monkeypatch.setattr(ce, "load_dataset", lambda *a, **k: type("S", (), {"shuffle": lambda self, **kw: self})())
    path = tmp_path / "texts.json"
    first = ce.load_texts(path, 3, 20)
    assert set(first) == set(ce.DATASETS) and len(calls) == 4
    assert ce.load_texts(path, 3, 20) == first and len(calls) == 4  # served from the cache
    ce.load_texts(path, 5, 20)
    assert len(calls) == 8 and json.loads(path.read_text(encoding="utf-8"))["n"] == 5


def test_deberta_legacy_head_keys_are_mapped_to_the_names_transformers_uses():
    state = {"deberta.embeddings.word_embeddings.weight": torch.zeros(2, 2), "lm_predictions.lm_head.bias": torch.ones(3),
             "lm_predictions.lm_head.dense.weight": torch.ones(2, 2), "lm_predictions.lm_head.dense.bias": torch.ones(2),
             "lm_predictions.lm_head.LayerNorm.weight": torch.ones(2), "lm_predictions.lm_head.LayerNorm.bias": torch.ones(2)}
    mapped = ce.legacy_deberta_head(state)
    assert set(mapped) == {"cls.predictions.bias", "cls.predictions.decoder.bias", "cls.predictions.transform.dense.weight",
                           "cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias"}
    assert mapped["cls.predictions.bias"] is mapped["cls.predictions.decoder.bias"]
    with pytest.raises(ValueError, match="lm_head tensors"):
        ce.legacy_deberta_head({"deberta.x": torch.zeros(1)})  # a checkpoint without the head must not pass silently

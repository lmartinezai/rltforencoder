"""zeroshot_nli: label words, prompts, scoring and calibration arithmetic, and the guarantee that only dev splits can be read. Tokenizers only (no model)."""
import inspect

import numpy as np
import pytest
from transformers import AutoTokenizer

import zeroshot_nli as z

TOKENIZERS = ["answerdotai/ModernBERT-large", "FacebookAI/roberta-base", "distilbert/distilbert-base-cased", "microsoft/deberta-base"]


@pytest.fixture(scope="module", params=TOKENIZERS)
def tok(request):
    return AutoTokenizer.from_pretrained(request.param)


def test_every_label_word_is_one_distinct_token_for_every_tokenizer(tok):
    for task in z.TASKS.values():
        ids = z.label_word_ids(tok, task["words"])
        assert len(ids) == len(task["words"]) and len(set(ids)) == len(ids)
        assert all(len(tok.encode(" " + w, add_special_tokens=False)) == 1 or len(tok.encode(w, add_special_tokens=False)) == 1 for w in task["words"])


def test_a_word_that_is_not_a_single_token_is_an_error():
    tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
    with pytest.raises(ValueError, match="single token"):
        z.label_word_ids(tok, ("Yes", "Antidisestablishmentarianism"))


def test_prompts_have_exactly_one_mask_and_cut_a_long_first_sentence_not_the_mask(tok):
    long_first = "word " * 1000
    for template in z.TEMPLATES.values():
        prompt = z.build_prompt(template, tok, long_first, "A short hypothesis.")
        enc = tok(prompt, truncation=True, max_length=384)
        assert enc["input_ids"].count(tok.mask_token_id) == 1
        assert prompt.count(tok.mask_token) == 1 and prompt.endswith(("hypothesis.", tok.mask_token + " ."))


def test_only_validation_splits_can_be_read():
    for bad in ("train", "test", "train[:10]"):
        with pytest.raises(ValueError, match="validation"):
            z.dev_split("rte", bad)
    assert "split=\"train\"" not in inspect.getsource(z) and "'train'" not in inspect.getsource(z).replace("'train[:10]'", "")


def test_scoring_raw_calibrated_and_collapse_statistics():
    # three labels; the model always prefers label 0 (collapse), but relative to the content-free prior the evidence points at the true label
    prior = np.log(np.array([0.7, 0.2, 0.1]))
    labels = np.array([0, 1, 2, 1, 0, 2])
    evidence = np.log(np.array([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0], [0.1, 1.0, 0.1], [1.0, 0.1, 0.1], [0.1, 0.1, 1.0]]))
    logprobs = evidence + prior[None, :] * 2  # the strong prior for label 0 swamps the evidence in the raw argmax
    s = z.score(logprobs, labels, prior * 2)
    assert s["calibrated"] == 1.0 and s["raw"] < s["calibrated"]
    assert s["raw_top_label_share"] > s["calibrated_top_label_share"] and s["majority"] == pytest.approx(2 / 6) and s["n"] == 6

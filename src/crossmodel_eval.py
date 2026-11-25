"""Masked-LM comparison across models with different tokenizers: bits per byte of text, on held-out text from four sources, at every loop depth.

    uv run python src/crossmodel_eval.py --out results/mlm_crossmodel \
        --models modernbert-large=hf:answerdotai/ModernBERT-large roberta-base=hf:FacebookAI/roberta-base \
                 distilbert-cased=hf:distilbert/distilbert-base-cased a_s1=bort:runs/small/a_s1/checkpoint-3072 c_s1=flat:runs/small_c/c_s1/checkpoint-3072

Per-token loss is not comparable across tokenizers (a model with finer tokens has easier tokens), so the unit is the text itself. Every text is cut to its
first `--chars` characters and scored by PARTITION MASKING: the body tokens are split into K interleaved groups (token j is in group j mod K); K forward
passes each replace one group by [MASK] and read the log-probability the model gives the true token. Every token is scored exactly once, at a mask rate of
1/K (12.5%, like RoBERTa/BERT pre-training); the sum over tokens divided by the UTF-8 bytes of the covered text is the bits per byte (BPB, lower is
better). It is the masked-LM cousin of bits-per-character; it is not the same as a left-to-right BPB (a masked model sees both sides, so its BPB is lower
than a causal model's, and subword models are helped by the visible pieces of a split word), so compare models to each other, not to causal-LM numbers.
Two limits of the comparison across tokenizers: a WordPiece model (DistilBERT) does not predict whitespace, which is in its favour (its out-of-vocabulary
[UNK] tokens are too rare to matter: 0.14% of its tokens on the code texts, under 0.01% elsewhere); and a text longer than `--max_len` tokens is cut, which shortens the python-code texts for RoBERTa and DeBERTa by 0.5% (the bytes
are counted per model, so each BPB is still exact for the text that model scored). Students and references are scored under bf16 autocast (fp32 changes a
student's BPB by about 0.0003).

Texts (`--texts` of each, 1000 characters from the start of the document, fixed and cached in `<out>/texts.json`): FineWeb hold-out (the training
distribution's held-out documents), arXiv articles, English Wikipedia, Python source (codeparrot-clean-valid). The reference models have seen some of
these in pre-training (Wikipedia for all of them, code for ModernBERT); the students have only ever seen FineWeb, through the teacher.

Models: `hf:` any AutoModelForMaskedLM checkpoint (DeBERTa v1's head is mapped from its original key names and it is scored in fp32, its attention masking overflows in bf16; DeBERTa-v3's lm_head is untrained, v3-base cannot be scored), `bort:` a looped Bort (every depth 1..8 from ONE pass through the loop: the state after each iteration
is decoded through the coda), `flat:` the flat baseline."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

import distill
import pins
from arch import BortForMaskedLM
from compare_arch import FlatForMaskedLM
from data import is_holdout
from halting import logits_by_depth

DATASETS = ("fineweb-holdout", "arxiv", "wikipedia", "python-code")
DEPTHS = range(1, 9)


# ---------------------------------------------------------------- texts

def _first(stream, key, n, chars, keep=lambda row: True):
    out = []
    for row in stream:
        if keep(row) and len(row[key]) >= chars:
            out.append(row[key][:chars])
            if len(out) == n:
                return out
    raise RuntimeError(f"only {len(out)} of {n} usable documents")


def load_texts(path, n, chars):
    """The fixed evaluation texts: read from `path` if it exists (so every run, and every machine, scores identical text), else built and saved."""
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["n"] == n and saved["chars"] == chars:
            return saved["texts"]
    texts = {
        "fineweb-holdout": _first(load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True, **pins.kw("HuggingFaceFW/fineweb")), "text", n, chars,
                                  lambda r: is_holdout(r["id"], 1)),
        "arxiv": _first(load_dataset("ccdv/arxiv-summarization", split="validation", streaming=True, **pins.kw("ccdv/arxiv-summarization")), "article", n, chars),
        "wikipedia": _first(load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, **pins.kw("wikimedia/wikipedia")).shuffle(seed=0, buffer_size=2000), "text", n, chars),
        "python-code": _first(load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True, **pins.kw("codeparrot/codeparrot-clean-valid")), "content", n, chars),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n": n, "chars": chars, "texts": texts}, ensure_ascii=False), encoding="utf-8")
    return texts


# ---------------------------------------------------------------- scoring

def partition_batch(tok, text, k, max_len):
    """-> (ids [K, L], mask [K, L] bool, true ids [K, L], bytes covered, n body tokens): K copies of the text, copy g with group g masked."""
    enc = tok(text, return_offsets_mapping=True, return_special_tokens_mask=True, truncation=True, max_length=max_len)
    ids = torch.tensor(enc["input_ids"])
    body = [i for i, special in enumerate(enc["special_tokens_mask"]) if not special]
    batch = ids.repeat(k, 1)
    mask = torch.zeros_like(batch, dtype=torch.bool)
    for j, position in enumerate(body):
        mask[j % k, position] = True
    batch = batch.masked_fill(mask, tok.mask_token_id)
    covered = len(text[: enc["offset_mapping"][body[-1]][1]].encode("utf-8"))
    return batch, mask, ids.repeat(k, 1), covered, len(body)


def nats(logits, true, mask):
    """Summed negative log-probability of the true tokens at the masked positions. logits [N, V] in mask's row-major order."""
    return -torch.log_softmax(logits.float(), -1).gather(1, true[mask].unsqueeze(1)).sum().item()


def masked_logits(kind, model, ids, attention, selected):
    """{depth: logits [N_selected, V]}: every loop depth of a looped Bort from one pass through its loop, depth 1 only for any other model."""
    if kind == "bort":
        return logits_by_depth(model, ids, attention, selected, DEPTHS)
    return {1: model(input_ids=ids, attention_mask=attention).logits[selected]}


@torch.no_grad()
def score_texts(kind, model, tok, texts, k, max_len, device, bf16=True):
    """-> {depth: {"nats": ..., "tokens": ..., "bytes": ...}} summed over texts (depth 1 only for models without a loop)."""
    totals = {}
    for text in texts:
        batch, mask, true, covered, n_tokens = partition_batch(tok, text, k, max_len)
        batch, mask, true = batch.to(device), mask.to(device), true.to(device)
        attention = torch.ones_like(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            per_depth = masked_logits(kind, model, batch, attention, mask)
        for d, logits in per_depth.items():
            t = totals.setdefault(d, {"nats": 0.0, "tokens": 0, "bytes": 0})
            t["nats"] += nats(logits, true, mask)
            t["tokens"] += n_tokens
            t["bytes"] += covered
    return totals


def legacy_deberta_head(state):
    """DeBERTa checkpoints store the masked-LM head under the original code's names (lm_predictions.lm_head.*); transformers' DebertaForMaskedLM looks for
    cls.predictions.* and silently leaves the head randomly initialised (11-17 nats instead of ~1). -> the tensors under the names transformers uses
    (the decoder weight is tied to the word embeddings)."""
    out = {}
    for key, value in state.items():
        if key.startswith("lm_predictions.lm_head."):
            tail = key[len("lm_predictions.lm_head."):]
            if tail == "bias":
                out["cls.predictions.bias"] = out["cls.predictions.decoder.bias"] = value
            else:
                out["cls.predictions.transform." + tail] = value
    if len(out) != 6:
        raise ValueError(f"expected the 5 lm_head tensors of a DeBERTa checkpoint, found {len(out)} mapped tensors")
    return out


def load_model(kind, path, device):
    if kind == "hf":
        model = AutoModelForMaskedLM.from_pretrained(path, **pins.kw(path))
        tok = AutoTokenizer.from_pretrained(path, **pins.kw(path))
        if model.config.model_type == "deberta":  # DeBERTa v1 has a trained head under its original names; v3 does not (replaced-token detection)
            from huggingface_hub import hf_hub_download

            state = torch.load(hf_hub_download(path, "pytorch_model.bin", **pins.kw(path)), map_location="cpu", weights_only=True)
            model.load_state_dict(legacy_deberta_head(state), strict=False)
    else:
        model = (BortForMaskedLM if kind == "bort" else FlatForMaskedLM).from_pretrained(path)
        model.sparse_prediction = False  # full logits: there is no label tensor here
        tok = AutoTokenizer.from_pretrained(distill.TEACHER, **pins.kw(distill.TEACHER))
    return model.to(device).eval(), tok


def uses_bf16(model):
    """DeBERTa's attention masking overflows in bf16: it is scored in fp32, every other model under bf16 autocast."""
    return not model.config.model_type.startswith("deberta")


def score_models(specs, raw_path, score):
    """Score every NAME=KIND:PATH in `specs` with `score(name, kind, path) -> json-able result` into the dict stored at `raw_path`, saved after each
    model; a model already in the file is skipped, so an interrupted run continues where it stopped."""
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    for spec in specs:
        name, _, rest = spec.partition("=")
        kind, _, path = rest.partition(":")
        if name in raw:
            print(f"[{name}] already scored, skipped")
            continue
        raw[name] = score(name, kind, path)
        raw_path.write_text(json.dumps(raw, indent=1))
        torch.cuda.empty_cache()
    print(f"[done] {raw_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="NAME=KIND:PATH with KIND hf | bort | flat")
    p.add_argument("--out", default="results/mlm_crossmodel")
    p.add_argument("--texts", type=int, default=128)
    p.add_argument("--chars", type=int, default=1000)
    p.add_argument("--groups", type=int, default=8, help="K: tokens are scored in K interleaved masked passes (mask rate 1/K)")
    p.add_argument("--max_len", type=int, default=512)
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    distill.configure_cuda()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = load_texts(out / "texts.json", a.texts, a.chars)

    def score(name, kind, path):
        model, tok = load_model(kind, path, "cuda")
        result = {}
        for dataset in DATASETS:
            totals = score_texts(kind, model, tok, texts[dataset], a.groups, a.max_len, "cuda", bf16=uses_bf16(model))
            for depth, t in totals.items():
                result.setdefault(str(depth), {})[dataset] = t
            print(f"[{name}] {dataset:16s} BPB " + " ".join(f"d{d}:{t['nats'] / math.log(2) / t['bytes']:.4f}" for d, t in totals.items())
                  + f"  ({totals[1]['tokens'] / totals[1]['bytes']:.3f} tokens/byte)")
        return result

    score_models(a.models, out / "raw.json", score)


if __name__ == "__main__":
    main()

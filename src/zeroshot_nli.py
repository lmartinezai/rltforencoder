"""Zero-shot NLI with a masked LM: no training on anything, GLUE's dev sets only.

    uv run python src/zeroshot_nli.py --out results/zeroshot_nli --models roberta-base=hf:FacebookAI/roberta-base a_s1=bort:runs/small/a_s1/checkpoint-3072 ...

The model is shown a prompt with one [MASK] and asked which label word it prefers there (a "cloze" classifier, as in PET / LM-BFF): the premise, a question
mark, the mask, a comma and the hypothesis; the label words are Yes (entailment), Maybe (neutral), No (contradiction) for MNLI and Yes / No (entailment / not
entailment) for QNLI and RTE. Two fixed prompts, declared here before any model was run and both reported (nothing is picked on the dev set):

    T1:  {premise} ? [MASK] , {hypothesis}
    T2:  {premise} Question: {hypothesis} ? Answer: [MASK] .

For QNLI the "premise" is the question and the "hypothesis" the sentence (LM-BFF's order). Each example is scored by the log-probability of each label word at
the mask. `raw` takes the argmax; `calibrated` first subtracts, per label word, the log-probability the same model gives on a content-free prompt
(premise = hypothesis = "N/A"), the usual remedy for a model that prefers a word whatever it reads (Zhao et al. 2021); it uses no labels. Zero-shot cloze NLI
is near chance for most encoders, so the majority-class rate and the share of predictions that fall on one label are reported too.

Looped Bort models are scored at every depth 1..8 from one pass through the loop. DeBERTa v1 uses its original masked-LM head (see crossmodel_eval.py);
DeBERTa-v3 has no trained one. Only the validation splits are ever read (`dev_split` refuses anything else)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

import distill
import pins
from crossmodel_eval import DEPTHS, load_model, masked_logits, score_models, uses_bf16

TEMPLATES = {"T1": "{premise} ? {mask} , {hypothesis}", "T2": "{premise} Question: {hypothesis} ? Answer: {mask} ."}
TASKS = {  # dev splits, columns (first, second), label words in label-id order
    "mnli": {"splits": ("validation_matched", "validation_mismatched"), "columns": ("premise", "hypothesis"), "words": ("Yes", "Maybe", "No")},
    "qnli": {"splits": ("validation",), "columns": ("question", "sentence"), "words": ("Yes", "No")},
    "rte": {"splits": ("validation",), "columns": ("sentence1", "sentence2"), "words": ("Yes", "No")},
}
MAX_WORDS = 160  # the first-sentence side is cut to this many words so the mask is never truncated away


def dev_split(task, split):
    """The dev examples of a GLUE task. Anything but a validation split is refused: nothing here may touch training data."""
    if not split.startswith("validation"):
        raise ValueError(f"only validation splits may be read, not {split!r}")
    return load_dataset("nyu-mll/glue", task, split=split, **pins.kw("nyu-mll/glue"))


def label_word_ids(tok, words):
    """One token id per label word: ' Yes' (BPE, after a space) or 'Yes' (word piece); a word that is not a single token is an error, not a guess."""
    ids = []
    for word in words:
        for variant in (" " + word, word):
            encoded = tok.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])
                break
        else:
            raise ValueError(f"label word {word!r} is not a single token for {tok.name_or_path}")
    if len(set(ids)) != len(ids):
        raise ValueError("two label words share a token")
    return ids


def build_prompt(template, tok, first, second):
    first = " ".join(first.split()[:MAX_WORDS])
    return template.format(premise=first, hypothesis=second, mask=tok.mask_token)


@torch.no_grad()
def label_logprobs(kind, model, tok, prompts, word_ids, device, bf16, max_len=384, batch_size=16):
    """-> {depth: array [N, C]}: log-probability of each label word at the mask, for every prompt (depth 1 only for models without a loop)."""
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out = {d: np.zeros((len(prompts), len(word_ids)), dtype=np.float32) for d in (DEPTHS if kind == "bort" else (1,))}
    ids_t = torch.tensor(word_ids, device=device)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        enc = tok([prompts[i] for i in chunk], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        ids, attention = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        selected = ids == tok.mask_token_id
        if not (selected.sum(1) == 1).all():
            raise ValueError("every prompt must contain exactly one mask after tokenisation (truncation cut one?)")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            per_depth = masked_logits(kind, model, ids, attention, selected)
        for d, logits in per_depth.items():
            out[d][chunk] = torch.log_softmax(logits.float(), -1)[:, ids_t].cpu().numpy()
    return out


def score(logprobs, labels, content_free):
    """-> accuracy raw / calibrated, the share of predictions on the most predicted label (raw / calibrated), and the majority-class rate."""
    labels = np.asarray(labels)
    raw = logprobs.argmax(1)
    cal = (logprobs - content_free[None, :]).argmax(1)
    share = lambda guess: float(np.bincount(guess, minlength=logprobs.shape[1]).max() / len(guess))
    return {"raw": float((raw == labels).mean()), "calibrated": float((cal == labels).mean()), "raw_top_label_share": share(raw),
            "calibrated_top_label_share": share(cal), "majority": float(np.bincount(labels).max() / len(labels)), "n": len(labels)}


def run_model(name, kind, path, tasks, device="cuda"):
    model, tok = load_model(kind, path, device)
    bf16 = uses_bf16(model)
    results = {}
    for task in tasks:
        spec = TASKS[task]
        word_ids = label_word_ids(tok, spec["words"])
        for template_name, template in TEMPLATES.items():
            free = label_logprobs(kind, model, tok, [build_prompt(template, tok, "N/A", "N/A")], word_ids, device, bf16)
            for split in spec["splits"]:
                data = dev_split(task, split)
                prompts = [build_prompt(template, tok, a, b) for a, b in zip(data[spec["columns"][0]], data[spec["columns"][1]], strict=True)]
                lp = label_logprobs(kind, model, tok, prompts, word_ids, device, bf16)
                for depth, values in lp.items():
                    results.setdefault(str(depth), {}).setdefault(f"{task}/{split}", {})[template_name] = score(values, data["label"], free[depth][0])
        print(f"[{name}] {task} done", flush=True)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="NAME=KIND:PATH with KIND hf | bort | flat")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--out", default="results/zeroshot_nli")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    distill.configure_cuda()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    score_models(a.models, out / "raw.json", lambda name, kind, path: run_model(name, kind, path, a.tasks))


if __name__ == "__main__":
    main()

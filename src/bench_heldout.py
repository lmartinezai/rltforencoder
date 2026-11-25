"""Held-out benchmark of distilled students on text from a different distribution than the training corpus (FineWeb):
arXiv articles (ccdv/arxiv-summarization, validation split, streamed), plus a FineWeb hold-out control, at every recurrence depth.

    uv run python src/bench_heldout.py --models A=runs/arm_a_merge/checkpoint-12208 B=runs/arm_b_nomerge/checkpoint-12208 \
        C=runs/arm_c_noloop/checkpoint-12208:flat

Every model sees the same masked batches (fixed collator seed). Per model and depth, over all masked positions: masked-LM cross-entropy
against the real tokens, forward KL(teacher || student) at T=1 and T=2 over the full vocabulary, the training objective (0.6 CE + 1.6 KL_T2, what the
runs' own eval logged), and top-1 accuracy. The teacher's own
cross-entropy is the reference. A `:flat` model has no recurrence (one row, depth ignored). A few minutes on one GPU."""
import argparse
import json
import sys
from itertools import islice
from types import SimpleNamespace

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

import distill
import pins
from arch import BortForMaskedLM
from compare_arch import FlatForMaskedLM
from data import make_collator, make_eval_dataset, pack_dataset
from targets import full_kl, teacher_hidden

OBJECTIVE = SimpleNamespace(ce_weight=0.6, kd_weight=0.4, temperature=2.0)  # the training objective of every run here (distill.py's defaults)


def arxiv_sequences(tokenizer, n, seq_len, name, text_column, split):
    """The first n packed sequences of a streamed academic corpus (a fixed, reproducible slice of its first documents)."""
    ds = load_dataset(name, split=split, streaming=True, **pins.kw(name))
    columns = list(next(iter(ds)))
    ds = ds.rename_column(text_column, "text")
    columns = ["text" if c == text_column else c for c in columns]
    return Dataset.from_list(list(islice(pack_dataset(ds, tokenizer, seq_len, columns=columns), n)))


def fineweb_sequences(n, seq_len):
    """The training run's own hold-out (the 1% of FineWeb documents that never reach training): the in-distribution control."""
    args = SimpleNamespace(dataset="HuggingFaceFW/fineweb", dataset_config="sample-10BT", dataset_revision=pins.revision("HuggingFaceFW/fineweb") or "", train_split="train",
                           streaming=True, holdout_pct=1, id_column="id", seq_len=seq_len, eval_samples=n)
    return make_eval_dataset(AutoTokenizer.from_pretrained(distill.TEACHER, **pins.kw(distill.TEACHER)), args, n)


def masked_batches(sequences, tokenizer, batch_size, mask_prob, seed):
    collate = make_collator(tokenizer, SimpleNamespace(mask_prob=mask_prob), seed=seed)
    rows = [{"input_ids": r["input_ids"]} for r in sequences]
    return [collate(rows[i:i + batch_size]) for i in range(0, len(rows), batch_size)]


def heldout_batches(samples, seq_len=512, batch_size=4, mask_prob=0.3, arxiv=("ccdv/arxiv-summarization", "article", "validation")):
    """The two held-out sets every evaluation here uses, as masked batches (one fixed collator seed, so every model sees the same masks):
    {"arxiv": [...], "fineweb-holdout": [...]}. `arxiv` = (dataset, text column, split)."""
    tok = AutoTokenizer.from_pretrained(distill.TEACHER, **pins.kw(distill.TEACHER))
    sets = {"arxiv": arxiv_sequences(tok, samples, seq_len, *arxiv), "fineweb-holdout": fineweb_sequences(samples, seq_len)}
    return {k: masked_batches(v, tok, batch_size, mask_prob, seed=1234) for k, v in sets.items()}


def load_teacher():
    """The frozen bf16 teacher on the GPU."""
    return distill.build_teacher(SimpleNamespace(teacher=distill.TEACHER, device="cuda"))


@torch.no_grad()
def score(model, teacher, bias, batches, depth):
    """-> {ce, kl (T=1), kl2 (T=2), loss (the training objective 0.6 CE + 0.4 T^2 KL_T2), acc}: token-weighted means over all masked positions."""
    ce = kl = kl2 = hits = n = 0.0
    for batch in batches:
        batch = {k: v.cuda() for k, v in batch.items()}
        count = int((batch["labels"] != -100).sum())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = teacher_hidden(teacher, batch)
            out = model(**batch, recurrence_depth=depth)
            kl += full_kl(out.logits, z, model.decoder.weight, bias, 1.0).item() * count
            kl2 += full_kl(out.logits, z, model.decoder.weight, bias, 2.0).item() * count
        ce += out.loss.item() * count
        hits += (out.logits.argmax(-1) == batch["labels"][batch["labels"] != -100]).sum().item()
        n += count
    return {"ce": ce / n, "kl": kl / n, "kl2": kl2 / n, "loss": distill.combine(ce / n, kl2 / n, OBJECTIVE), "acc": hits / n}


@torch.no_grad()
def teacher_ce(teacher, batches):
    total = n = 0.0
    for batch in batches:
        batch = {k: v.cuda() for k, v in batch.items()}
        count = int((batch["labels"] != -100).sum())
        total += teacher(**batch).loss.item() * count
        n += count
    return total / n


def load_model(spec):
    name, _, rest = spec.partition("=")
    path, _, kind = rest.partition(":")
    cls = FlatForMaskedLM if kind == "flat" else BortForMaskedLM
    model = cls.from_pretrained(path).cuda().eval()
    return name, model, (1,) if kind == "flat" else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="NAME=checkpoint_dir[:flat]")
    p.add_argument("--dataset", default="ccdv/arxiv-summarization")
    p.add_argument("--text_column", default="article")
    p.add_argument("--split", default="validation")
    p.add_argument("--samples", type=int, default=512, help="packed sequences per dataset")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--mask_prob", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--depths", type=int, nargs="+", default=list(range(1, 9)))
    p.add_argument("--out", default="runs/bench_heldout.json")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    distill.configure_cuda()

    batches = heldout_batches(a.samples, a.seq_len, a.batch_size, a.mask_prob, (a.dataset, a.text_column, a.split))
    for k, v in batches.items():
        print(f"[data] {k}: {sum(len(b['labels']) for b in v)} sequences x {a.seq_len} tokens, {sum(int((b['labels'] != -100).sum()) for b in v)} masked positions")

    teacher = load_teacher()
    bias = teacher.decoder.bias.float()
    results = {k: {"teacher_ce": teacher_ce(teacher, b)} for k, b in batches.items()}
    for k in results:
        print(f"[{k}] teacher CE {results[k]['teacher_ce']:.4f}")

    for spec in a.models:
        name, model, fixed = load_model(spec)
        for k, b in batches.items():
            results[k][name] = {d: score(model, teacher, bias, b, d) for d in (fixed or a.depths)}
            for d, r in results[k][name].items():
                print(f"[{k}] {name} depth {d}: CE {r['ce']:.4f}  KL {r['kl']:.4f}  KL(T=2) {r['kl2']:.4f}  train-loss {r['loss']:.4f}  acc {r['acc']:.4f}")
        del model
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[done] wrote {a.out}")
    return results


if __name__ == "__main__":
    main()

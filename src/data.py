"""Data: FineWeb (or any HF text dataset), streamed or not, packed into fixed-length MLM sequences."""
import zlib
from functools import partial
from itertools import islice

from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling


def is_holdout(doc_id, pct):
    """Deterministic document-level hold-out: ~pct% of documents (by CRC32 of the id) never reach training."""
    return zlib.crc32(str(doc_id).encode()) % 100 < pct


def _pack(batch, tokenizer, body):
    ids = [i for row in tokenizer(batch["text"], add_special_tokens=False)["input_ids"] for i in row]
    chunks = [ids[i:i + body] for i in range(0, len(ids) - body + 1, body)]
    return {"input_ids": [[tokenizer.cls_token_id, *c, tokenizer.sep_token_id] for c in chunks]}


def pack_dataset(ds, tokenizer, seq_len, columns=None):
    """Standard MLM packing: concatenate documents, cut into [CLS] + (seq_len-2) tokens + [SEP]. No padding.
    Works on map-style and streaming datasets. Remainders (<seq_len tokens per 1000 docs) are dropped.
    Module-level functions only (no lambdas/closures): the dataset must be picklable for spawned DataLoader workers."""
    columns = columns or ds.column_names or list(next(iter(ds)))
    return ds.map(partial(_pack, tokenizer=tokenizer, body=seq_len - 2), batched=True, remove_columns=columns)


def _is_train_row(row, id_column, pct):
    return not is_holdout(row[id_column], pct)


def _is_holdout_row(row, id_column, pct):
    return is_holdout(row[id_column], pct)


def _load(args, split):
    return load_dataset(args.dataset, args.dataset_config, split=split, streaming=args.streaming,
                        revision=args.dataset_revision or None)


def make_train_dataset(tokenizer, args):
    """Packed training sequences. Streaming: document shuffle buffer -> pack -> sequence shuffle buffer."""
    ds = _load(args, args.train_split)
    if args.holdout_pct:
        ds = ds.filter(partial(_is_train_row, id_column=args.id_column, pct=args.holdout_pct))
    if args.streaming:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    ds = pack_dataset(ds, tokenizer, args.seq_len)
    return ds.shuffle(seed=args.seed + 1, buffer_size=args.shuffle_buffer) if args.streaming else ds


def make_eval_dataset(tokenizer, args, n=None):
    """Held-out packed sequences, materialised once so every evaluation sees the same data."""
    n = n or args.eval_samples
    if args.holdout_pct:  # same corpus, disjoint documents
        ds = _load(args, args.train_split).filter(partial(_is_holdout_row, id_column=args.id_column, pct=args.holdout_pct))
        return Dataset.from_list(list(islice(pack_dataset(ds, tokenizer, args.seq_len), n)))
    ds = pack_dataset(_load(args, args.eval_split), tokenizer, args.seq_len)
    return ds.select(range(min(n, len(ds)))) if not args.streaming else Dataset.from_list(list(islice(ds, n)))


def make_collator(tokenizer, args, seed=None):
    """30% of non-special/non-pad tokens selected; of those 80% [MASK] / 10% random / 10% unchanged."""
    return DataCollatorForLanguageModeling(tokenizer, mlm_probability=args.mask_prob, mask_replace_prob=0.8,
                                           random_replace_prob=0.1, seed=seed)

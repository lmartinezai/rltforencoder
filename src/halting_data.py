"""One-time shared data for halting-head training: teacher targets stored on disk, so that no model ever needs the teacher again.

    uv run python src/halting_data.py --out stores/halting --train 4096 --eval 512

Writes  <out>/fineweb/train  FineWeb training sequences in the order of data seed 4242 with their teacher targets. It is a different stream from the ones the
                             models were distilled on (seed 42), but drawn from the same training documents: a model may have seen some of these texts,
        <out>/fineweb/eval   the FineWeb hold-out documents (never in any training stream),
        <out>/arxiv/eval     arXiv articles (a different distribution: the head must not just learn FineWeb). Its meta.json `data` block repeats the
                             FineWeb settings of the extraction arguments; the sequences are arXiv.
Every model's features are extracted from these same sequences and masks (src/halting_features.py --source store), so heads and their scores are comparable
across models and the teacher runs exactly once, here."""
import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

import distill
import pins
from bench_heldout import arxiv_sequences
from data import make_collator
from store import extract


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="stores/halting")
    p.add_argument("--train", type=int, default=4096, help="FineWeb training sequences (512 tokens each)")
    p.add_argument("--eval", type=int, default=512, help="held-out sequences per eval set")
    p.add_argument("--seed", type=int, default=4242)
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    out = Path(a.out)
    args = distill.DistillArguments(output_dir="runs/_halting_data", stage="extract", dataset_revision=pins.revision("HuggingFaceFW/fineweb") or "",
                                    targets_dir=str(out / "fineweb"), extract_tokens=a.train * 512,
                                    extract_batch=64, eval_samples=a.eval, seed=a.seed, report_to="none")
    distill.run_extract(args)  # fineweb/train and fineweb/eval, the teacher is freed on return
    tok = AutoTokenizer.from_pretrained(args.teacher, **pins.kw(args.teacher))
    teacher = distill.build_teacher(args)
    W, b = (t.to(args.device) for t in distill.teacher_decoder_params(args))
    meta = extract(teacher, W, b, arxiv_sequences(tok, a.eval, 512, "ccdv/arxiv-summarization", "article", "validation"), make_collator(tok, args, a.seed + 2),
                   out / "arxiv" / "eval", args, a.eval, 64)
    print(f"[halting_data] arxiv eval: {meta['n_seq']} sequences, {meta['masked']} masked positions")


if __name__ == "__main__":
    main()

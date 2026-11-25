# A gated state merge for looped transformer encoders

Code, results, and the paper (`paper/main.tex`, `paper/main.pdf`) for a looped encoder built from ModernBERT-large layers and distilled from
ModernBERT-large: a prelude applied once, a core applied R times, and a coda applied once, with the gated merge of Zhang et al. moved from the
token axis to the depth axis. The model is `src/arch.py` (named Bort in the code); the training entry points are `src/distill.py` (looped models)
and `src/compare_arch.py` (flat models).

No checkpoints are released. Every trained model can be rebuilt with the reproduction script, which uses the commands the reported runs were
started with. Its training commands are checked field by field against the arguments each of the 18 trained models stored with itself
(`reproduce/AUDIT.md`). Without a second machine, `--single-machine` records the teacher stream to disk and replays it to the trainers, so one
GPU suffices; the replayed stream has the same fingerprint as the one the reported runs read.

## Layout

| path | contents |
|---|---|
| `src/` | the model, training, evaluation, hardware measurement, and report scripts |
| `reproduce/reproduce.py` | every step behind the paper's numbers, in order: data, training, evaluation, reports, figures, number check |
| `reproduce/audit.py`, `reproduce/AUDIT.md` | the plan checked field by field against the recorded runs, and the result |
| `reproduce/as_run/` | the launchers the reported runs were started with, as run (private addresses redacted) |
| `results/` | every result file the paper reads; `results/run_records/` holds each trained model's recorded arguments, configuration, final step, and weight hash |
| `paper/` | the paper, its figure script (`figures.py`), the number check (`check_numbers.py`), and the provenance table (`MANIFEST.md`) |
| `tests/` | `uv run pytest -m "not gpu"` (CPU); the `gpu` tests need CUDA and the real teacher |
| `paper/make_arxiv.py` | the arXiv upload (`paper/arxiv.tar.gz`), compiled as arXiv does: pdflatex and BibTeX from TeX Live 2025 in Docker |
| `tools/build_bib.py` | builds `references.bib` from the arXiv API and checks every ID against its title |

## Reproducing the paper

    uv sync
    uv run python reproduce/reproduce.py --list          # every step and its recorded quirks (second machine, micro-batch split, shared teacher pass)
    uv run python reproduce/reproduce.py --dry-run       # the exact commands
    uv run python reproduce/reproduce.py --stage store train-4 eval halting report
    uv run python src/teacher_server.py --port 5555      # on a machine with ~16 GB of GPU memory, for the eight-block runs
    uv run python reproduce/reproduce.py --stage train-8 --teacher-host <that machine>
    uv run python reproduce/audit.py                     # re-check the plan against results/run_records

The report stage regenerates the tables and figures from the result files in seconds on a CPU and ends with `paper/check_numbers.py`, which
recomputes the paper's numbers from the result files and checks that `paper/main.tex` states them. The four-block study needs one 12 GB GPU
(26 to 28 minutes per run on an RTX 5070, 15 runs); each eight-block run took 3.2 hours on an RTX 5070 with the teacher on a second GPU.

## Exact revisions

Library versions are pinned in `uv.lock` (torch 2.11.0, transformers 5.22.2, openvino 2026.4.1). The stored teacher targets did not pin a dataset
revision; each repository's newest Hub commit predates every run (checked 2026-11-16), so these are the revisions every result used. The
reproduction script does not pin them either, as the runs did not: a later reproduction reads whatever the Hub then serves, so a dataset
changed after 2026-11-16 changes the data.

| repository | revision | commit date |
|---|---|---|
| HuggingFaceFW/fineweb (sample-10BT; training data, FineWeb hold-out) | `9bb295ddab0e05d785b879661af7260fed5140fc` | 2025-07-11 |
| answerdotai/ModernBERT-large (teacher, embeddings, layer init) | `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` | 2025-01-15 |
| ccdv/arxiv-summarization (held-out arXiv) | `240aaf1a969b3f8cd0ade6986bfad0cd730ee288` | 2024-08-08 |
| wikimedia/wikipedia (20231101.en) | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | 2024-01-09 |
| codeparrot/codeparrot-clean-valid | `4db92d2ec0c1b4c41eeb439cfae16854511d9dcd` | 2022-10-10 |
| nyu-mll/glue (dev splits only) | `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` | 2024-01-30 |

The three eight-block runs read the same teacher stream and were evaluated on the same eval set: A and B record identical `stream_sha256`
(`a198763c…`) and `eval_sha256` (`825edcdb…`) after 390,656 sequences (`results/run_records/A.json`, `B.json`); C's record stayed on the
second machine.

## The teacher server

The eight-block runs do not hold the teacher: `src/teacher_server.py` streams FineWeb, tokenizes, packs, and masks it, runs the teacher in
batches of 64 sequences, and sends the teacher's final hidden state at the masked positions (the student's frozen, tied output layer turns it
into the exact teacher distribution at any temperature). The stream is a pure function of (data seed, position): batch *i* always holds the same
sequences, masks, and teacher outputs however the run is timed, buffered, interrupted, or reconnected, and a checkpoint records its position
(`stream_position.json`) so that a resumed run continues bit for bit. Several trainers on the same seed share one teacher pass.
`src/teacher_supervisor.py` restarts a server that exits or stops answering; the trainers reconnect at the batch they were at.

## Notes on the code

- `--init teacher_layers` (every reported run) starts each block from the nearest teacher layer of the same attention type; the merge's
  output projection starts at zero, so the models with and without the merge begin as the same function.
- The four-block store (`stores/fineweb_s42_50m`) is written by `src/distill.py --stage extract`; training from it (`--targets_dir`) needs no
  teacher. A store records its teacher, tokenizer, and decoder fingerprints and its data settings, and is refused if they do not match.
- Halting was studied twice: `src/halting.py`, `src/halting_eval.py`, `src/freeze.py`, and `src/flops_score.py` are the first attempt, whose
  compute is counted in the project total; the reported heads come from `src/halting_data.py`, `src/halting_features.py`, `src/halting_fit.py`,
  and `src/halting_train.py` (`results/halting_v2`).
- During the latency sweeps, `src/hw_power.py` (a power logger whose output the paper does not use) ran in the background; the reproduction
  script runs it too, so the machine's load matches.

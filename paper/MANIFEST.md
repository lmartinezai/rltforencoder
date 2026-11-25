# Manifest: paper item -> result file -> reproduction step

`uv run python paper/check_numbers.py` recomputes the paper's tables and most prose numbers from the result files below (each table row bound to its
label, the code listing checked against `src/arch.py`) and checks that `main.tex` states them. It does not check every number in the paper, and
several checks read derived CSV or JSON files rather than raw scores.

Every step named below is a step of `reproduce/reproduce.py` (`--list` prints them with their recorded quirks); `reproduce/AUDIT.md` shows the
training steps checked against the arguments each run stored with itself (`results/run_records/`). No checkpoints are released; each run's
weight hash is in its record. A reproduction writes its evaluation outputs to `reproduced/`, never to `results/`, and ends with
`RESULTS_DIR=reproduced uv run python paper/check_numbers.py`.

| paper item | result file(s) | reproduction step(s) |
|---|---|---|
| Table 1 (models: parameters, weight storage, GFLOP) | `results/hardware/static.csv` | `eval:static-cost` |
| Table 2, Figure 2 (four-block study) | `results/small_study/summary.json`, `curves.csv`; `results/mlm_crossmodel/bpb.csv` | `store`, `train-4:*`, `eval:pm-bpb`, `report:four-block`, `report:tables`, `report:figures` |
| Table 3, Figure 3 (eight-block PM-BPB) | `results/mlm_crossmodel/bpb.csv`, `raw.json` | `train-8:*`, `eval:pm-bpb`, `report:tables`, `report:figures` |
| Eight-block held-out distillation loss | `results/arms_200m/summary.md`, `bench_heldout.json` | `eval:heldout-loss-8`, `report:eight-block` |
| Training compute | `results/compute_ledger.json` (`measured_gflop_per_sequence`) | `eval:compute` |
| Table 4 (latency at R = 4, every backend) | `results/hardware/depth4_pass/latency.jsonl`, `latency_e.jsonl`; `summary.md` (its R = 4 section) | `hardware:r4-*`, `report:hardware` |
| Earlier latency sweep (other depths; the session agreement stated with Table 4) | `results/hardware/latency.jsonl` | `hardware:latency*` |
| Latency repeatability | `results/hardware/repeatability/` | re-runs of `src/hw_bench.py`, described in its `summary.md` (not in the reproduction script) |
| Weight quantization | `results/hardware/quant.json` | `eval:quantization` |
| Halting (reported heads) | `results/halting_v2/*.json`, `summary.md` | `halting:store`, `halting:heads` |
| Halting (earlier exploration on the eight-block merged model) | `results/halting_v2/target_variants/` | `halting:explore-*` |
| Zero-shot NLI | `results/zeroshot_nli/accuracy.csv`, `raw.json`, `summary.md` | `eval:nli`, `report:nli` |

## Cost-accounting convention

Training compute in the paper is counted operations: `torch.utils.flop_counter` over one distillation step (matrix multiplications and
attention, forward and backward, including the decoding of the stored teacher state), counted at depths 1 and 2 and extended linearly to depths
3 to 8, averaged over depths 1 to 8, times the sequences of a run. The looped profile was counted with the merge and is used for the no-merge
models too (it overstates them by the merge's share). The teacher forward is assigned to each eight-block run, although A and C shared one
teacher pass. The ledger total adds a 2% allowance for evaluation during training and approximate counts for the earliest runs, and excludes the
later evaluations.

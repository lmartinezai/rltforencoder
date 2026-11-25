#!/usr/bin/env bash
# D' seed 2 (so every small-study arm has seeds 1 2 3), on the 5070 with the unchanged launcher, then the same evaluations as the other arms:
# held-out bits per byte (results/mlm_crossmodel, appended to raw.json) and the regenerated reports.
cd <repo>
export PYTHONUNBUFFERED=1
bash experiments/iso_flop/launch_5070.sh 2 || { echo TRAIN_FAILED; exit 1; }
uv run python nb/crossmodel_eval.py --models d_s2=flat:runs/iso_flop/d_s2/checkpoint-3072 || { echo EVAL_FAILED; exit 1; }
uv run python nb/mlm_report.py > /dev/null && uv run python nb/small_report.py --runs runs/small runs/small_c runs/iso_flop --out results/small_study
echo D2_DONE

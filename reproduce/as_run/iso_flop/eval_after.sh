#!/usr/bin/env bash
# After the hardware sweep: held-out bits per byte of the iso-FLOP control (3 seeds), then the weight-quantisation robustness evaluation.
cd <repo>
export PYTHONUNBUFFERED=1
S=$(date +%s)
uv run python nb/crossmodel_eval.py --models d_s1=flat:runs/iso_flop/d_s1/checkpoint-3072 d_s3=flat:runs/iso_flop/d_s3/checkpoint-3072
echo "BPB_SECONDS $(( $(date +%s) - S ))"
uv run python nb/quant_eval.py --models c_s1 d_s1 a_s1 b_s1 C8 A8 B8 roberta-base teacher
echo EVAL_DONE

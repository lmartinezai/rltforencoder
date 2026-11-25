#!/usr/bin/env bash
# 200M-token comparison, arms A then B on the 5070 (one after the other: no overlapping GPU jobs; B starts only if A ended cleanly).
#   A  looped + gated merge     B  looped, no gated merge
# Arm C (no loop) runs at the same time as A on <lan-host>: runs/launch_200m_c.bat.
# 12208 steps x 32 sequences x 512 tokens = 200.0M tokens. Same seed (42), data, masks, teacher outputs, eval set and schedule in every arm.
cd <repo>
COMMON="--teacher_addr <teacher-host>:5555 --compile_layers --stream_buffer 2048 --init teacher_layers --max_steps 12208 --warmup_steps 100 --save_steps 1000 --eval_steps 500 --save_total_limit 2 --disable_tqdm true --report_to none"
export PYTHONUNBUFFERED=1
uv run python nb/distill.py --output_dir runs/arm_a_merge --gated_merge true $COMMON > runs/arm_a_merge.log 2>&1 \
  && uv run python nb/distill.py --output_dir runs/arm_b_nomerge --gated_merge false $COMMON > runs/arm_b_nomerge.log 2>&1

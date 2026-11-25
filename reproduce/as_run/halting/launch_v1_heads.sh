#!/usr/bin/env bash
# Halting heads for the two looped arms: token level and sequence level each, one after the other on the 5070 (no overlapping GPU jobs;
# each starts only if the previous ended cleanly). Backbone frozen; only the 1025-parameter halting head trains. Fresh data (--data_seed 4242),
# teacher on <lan-host>. Then: uv run python nb/halting_eval.py --models A=...,runs/halt_a_token/halting_head.pt,runs/halt_a_seq/halting_head.pt ...
cd <repo>
COMMON="--teacher_addr <teacher-host>:5555 --data_seed 4242 --stream_buffer 2048 --max_steps 300 --eval_steps 50 --disable_tqdm true --report_to none"
export PYTHONUNBUFFERED=1
run() {  # name checkpoint level
  uv run python nb/halting.py --student "$2" --output_dir "runs/halt_$1" --halt_level "$3" $COMMON > "runs/halt_$1.log" 2>&1
}
run a_token runs/arm_a_merge/checkpoint-12208 token \
  && run a_seq runs/arm_a_merge/checkpoint-12208 sequence \
  && run b_token runs/arm_b_nomerge/checkpoint-12208 token \
  && run b_seq runs/arm_b_nomerge/checkpoint-12208 sequence

#!/usr/bin/env bash
# Iso-FLOP flat control for the small study. Small A' (1 + 2 + 1 blocks) at its best depth 4 runs 1 + 2x4 + 1 = 10 block passes, so the control is a plain
# ModernBERT of 10 distinct blocks (full attention every 3rd, as in the teacher), no loop, initialised from teacher layers like every other arm. Same stored
# targets, schedule and seeds as the small study. Usage: launch_5070.sh SEED...   (runs one after the other on this GPU; finished runs are skipped)
cd <repo>
COMMON="--targets_dir stores/fineweb_s42_50m --compile_layers --init teacher_layers --prelude_layers 1 --looped_layers 8 --coda_layers 1 --flat_layer_types modernbert --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --max_steps 3072 --warmup_steps 100 --eval_steps 512 --save_steps 1024 --save_total_limit 1 --disable_tqdm true --report_to none"
export PYTHONUNBUFFERED=1
mkdir -p runs/iso_flop
for seed in "$@"; do
  n="d_s$seed"
  [ -f "runs/iso_flop/$n/DONE" ] && continue
  rm -rf "runs/iso_flop/$n"
  uv run python nb/compare_arch.py --output_dir "runs/iso_flop/$n" --seed "$seed" $COMMON > "runs/iso_flop/$n.log" 2>&1 && touch "runs/iso_flop/$n/DONE" || exit 1
done

#!/usr/bin/env bash
# Small-model study, looped arms on the 5070: A' = looped + gated merge, B' = looped without merge; Bort 1 + 2 + 1 = 4 blocks (~50M trainable
# parameters), three seeds each. Every run reads the same stored teacher targets (stores/fineweb_s42_50m: 98,304 sequences = 50.3M tokens,
# one epoch = 3072 steps of 32 sequences), same schedule, same init recipe (--init teacher_layers); only arm and --seed differ (seed = depth schedule and
# data order; with --init teacher_layers the only randomly initialised weights are arm A's merge gate, and the flat arm has none). Runs one after the other (no overlapping GPU jobs), seeds interleaved, finished runs (DONE) are skipped on a re-launch.
# The flat arm C' runs on the 3060 Ti at the same time: runs/launch_small_c.bat.
cd <repo>
COMMON="--targets_dir stores/fineweb_s42_50m --compile_layers --init teacher_layers --prelude_layers 1 --looped_layers 2 --coda_layers 1 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --max_steps 3072 --warmup_steps 100 --eval_steps 256 --save_steps 1024 --save_total_limit 1 --eval_depths 1 2 4 8 --disable_tqdm true --report_to none"
export PYTHONUNBUFFERED=1
run() {  # name merge seed
  [ -f "runs/small/$1/DONE" ] && return 0
  rm -rf "runs/small/$1"
  uv run python nb/distill.py --output_dir "runs/small/$1" --gated_merge "$2" --seed "$3" $COMMON > "runs/small/$1.log" 2>&1 && touch "runs/small/$1/DONE"
}
for seed in 1 2 3; do
  run "a_s$seed" true "$seed" || exit 1
  run "b_s$seed" false "$seed" || exit 1
done

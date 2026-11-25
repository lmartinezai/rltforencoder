#!/usr/bin/env bash
# Arm E' of the small-model study: looped + gated merge in RLT's form (--merge_base anchor: every iteration starts from the prelude output and the
# loop state arrives as the gated feedback, u = e + alpha g W RMSNorm(s)), i.e. with input injection. Everything else exactly as A' (experiments/
# small_study/launch_5070.sh): same stored teacher targets, Bort 1 + 2 + 1, schedule, teacher-layer init, seeds 1 2 3. Finished runs (DONE) are
# skipped on a re-launch. Then the same evaluations as the other arms: held-out bits per byte (results/mlm_crossmodel) and the reports.
cd <repo>
COMMON="--targets_dir stores/fineweb_s42_50m --compile_layers --init teacher_layers --prelude_layers 1 --looped_layers 2 --coda_layers 1 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --max_steps 3072 --warmup_steps 100 --eval_steps 256 --save_steps 1024 --save_total_limit 1 --eval_depths 1 2 4 8 --disable_tqdm true --report_to none"
export PYTHONUNBUFFERED=1
for seed in 1 2 3; do
  n="e_s$seed"
  [ -f "runs/small/$n/DONE" ] && continue
  rm -rf "runs/small/$n"
  uv run python nb/distill.py --output_dir "runs/small/$n" --gated_merge true --merge_base anchor --seed "$seed" $COMMON > "runs/small/$n.log" 2>&1 \
    && touch "runs/small/$n/DONE" || { echo "TRAIN_FAILED $n"; exit 1; }
done
uv run python nb/crossmodel_eval.py --models e_s1=bort:runs/small/e_s1/checkpoint-3072 e_s2=bort:runs/small/e_s2/checkpoint-3072 e_s3=bort:runs/small/e_s3/checkpoint-3072 \
  || { echo EVAL_FAILED; exit 1; }
uv run python nb/mlm_report.py > /dev/null && uv run python nb/small_report.py --runs runs/small runs/small_c runs/iso_flop --out results/small_study
echo E_DONE

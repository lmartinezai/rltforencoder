@echo off
rem Iso-FLOP flat control, seed 3, on the 3060 Ti (see launch_5070.sh): same settings, batch 4 x 8 accumulation.
cd /d <repo>
set PYTHONUNBUFFERED=1
if not exist runs\iso_flop mkdir runs\iso_flop
if not exist runs\iso_flop\d_s3\DONE (
  if exist runs\iso_flop\d_s3 rmdir /s /q runs\iso_flop\d_s3
  python -m uv run python nb\compare_arch.py --output_dir runs\iso_flop\d_s3 --seed 3 --targets_dir stores\fineweb_s42_50m --compile_layers --init teacher_layers --prelude_layers 1 --looped_layers 8 --coda_layers 1 --flat_layer_types modernbert --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --max_steps 3072 --warmup_steps 100 --eval_steps 512 --save_steps 1024 --save_total_limit 1 --disable_tqdm true --report_to none > runs\iso_flop\d_s3.log 2>&1
  if not errorlevel 1 echo done > runs\iso_flop\d_s3\DONE
)

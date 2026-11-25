@echo off
rem Arm E' (input injection, RLT form), seed 3, on the 3060 Ti, in parallel with seeds 1-2 on the 5070 (launch_5070.sh). Same settings as the 5070
rem runs except the micro-batch split: 4 x 8 accumulation here (8 GB card), 8 x 4 there; the effective batch is 32 sequences in both, the depth
rem schedule and data order are the same per optimizer step, and the stored teacher targets fix the masks.
cd /d <repo>
set PYTHONUNBUFFERED=1
if not exist runs\small\e_s3\DONE (
  if exist runs\small\e_s3 rmdir /s /q runs\small\e_s3
  python -m uv run python nb\distill.py --output_dir runs\small\e_s3 --gated_merge true --merge_base anchor --seed 3 --targets_dir stores\fineweb_s42_50m --compile_layers --init teacher_layers --prelude_layers 1 --looped_layers 2 --coda_layers 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --max_steps 3072 --warmup_steps 100 --eval_steps 256 --save_steps 1024 --save_total_limit 1 --eval_depths 1 2 4 8 --disable_tqdm true --report_to none > runs\small\e_s3.log 2>&1
  if not errorlevel 1 echo done > runs\small\e_s3\DONE
)
echo EXIT %ERRORLEVEL% >> runs\small\e_s3.log

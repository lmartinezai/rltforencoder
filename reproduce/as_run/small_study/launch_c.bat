@echo off
rem Small-model study, flat arm C' (4 plain ModernBERT blocks, no loop) on the 3060 Ti, three seeds, same stored targets and settings as runs/launch_small_5070.sh.
cd /d <repo>
set PYTHONUNBUFFERED=1
for %%S in (1 2 3) do (
  if not exist runs\small\c_s%%S\DONE (
    if exist runs\small\c_s%%S rmdir /s /q runs\small\c_s%%S
    python -m uv run python nb\compare_arch.py --output_dir runs\small\c_s%%S --seed %%S --targets_dir stores\fineweb_s42_50m --compile_layers --init teacher_layers --prelude_layers 1 --looped_layers 2 --coda_layers 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --max_steps 3072 --warmup_steps 100 --eval_steps 256 --save_steps 1024 --save_total_limit 1 --disable_tqdm true --report_to none > runs\small\c_s%%S.log 2>&1
    if errorlevel 1 exit /b 1
    echo done > runs\small\c_s%%S\DONE
  )
)
echo ALLDONE >> runs\small\ALLDONE.txt

@echo off
rem Arm C of the 200M-token comparison (no loop, the flat 8-layer baseline) on the 3060 Ti. Started together with arm A on the 5070.
cd /d <repo>
set PYTHONUNBUFFERED=1
python -m uv run python nb\compare_arch.py --output_dir runs\arm_c_noloop --teacher_addr <teacher-host>:5555 --compile_layers --stream_buffer 2048 --init teacher_layers --max_steps 12208 --warmup_steps 100 --save_steps 1000 --eval_steps 500 --save_total_limit 2 --disable_tqdm true --report_to none >> arm_c_noloop.log 2>&1
echo EXIT %ERRORLEVEL% >> arm_c_noloop.log

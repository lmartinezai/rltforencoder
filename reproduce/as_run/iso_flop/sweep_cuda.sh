#!/usr/bin/env bash
# Re-run of the RTX 5070 part of the sweep (the first attempt hit a bf16 -> numpy bug in the output check), with the power logger on.
cd <repo>
export PYTHONUNBUFFERED=1
rm -f results/hardware/stop
uv run python nb/hw_power.py --out results/hardware > results/hardware/power2.log 2>&1 &
sleep 20
uv run python nb/hw_bench.py --backend torch --devices cuda --retry_errors
touch results/hardware/stop
wait
echo CUDA_DONE

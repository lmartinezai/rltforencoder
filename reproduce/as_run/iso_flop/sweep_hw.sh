#!/usr/bin/env bash
# The hardware sweep, alone on the machine: power logger, 25 s idle, then OpenVINO on NPU / iGPU / CPU, the RTX 5070 (PyTorch), and PyTorch on the CPU (batch 1).
# Finished configurations are skipped on a re-run, failed ones retried (results/hardware/latency.jsonl).
cd <repo>
export PYTHONUNBUFFERED=1
uv run python nb/hw_power.py --out results/hardware > results/hardware/power.log 2>&1 &
sleep 25
uv run python nb/hw_bench.py --backend openvino --devices NPU GPU.0 CPU --retry_errors
uv run python nb/hw_bench.py --backend torch --devices cuda --retry_errors
uv run python nb/hw_bench.py --backend torch --devices cpu --batches 1 --retry_errors
touch results/hardware/stop
wait
echo SWEEP_DONE

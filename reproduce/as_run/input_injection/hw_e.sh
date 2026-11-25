#!/usr/bin/env bash
# Arm E' (seed 1, as every small arm in the hardware study) through the same hardware measurements, alone on the machine:
# (1) latency: OpenVINO NPU / iGPU / CPU, the RTX 5070 (PyTorch), PyTorch CPU at batch 1 (appended to results/hardware/latency.jsonl; finished rows skipped);
# (2) the energy pass (512 tokens, batch 1, depth 4, every device) with its own power log and idle level in results/hardware/energy_e, because the power
# logger rewrites its files and the first energy pass (results/hardware/energy) must stay as measured; (3) weight quantisation; (4) static cost; report.
# First attempt (2026-11-17) ran while a game used the GPU and was set aside (runs/hw_e_contended): close everything else before running this.
cd <repo>
export PYTHONUNBUFFERED=1
S=$(date +%s)
uv run python nb/hw_bench.py --backend openvino --devices NPU GPU.0 CPU --models e_s1
uv run python nb/hw_bench.py --backend torch --devices cuda --models e_s1
uv run python nb/hw_bench.py --backend torch --devices cpu --batches 1 --models e_s1
echo "LATENCY_SECONDS $(( $(date +%s) - S ))"
mkdir -p results/hardware/energy_e; rm -f results/hardware/energy_e/stop
uv run python nb/hw_power.py --out results/hardware/energy_e > results/hardware/energy_e/power.log 2>&1 &
sleep 25
E="--seqs 512 --batches 1 --depths 4 --models e_s1 --out results/hardware/energy_e"
uv run python nb/hw_bench.py --backend openvino --devices GPU.0 CPU $E
uv run python nb/hw_bench.py --backend torch --devices cuda $E
uv run python nb/hw_bench.py --backend torch --devices cpu $E
uv run python nb/hw_bench.py --backend openvino --devices NPU $E
touch results/hardware/energy_e/stop
wait
echo "ENERGY_SECONDS $(( $(date +%s) - S ))"
uv run python nb/quant_eval.py --models e_s1
uv run python nb/model_cost.py --out results/hardware
uv run python nb/hw_report.py
echo "HW_E_DONE $(( $(date +%s) - S ))"

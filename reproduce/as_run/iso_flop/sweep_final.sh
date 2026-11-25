#!/usr/bin/env bash
# Last parts of the hardware study, alone on the machine. (1) The NPU re-run for the models whose first attempts failed after the device hung on an earlier
# configuration. (2) The dedicated energy pass: 512 tokens, batch 1, looped models at depth 4, every device, one clean power log (results/hardware/energy).
# The NPU goes last in each: DeBERTa at 512 tokens hangs it (excluded).
cd <repo>
export PYTHONUNBUFFERED=1
NPU_MODELS="c_s1 d_s1 a_s1 b_s1 C8 A8 B8 distilbert-cased teacher roberta-base"
mkdir -p results/hardware/energy; rm -f results/hardware/energy/stop
uv run python nb/hw_power.py --out results/hardware/energy > results/hardware/energy/power.log 2>&1 &
sleep 25
E="--seqs 512 --batches 1 --depths 4 --out results/hardware/energy"
uv run python nb/hw_bench.py --backend openvino --devices GPU.0 CPU $E
uv run python nb/hw_bench.py --backend torch --devices cuda $E
uv run python nb/hw_bench.py --backend torch --devices cpu $E
uv run python nb/hw_bench.py --backend openvino --devices NPU --models $NPU_MODELS $E
touch results/hardware/energy/stop
wait
echo FINAL_DONE

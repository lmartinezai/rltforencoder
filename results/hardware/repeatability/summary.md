# Repeatability of the hardware numbers (2026-11-17, machine idle, after E' was added)

A' (`a_s1`) was re-timed in the session that measured E' (`e_s1`), and A' and E' were timed side by side, to separate an architecture effect from
session-to-session drift. 512 tokens, batch 1, p50 ms.

| measurement | original sweep (A') | this session (A') | this session (E') |
|---|---:|---:|---:|
| PyTorch CPU, depth 4 | 155.8 | 159.2 (before E' pass), 155.4 (after), 199.1 (side by side) | 197.4*, 160.0 (side by side) |
| PyTorch CPU, depth 8 | 262.1 | 284.5 (before), 277.5 (after), 283.6 (side by side) | 279.4*, 277.5 (side by side) |
| OpenVINO NPU, depth 8 | 229.9 | 225.9 | 228.6 |
| OpenVINO iGPU, depth 8 | 170.1 | 193.2 | 186.5 |
| OpenVINO CPU, depth 8 | 336.1 | 386.9 | 342.7 |

`*` from the E' pass in results/hardware (latency.jsonl), timed ~10 minutes earlier; every other E' cell is from a side-by-side run.

Reading: timed side by side, E' costs what A' costs (as its FLOPs say: identical per sequence). Latency drifts by up to ~15% between sessions on the
OpenVINO CPU and iGPU paths and ~6-8% on PyTorch CPU, where depth 4 is moreover bimodal (~155 or ~199 ms, for either model: the E' pass's 197.4 is
the slow mode, not an E' effect); NPU and RTX 5070 latencies repeat within ~2%. Files: openvino_latency/, torch_cpu_before.jsonl,
torch_cpu_after.jsonl, torch_cpu_side_by_side/.

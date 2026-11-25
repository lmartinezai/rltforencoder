# Hardware study

## Parameters and FLOPs (static)

| model | body M | embeddings M | body MB fp16 | body MB int8 | encoder GFLOP, 128 tok | encoder GFLOP, 512 tok |
|---|---:|---:|---:|---:|---:|---:|
| A looped+merge (8 blocks) | 101.2 | 51.6 | 202 | 101 | 26 / 66 / 119 | 104 / 264 / 478 |
| B looped (8 blocks) | 98.1 | 51.6 | 196 | 98 | 25 / 63 / 113 | 100 / 251 / 452 |
| C flat (8 blocks) | 98.1 | 51.6 | 196 | 98 | 25 | 100 |
| A' looped+merge (4 blocks) | 52.2 | 51.6 | 104 | 52 | 13 / 35 / 63 | 53 / 138 / 252 |
| B' looped (4 blocks) | 49.0 | 51.6 | 98 | 49 | 13 / 31 / 56 | 50 / 125 / 226 |
| C' flat (4 blocks) | 49.0 | 51.6 | 98 | 49 | 13 | 50 |
| D' flat, 10 blocks (iso-FLOP) | 122.6 | 51.6 | 245 | 123 | 31 | 125 |
| E' looped+merge, input injection (4 blocks) | 52.2 | 51.6 | 104 | 52 | 13 / 35 / 63 | 53 / 138 / 252 |
| ModernBERT-large (teacher) | 343.2 | 51.6 | 686 | 343 | 88 | 351 |
| RoBERTa-base | 85.1 | 39.0 | 170 | 85 | 22 | 87 |
| DeBERTa-base | 99.2 | 39.4 | 198 | 99 | 31 | 145 |
| DistilBERT-cased | 42.5 | 22.7 | 85 | 43 | 11 | 43 |

(looped models: depths 1 / 4 / 8)

## Latency, 512 tokens, batch 1: p50 ms

| model | depth | Intel NPU (OpenVINO, fp16) | Intel iGPU (OpenVINO, fp16) | Core Ultra 7 265K CPU (OpenVINO, fp32) | RTX 5070 (PyTorch eager, bf16) | CPU (PyTorch, fp32) |
|---|---|---:|---:|---:|---:|---:|
| A looped+merge (8 blocks) | 1 | 73.3 | 90.5 | 142.3 | 6.2 | 116.8 |
| A looped+merge (8 blocks) | 4 | 192.5* | 176.7 | 334.9 | 14.1 | 278.6 |
| A looped+merge (8 blocks) | 8 | 351.5 | 300.0 | 619.3 | 24.8 | 487.2 |
| B looped (8 blocks) | 1 | - | - | - | 5.8 | 113.9 |
| B looped (8 blocks) | 4 | - | - | - | 13.4 | 274.2 |
| B looped (8 blocks) | 8 | 275.6 | 305.6 | 575.7 | 26.3 | 485.4 |
| C flat (8 blocks) | - | 71.9 | 94.1 | 148.8 | 5.8 | 119.2 |
| A' looped+merge (4 blocks) | 1 | 49.1 | 63.3 | 90.6 | 3.4 | 72.8 |
| A' looped+merge (4 blocks) | 4 | 126.6* | 109.1* | 195.8* | 7.8 | 155.8 |
| A' looped+merge (4 blocks) | 8 | 229.9 | 170.1 | 336.1 | 14.2 | 262.1 |
| B' looped (4 blocks) | 1 | - | - | - | 3.3 | 71.0 |
| B' looped (4 blocks) | 4 | - | - | - | 7.1 | 151.7 |
| B' looped (4 blocks) | 8 | 147.3 | 164.5 | 304.4 | 12.4 | 274.9 |
| C' flat (4 blocks) | - | 42.5 | 64.2 | 100.2 | 3.1 | 68.9 |
| D' flat, 10 blocks (iso-FLOP) | - | 83.6 | 109.0 | 175.6 | 6.8 | 144.3 |
| E' looped+merge, input injection (4 blocks) | 1 | 46.7 | 79.8 | 127.7 | 3.6 | 81.1 |
| E' looped+merge, input injection (4 blocks) | 4 | 124.1* | 130.1* | 259.8* | 7.9 | 197.4 |
| E' looped+merge, input injection (4 blocks) | 8 | 227.3 | 197.1 | 435.8 | 15.3 | 279.4 |
| ModernBERT-large (teacher) | - | 208.1 | 239.4 | 431.8 | 17.3 | 433.0 |
| RoBERTa-base | - | 71.9 | 77.7 | 124.0 | 6.5 | 103.6 |
| DeBERTa-base | - | - | 119.5 | 188.3 | 18.2 | 220.0 |
| DistilBERT-cased | - | 38.3 | 42.3 | 64.5 | 3.0 | 48.5 |

## Latency, 128 tokens, batch 1: p50 ms

| model | depth | Intel NPU (OpenVINO, fp16) | Intel iGPU (OpenVINO, fp16) | Core Ultra 7 265K CPU (OpenVINO, fp32) | RTX 5070 (PyTorch eager, bf16) | CPU (PyTorch, fp32) |
|---|---|---:|---:|---:|---:|---:|
| A looped+merge (8 blocks) | 1 | 17.3 | 40.8 | 38.3 | 5.9 | 37.0 |
| A looped+merge (8 blocks) | 4 | 43.6* | 62.2 | 86.0 | 14.3 | 101.2 |
| A looped+merge (8 blocks) | 8 | 78.8 | 92.9 | 170.5 | 25.5 | 155.2 |
| B looped (8 blocks) | 1 | - | - | - | 5.7 | 35.4 |
| B looped (8 blocks) | 4 | - | - | - | 13.4 | 84.3 |
| B looped (8 blocks) | 8 | 58.4 | 92.7 | 153.1 | 23.2 | 192.7 |
| C flat (8 blocks) | - | 17.2 | 41.6 | 41.5 | 5.5 | 36.3 |
| A' looped+merge (4 blocks) | 1 | 11.3 | 34.4 | 25.2 | 3.2 | 22.3 |
| A' looped+merge (4 blocks) | 4 | 30.3* | 46.7* | 54.4* | 7.8 | 50.1 |
| A' looped+merge (4 blocks) | 8 | 55.7 | 62.9 | 93.2 | 14.3 | 86.7 |
| B' looped (4 blocks) | 1 | - | - | - | 3.0 | 20.7 |
| B' looped (4 blocks) | 4 | - | - | - | 6.9 | 46.7 |
| B' looped (4 blocks) | 8 | 32.2 | 58.1 | 82.0 | 12.3 | 80.6 |
| C' flat (4 blocks) | - | 10.4 | 34.3 | 24.5 | 2.8 | 22.9 |
| D' flat, 10 blocks (iso-FLOP) | - | 20.3 | 45.5 | 48.1 | 6.6 | 45.2 |
| E' looped+merge, input injection (4 blocks) | 1 | 11.6 | 42.3 | 31.2 | 3.3 | 28.0 |
| E' looped+merge, input injection (4 blocks) | 4 | 29.3* | 50.5* | 73.3* | 8.1 | 50.1 |
| E' looped+merge, input injection (4 blocks) | 8 | 53.0 | 61.4 | 129.4 | 14.6 | 113.6 |
| ModernBERT-large (teacher) | - | 45.9 | 104.6 | 110.6 | 17.4 | 130.2 |
| RoBERTa-base | - | 16.6 | 32.7 | 32.9 | 5.9 | 34.5 |
| DeBERTa-base | - | 600.1 | 35.2 | 36.6 | 17.3 | 55.4 |
| DistilBERT-cased | - | 8.6 | 18.0 | 17.4 | 3.0 | 16.9 |

## Latency, 512 tokens, batch 8: sequences per second

| model | depth | Intel iGPU (OpenVINO, fp16) | Core Ultra 7 265K CPU (OpenVINO, fp32) |
|---|---|---:|---:|
| A looped+merge (8 blocks) | 1 | 11.7 | 7.2 |
| A looped+merge (8 blocks) | 4 | 5.7 | - |
| A looped+merge (8 blocks) | 8 | 3.4* | - |
| ModernBERT-large (teacher) | - | 4.7 | 2.4 |
| RoBERTa-base | - | 13.9 | 8.4 |
| DeBERTa-base | - | 8.7 | 5.4 |
| DistilBERT-cased | - | 26.5 | 16.2 |

`*` = fitted line through the timed depths.

## Latency at R = 4, every backend (the paper's Table 4): 512 tokens, batch 1, p50 ms

| model | depth | RTX 5070 (PyTorch eager, bf16) | Intel NPU (OpenVINO, fp16) | Intel iGPU (OpenVINO, fp16) | Core Ultra 7 265K CPU (OpenVINO, fp32) | CPU (PyTorch, fp32) |
|---|---|---:|---:|---:|---:|---:|
| A looped+merge (8 blocks) | 4 | 13.7 | 195.4 | 183.3 | 354.9 | 276.7 |
| B looped (8 blocks) | 4 | 12.8 | 166.8 | 184.0 | 346.8 | 262.6 |
| C flat (8 blocks) | - | 5.5 | 71.5 | 92.9 | 153.4 | 121.9 |
| A' looped+merge (4 blocks) | 4 | 7.7 | 127.2 | 109.7 | 198.6 | 159.7 |
| B' looped (4 blocks) | 4 | 6.9 | 87.6 | 105.5 | 186.4 | 136.7 |
| C' flat (4 blocks) | - | 3.0 | 42.6 | 62.6 | 83.1 | 63.9 |
| D' flat, 10 blocks (iso-FLOP) | - | 6.6 | 84.9 | 108.2 | 185.2 | 141.5 |
| E' looped+merge, input injection (4 blocks) | 4 | 7.9 | 125.8 | 108.5 | 193.1 | 167.8 |
| ModernBERT-large (teacher) | - | 16.7 | 205.4 | 245.8 | 464.9 | 370.3 |
| RoBERTa-base | - | 6.0 | 74.3 | 79.3 | 134.1 | 104.8 |
| DeBERTa-base | - | 17.1 | - | 118.5 | 183.1 | 204.9 |
| DistilBERT-cased | - | 2.9 | 39.5 | 42.8 | 67.8 | 49.6 |

## Weight quantisation (round-to-nearest, no calibration; mean BPB, change against unquantised)

| model | depth | bf16 | int8 | int4 (group 64) |
|---|---|---:|---:|---:|
| A looped+merge (8 blocks) | 1 | 0.7076 | 0.7077 (+0.0001) | 0.7357 (+0.0281) |
| A looped+merge (8 blocks) | 4 | 0.6134 | 0.6136 (+0.0002) | 0.6388 (+0.0253) |
| A looped+merge (8 blocks) | 8 | 0.6210 | 0.6212 (+0.0002) | 0.6500 (+0.0289) |
| B looped (8 blocks) | 1 | 0.7082 | 0.7084 (+0.0002) | 0.7400 (+0.0319) |
| B looped (8 blocks) | 4 | 0.6158 | 0.6160 (+0.0002) | 0.6412 (+0.0254) |
| B looped (8 blocks) | 8 | 0.6251 | 0.6252 (+0.0001) | 0.6543 (+0.0292) |
| C flat (8 blocks) | - | 0.6550 | 0.6552 (+0.0002) | 0.6797 (+0.0247) |
| A' looped+merge (4 blocks) | 1 | 1.0607 | 1.0618 (+0.0011) | 1.1059 (+0.0452) |
| A' looped+merge (4 blocks) | 4 | 0.9292 | 0.9300 (+0.0008) | 0.9741 (+0.0449) |
| A' looped+merge (4 blocks) | 8 | 0.9475 | 0.9481 (+0.0005) | 1.0072 (+0.0597) |
| B' looped (4 blocks) | 1 | 1.0651 | 1.0659 (+0.0008) | 1.1094 (+0.0443) |
| B' looped (4 blocks) | 4 | 0.9307 | 0.9313 (+0.0007) | 0.9765 (+0.0459) |
| B' looped (4 blocks) | 8 | 0.9504 | 0.9513 (+0.0008) | 1.0102 (+0.0598) |
| C' flat (4 blocks) | - | 0.9806 | 0.9807 (+0.0002) | 1.0185 (+0.0380) |
| D' flat, 10 blocks (iso-FLOP) | - | 0.6794 | 0.6796 (+0.0003) | 0.7112 (+0.0318) |
| E' looped+merge, input injection (4 blocks) | 1 | 1.0116 | 1.0119 (+0.0003) | 1.0362 (+0.0246) |
| E' looped+merge, input injection (4 blocks) | 4 | 0.9469 | 0.9471 (+0.0002) | 0.9721 (+0.0251) |
| E' looped+merge, input injection (4 blocks) | 8 | 0.9475 | 0.9478 (+0.0003) | 0.9729 (+0.0255) |
| ModernBERT-large (teacher) | - | 0.2781 | 0.2785 (+0.0004) | 0.3074 (+0.0292) |
| RoBERTa-base | - | 0.9148 | 0.9144 (-0.0004) | 1.0295 (+0.1147) |

## Configurations that failed

| backend/device | model | depth | seq | batch | error |
|---|---|---|---:|---:|---|
| openvino/NPU | deberta-base | None | 128 | 8 | RuntimeError: Exception from src\inference\src\cpp\core.cpp:120:
Exception from src\inference\src\de |
| openvino/NPU | deberta-base | None | 512 | 1 | RuntimeError: Exception from src\inference\src\cpp\infer_request.cpp:224:
Exception from src\plugins |

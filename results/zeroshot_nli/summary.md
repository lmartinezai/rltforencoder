**Zero-shot accuracy (%), calibrated label-word scores, mean of the two prompts; mean +- sample std over 3 seeds where there are 3.**

| model | depth | MNLI | QNLI | RTE | mean | collapse |
|---|---|---:|---:|---:|---:|---:|
| majority class (the floor) | - | 35.3 | 50.5 | 52.7 | 46.2 | - |
| A' looped + gated merge (4 blocks) | 1 | 34.5 +- 0.5 | 49.3 +- 0.2 | 50.4 +- 0.2 | 44.7 | 0.79 |
| A' looped + gated merge (4 blocks) | 4 | 34.6 +- 0.5 | 48.8 +- 0.5 | 52.5 +- 0.3 | 45.3 | 0.92 |
| A' looped + gated merge (4 blocks) | 8 | 35.1 +- 0.3 | 49.0 +- 0.4 | 52.6 +- 0.9 | 45.6 | 0.88 |
| B' looped, no merge (4 blocks) | 1 | 34.7 +- 0.3 | 49.6 +- 0.2 | 49.5 +- 1.4 | 44.6 | 0.79 |
| B' looped, no merge (4 blocks) | 4 | 34.7 +- 0.3 | 49.0 +- 0.6 | 52.6 +- 0.1 | 45.4 | 0.90 |
| B' looped, no merge (4 blocks) | 8 | 35.1 +- 0.2 | 48.9 +- 0.7 | 52.6 +- 0.6 | 45.6 | 0.90 |
| C' flat (4 blocks) | - | 34.9 +- 0.3 | 49.0 +- 0.2 | 52.2 +- 0.8 | 45.4 | 0.84 |
| D' flat, 10 blocks (iso-FLOP control) | - | 38.4 +- 1.8 | 52.1 +- 0.7 | 52.9 +- 2.3 | 47.8 | 0.67 |
| E' looped + gated merge, input injection (4 blocks) | 1 | 35.1 +- 0.3 | 49.1 +- 0.4 | 51.1 +- 1.7 | 45.1 | 0.80 |
| E' looped + gated merge, input injection (4 blocks) | 4 | 34.5 +- 0.4 | 49.4 +- 0.1 | 52.0 +- 1.6 | 45.3 | 0.84 |
| E' looped + gated merge, input injection (4 blocks) | 8 | 34.5 +- 0.3 | 49.4 +- 0.1 | 51.9 +- 1.5 | 45.3 | 0.84 |
| A looped + gated merge (8 blocks) | 1 | 35.5 | 49.9 | 51.8 | 45.7 | 0.85 |
| A looped + gated merge (8 blocks) | 4 | 33.7 | 50.9 | 53.8 | 46.1 | 0.85 |
| A looped + gated merge (8 blocks) | 8 | 33.3 | 51.2 | 52.7 | 45.7 | 0.82 |
| B looped, no merge (8 blocks) | 1 | 37.2 | 49.8 | 52.7 | 46.6 | 0.86 |
| B looped, no merge (8 blocks) | 4 | 34.8 | 51.7 | 55.4 | 47.3 | 0.82 |
| B looped, no merge (8 blocks) | 8 | 34.7 | 52.8 | 54.9 | 47.5 | 0.81 |
| C flat (8 blocks) | - | 35.8 | 50.0 | 49.3 | 45.0 | 0.74 |
| ModernBERT-large (the teacher) | - | 41.8 | 50.7 | 54.0 | 48.8 | 0.86 |
| RoBERTa-base | - | 39.4 | 50.2 | 57.2 | 48.9 | 0.69 |
| DeBERTa-base (v1) | - | 42.4 | 49.8 | 56.0 | 49.4 | 0.80 |
| DistilBERT-base-cased | - | 36.8 | 49.4 | 51.3 | 45.8 | 0.72 |

**Zero-shot accuracy (%), raw label-word scores, mean of the two prompts; mean +- sample std over 3 seeds where there are 3.**

| model | depth | MNLI | QNLI | RTE | mean | collapse |
|---|---|---:|---:|---:|---:|---:|
| majority class (the floor) | - | 35.3 | 50.5 | 52.7 | 46.2 | - |
| A' looped + gated merge (4 blocks) | 1 | 34.5 +- 0.7 | 50.3 +- 0.7 | 47.0 +- 1.5 | 43.9 | 0.81 |
| A' looped + gated merge (4 blocks) | 4 | 34.3 +- 0.2 | 50.0 +- 0.4 | 46.9 +- 1.5 | 43.7 | 0.81 |
| A' looped + gated merge (4 blocks) | 8 | 34.3 +- 0.1 | 50.2 +- 0.3 | 47.7 +- 0.6 | 44.1 | 0.80 |
| B' looped, no merge (4 blocks) | 1 | 34.7 +- 0.6 | 50.4 +- 0.8 | 48.0 +- 1.3 | 44.3 | 0.80 |
| B' looped, no merge (4 blocks) | 4 | 34.7 +- 0.5 | 50.1 +- 0.1 | 48.0 +- 2.1 | 44.3 | 0.83 |
| B' looped, no merge (4 blocks) | 8 | 34.4 +- 0.2 | 49.9 +- 0.1 | 48.4 +- 0.5 | 44.3 | 0.83 |
| C' flat (4 blocks) | - | 35.3 +- 0.4 | 49.3 +- 0.3 | 47.4 +- 0.7 | 44.0 | 0.80 |
| D' flat, 10 blocks (iso-FLOP control) | - | 40.6 +- 0.3 | 52.1 +- 0.3 | 52.6 +- 1.2 | 48.4 | 0.72 |
| E' looped + gated merge, input injection (4 blocks) | 1 | 35.4 +- 0.1 | 49.6 +- 0.7 | 47.8 +- 0.4 | 44.3 | 0.79 |
| E' looped + gated merge, input injection (4 blocks) | 4 | 35.1 +- 0.1 | 49.4 +- 0.5 | 48.2 +- 1.4 | 44.3 | 0.77 |
| E' looped + gated merge, input injection (4 blocks) | 8 | 35.2 +- 0.1 | 49.2 +- 0.7 | 47.6 +- 1.4 | 44.0 | 0.76 |
| A looped + gated merge (8 blocks) | 1 | 35.4 | 50.6 | 49.6 | 45.2 | 0.88 |
| A looped + gated merge (8 blocks) | 4 | 38.5 | 50.6 | 51.4 | 46.9 | 0.88 |
| A looped + gated merge (8 blocks) | 8 | 39.5 | 51.3 | 54.3 | 48.4 | 0.79 |
| B looped, no merge (8 blocks) | 1 | 36.8 | 50.4 | 51.1 | 46.1 | 0.88 |
| B looped, no merge (8 blocks) | 4 | 41.4 | 51.1 | 53.6 | 48.7 | 0.76 |
| B looped, no merge (8 blocks) | 8 | 41.7 | 50.5 | 54.9 | 49.0 | 0.69 |
| C flat (8 blocks) | - | 38.4 | 50.9 | 50.7 | 46.7 | 0.82 |
| ModernBERT-large (the teacher) | - | 43.6 | 50.5 | 53.6 | 49.3 | 0.90 |
| RoBERTa-base | - | 39.6 | 49.6 | 54.3 | 47.9 | 0.85 |
| DeBERTa-base (v1) | - | 37.3 | 49.5 | 53.1 | 46.6 | 0.98 |
| DistilBERT-base-cased | - | 39.7 | 49.8 | 48.0 | 45.8 | 0.63 |

Masked-LM bits per byte of text (lower is better); mean +- sample std over seeds where there is more than one. Texts: 128 x 1000 characters
per source, partition masking at a 12.5% rate (every token scored once). `best` = the depth with the lowest mean on these same texts.

| model | seeds | depth | fineweb-holdout | arxiv | wikipedia | python-code | mean |
|---|---:|---|---:|---:|---:|---:|---:|
| A' looped + gated merge | 3 | 1 | 0.9166 +- 0.0004 | 1.0719 +- 0.0019 | 0.8860 +- 0.0035 | 1.3419 +- 0.0258 | 1.0541 |
| A' looped + gated merge | 3 | 4 (best) | 0.7891 +- 0.0009 | 0.9433 +- 0.0008 | 0.7638 +- 0.0001 | 1.2169 +- 0.0059 | 0.9283 |
| A' looped + gated merge | 3 | 8 | 0.7954 +- 0.0007 | 0.9630 +- 0.0004 | 0.7707 +- 0.0007 | 1.2653 +- 0.0081 | 0.9486 |
| B' looped, no merge | 3 | 1 | 0.9201 +- 0.0003 | 1.0749 +- 0.0023 | 0.8892 +- 0.0032 | 1.3485 +- 0.0264 | 1.0582 |
| B' looped, no merge | 3 | 4 (best) | 0.7917 +- 0.0006 | 0.9485 +- 0.0031 | 0.7662 +- 0.0008 | 1.2173 +- 0.0007 | 0.9309 |
| B' looped, no merge | 3 | 8 | 0.7988 +- 0.0005 | 0.9707 +- 0.0034 | 0.7740 +- 0.0006 | 1.2678 +- 0.0059 | 0.9528 |
| C' flat (no loop) | 3 | - | 0.8449 +- 0.0002 | 0.9937 +- 0.0033 | 0.8117 +- 0.0007 | 1.2609 +- 0.0091 | 0.9778 |
| D' flat, 10 blocks (the FLOPs of B' at depth 4) | 3 | - | 0.5965 +- 0.0009 | 0.7437 +- 0.0043 | 0.5821 +- 0.0005 | 0.8019 +- 0.0022 | 0.6810 |
| E' looped + gated merge, input injection (RLT form) | 3 | 1 | 0.8719 +- 0.0010 | 1.0280 +- 0.0059 | 0.8358 +- 0.0039 | 1.2908 +- 0.0146 | 1.0066 |
| E' looped + gated merge, input injection (RLT form) | 3 | 5 (best) | 0.8095 +- 0.0003 | 0.9587 +- 0.0040 | 0.7832 +- 0.0023 | 1.2186 +- 0.0118 | 0.9425 |
| E' looped + gated merge, input injection (RLT form) | 3 | 8 | 0.8091 +- 0.0005 | 0.9589 +- 0.0045 | 0.7828 +- 0.0026 | 1.2209 +- 0.0135 | 0.9429 |
| A looped + gated merge (8 blocks, 200M tokens) | 1 | 1 | 0.6149 | 0.7665 | 0.5859 | 0.8630 | 0.7076 |
| A looped + gated merge (8 blocks, 200M tokens) | 1 | 5 (best) | 0.5242 | 0.6666 | 0.5133 | 0.7494 | 0.6134 |
| A looped + gated merge (8 blocks, 200M tokens) | 1 | 8 | 0.5273 | 0.6720 | 0.5167 | 0.7682 | 0.6210 |
| B looped, no merge (8 blocks) | 1 | 1 | 0.6149 | 0.7651 | 0.5847 | 0.8680 | 0.7082 |
| B looped, no merge (8 blocks) | 1 | 4 (best) | 0.5266 | 0.6677 | 0.5137 | 0.7550 | 0.6158 |
| B looped, no merge (8 blocks) | 1 | 8 | 0.5294 | 0.6761 | 0.5185 | 0.7764 | 0.6251 |
| C flat (8 blocks) | 1 | - | 0.5665 | 0.7150 | 0.5431 | 0.7953 | 0.6550 |
| ModernBERT-large (the teacher) | 1 | - | 0.3145 | 0.3264 | 0.2810 | 0.1906 | 0.2781 |
| RoBERTa-base | 1 | - | 0.6182 | 0.7324 | 0.6649 | 1.6438 | 0.9148 |
| DeBERTa-base (v1) | 1 | - | 0.6734 | 0.7768 | 0.7215 | 1.5215 | 0.9233 |
| DistilBERT-base-cased | 1 | - | 0.9906 | 0.6806 | 0.7683 | 1.6248 | 1.0161 |

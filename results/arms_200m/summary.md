# 8-block arms, 200M tokens (one seed each): held-out text

512 sequences x 512 tokens per source, 78,314 masked positions; a flat arm has no loop (depth 1 only). `best` is chosen on these same sequences.

## arxiv (teacher CE 1.4179)

distillation loss (lower is better):

| arm | depth 1 | depth 2 | depth 4 | depth 5 | depth 8 | best |
|---|---:|---:|---:|---:|---:|---:|
| A looped + gated merge | 2.7866 | 2.4474 | 2.3081 | 2.3000 | 2.3224 | 2.3000 (depth 5) |
| B looped, no merge | 2.7869 | 2.4482 | 2.3177 | 2.3141 | 2.3489 | 2.3141 (depth 5) |
| C flat (no loop) | 2.5496 | - | - | - | - | 2.5496 (depth 1) |

cross-entropy against the true token:

| arm | depth 1 | depth 2 | depth 4 | depth 5 | depth 8 | best |
|---|---:|---:|---:|---:|---:|---:|
| A looped + gated merge | 2.9168 | 2.6485 | 2.5340 | 2.5269 | 2.5423 | 2.5269 (depth 5) |
| B looped, no merge | 2.9196 | 2.6510 | 2.5397 | 2.5354 | 2.5608 | 2.5354 (depth 5) |
| C flat (no loop) | 2.7388 | - | - | - | - | 2.7388 (depth 1) |

top-1 accuracy:

| arm | depth 1 | depth 2 | depth 4 | depth 5 | depth 8 | best |
|---|---:|---:|---:|---:|---:|---:|
| A looped + gated merge | 0.4947 | 0.5314 | 0.5477 | 0.5490 | 0.5461 | 0.5490 (depth 5) |
| B looped, no merge | 0.4963 | 0.5326 | 0.5487 | 0.5497 | 0.5470 | 0.5497 (depth 5) |
| C flat (no loop) | 0.5212 | - | - | - | - | 0.5212 (depth 1) |

## fineweb-holdout (teacher CE 1.5330)

distillation loss (lower is better):

| arm | depth 1 | depth 2 | depth 4 | depth 5 | depth 8 | best |
|---|---:|---:|---:|---:|---:|---:|
| A looped + gated merge | 2.3736 | 2.0758 | 1.9473 | 1.9364 | 1.9438 | 1.9347 (depth 6) |
| B looped, no merge | 2.3798 | 2.0814 | 1.9550 | 1.9448 | 1.9544 | 1.9435 (depth 6) |
| C flat (no loop) | 2.1619 | - | - | - | - | 2.1619 (depth 1) |

cross-entropy against the true token:

| arm | depth 1 | depth 2 | depth 4 | depth 5 | depth 8 | best |
|---|---:|---:|---:|---:|---:|---:|
| A looped + gated merge | 2.6062 | 2.3676 | 2.2640 | 2.2546 | 2.2590 | 2.2525 (depth 6) |
| B looped, no merge | 2.6122 | 2.3729 | 2.2710 | 2.2621 | 2.2675 | 2.2599 (depth 6) |
| C flat (no loop) | 2.4378 | - | - | - | - | 2.4378 (depth 1) |

top-1 accuracy:

| arm | depth 1 | depth 2 | depth 4 | depth 5 | depth 8 | best |
|---|---:|---:|---:|---:|---:|---:|
| A looped + gated merge | 0.5247 | 0.5604 | 0.5761 | 0.5778 | 0.5798 | 0.5798 (depth 8) |
| B looped, no merge | 0.5240 | 0.5591 | 0.5764 | 0.5774 | 0.5778 | 0.5781 (depth 7) |
| C flat (no loop) | 0.5508 | - | - | - | - | 0.5508 (depth 1) |

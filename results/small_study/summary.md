Final held-out distillation loss (256 FineWeb hold-out sequences), mean +- sample std over seeds. A flat arm has no loop: one value.
The 10-block flat arm is the iso-FLOP control: as many block passes as a looped arm at depth 4.

| arm | seeds | depth 1 | depth 2 | depth 4 | depth 8 |
|---|---:|---:|---:|---:|---:|
| looped + gated merge | 3 | 3.4337 +- 0.0025 | 3.0714 +- 0.0011 | 2.9254 +- 0.0011 | 2.9363 +- 0.0017 |
| looped, no merge | 3 | 3.4452 +- 0.0029 | 3.0801 +- 0.0002 | 2.9343 +- 0.0019 | 2.9476 +- 0.0032 |
| flat (no loop) | 3 | 3.1530 +- 0.0014 | - | - | - |
| flat, 10 blocks (iso-FLOP control) | 3 | 2.2675 +- 0.0014 | - | - | - |
| looped + gated merge, input injection (RLT form) | 3 | 3.2742 +- 0.0033 | 3.0480 +- 0.0018 | 2.9813 +- 0.0024 | 2.9770 +- 0.0028 |

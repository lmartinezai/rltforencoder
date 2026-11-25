# Halting targets compared (arm A, 8 blocks): an earlier exploration, kept as run on 2026-11-15

Produced by `src/halting_fit.py` (A8_linear.json: linear heads; A8_mlp.json: Linear-GELU-Linear heads) on feature shards of
`runs/arm_a_merge/checkpoint-12208` (`feats/a_*`, stream data seed 4242): 10,240 training sequences (9,728 + 512) and 512 held-out FineWeb and
512 held-out arXiv sequences. These held-out sets differ from the ones in ../summary.md (stored-target hold-out), so the ceilings differ.
Targets: v1 = within 0.02 of the sequence's best depth over all eight (the target of the head in ../summary.md); future0 / future = no later
depth better by more than 0 / 0.005 (causal); gain = regress the improvement still to come and stop below a price per loop.
Features: S state, SD state + depth, CD scalars (KL to previous depth, state change, entropy, top probability) + depth, SCD all.
Gain = loss improvement over linear interpolation of the fixed-depth losses at the policy's mean loop count, over operating points with
2.5-7.5 loops; best = max over the thresholds, chosen on the held-out set itself (optimistic), mean = mean over those points. Thresholds: 30
probability levels from 0.05 to 0.99 for the stop targets, 25 quantiles of the predicted gain for the gain regression.
Linear heads: every target and feature set, 30 epochs. MLP heads (256 hidden units, 10 epochs): targets v1, future and gain with feature sets SD
and SCD only. Both: lr 3e-3, seed 0, features standardised with the training mean and std. Exact commands and code revisions: paper/MANIFEST.md.

Ceiling (hindsight, sweep of 60 multipliers): FineWeb best +0.0149, arXiv best +0.0213.

| head | target/features | FineWeb best | FineWeb mean | arXiv best | arXiv mean |
|---|---|---:|---:|---:|---:|
| linear | v1/S | +0.0014 | +0.0001 | +0.0033 | +0.0013 |
| linear | v1/SD | +0.0014 | +0.0002 | +0.0033 | +0.0014 |
| linear | v1/CD | +0.0025 | +0.0003 | +0.0041 | +0.0004 |
| linear | v1/SCD | +0.0019 | +0.0003 | +0.0046 | +0.0017 |
| linear | future0/S | +0.0003 | -0.0006 | +0.0013 | +0.0006 |
| linear | future0/SD | +0.0006 | -0.0007 | +0.0015 | +0.0007 |
| linear | future0/CD | +0.0005 | -0.0004 | +0.0014 | -0.0005 |
| linear | future0/SCD | +0.0007 | -0.0007 | +0.0014 | +0.0007 |
| linear | future/S | +0.0005 | -0.0004 | +0.0016 | +0.0002 |
| linear | future/SD | +0.0007 | -0.0003 | +0.0016 | +0.0002 |
| linear | future/CD | +0.0009 | -0.0011 | +0.0018 | -0.0029 |
| linear | future/SCD | +0.0009 | -0.0003 | +0.0016 | +0.0002 |
| linear | gain/S | +0.0038 | -0.0040 | +0.0038 | -0.0013 |
| linear | gain/SD | +0.0037 | -0.0034 | +0.0045 | -0.0004 |
| linear | gain/CD | +0.0033 | -0.0017 | +0.0044 | -0.0033 |
| linear | gain/SCD | +0.0021 | -0.0036 | +0.0027 | -0.0003 |
| MLP | v1/SD | +0.0009 | -0.0004 | +0.0040 | +0.0006 |
| MLP | v1/SCD | +0.0019 | -0.0001 | +0.0051 | +0.0009 |
| MLP | future/SD | -0.0000 | -0.0009 | +0.0011 | +0.0004 |
| MLP | future/SCD | +0.0001 | -0.0006 | +0.0013 | +0.0003 |
| MLP | gain/SD | +0.0054 | +0.0003 | +0.0063 | -0.0002 |
| MLP | gain/SCD | +0.0056 | +0.0005 | +0.0061 | +0.0011 |

# Compute ledger

Total: **2,652 PFLOP** (2.65 EFLOP).

| group | PFLOP |
|---|---:|
| history before this study | 640.6 |
| 200M-token arms | 1,147.3 |
| halting v1 (4 heads) | 125.7 |
| evaluations | 46.0 |
| halting v2 features | 13.8 |
| small-model study | 678.8 |

| group | job | sequences | GFLOP/seq | PFLOP | note |
|---|---|---:|---:|---:|---|
| history before this study | bort1 (joint, local teacher): student training | 32,000 | 938 | 30.0 | approximate (log) |
| history before this study | bort1 (joint, local teacher): teacher | 32,000 | 352 | 11.3 | approximate (log) |
| history before this study | bort100m (joint, local teacher): student training | 102,400 | 938 | 96.1 | approximate (log; part of it resumed from bort1) |
| history before this study | bort100m (joint, local teacher): teacher | 102,400 | 352 | 36.0 | approximate (log; part of it resumed from bort1) |
| history before this study | bort_shared: student training | 198,400 | 938 | 186.2 | shared teacher stream: one teacher pass for both |
| history before this study | bort_shared: teacher | 198,400 | 352 | 69.8 | shared teacher stream: one teacher pass for both |
| history before this study | flat_shared (assumed same length): student training | 198,400 | 356 | 70.7 | teacher pass counted once, under bort_shared; approximate |
| history before this study | flat100m: student training | 198,400 | 356 | 70.7 | approximate (log) |
| history before this study | flat100m: teacher | 198,400 | 352 | 69.8 | approximate (log) |
| 200M-token arms | A looped + merge: student training | 390,656 | 938 | 366.6 | 12,208 steps x 32 seq |
| 200M-token arms | B looped, no merge: student training | 390,656 | 938 | 366.6 | 12,208 steps x 32 seq |
| 200M-token arms | C flat: student training | 390,656 | 356 | 139.3 | 12,208 steps x 32 seq |
| 200M-token arms | teacher streams (A+C shared one pass, B a second) | 781,312 | 352 | 274.8 | 2 passes x 390,656 seq |
| halting v1 (4 heads) | frozen student at 8 depths (slow, per-depth runs) | 38,400 | 2,469 | 94.8 | 300 steps x 32 seq x 4 heads |
| halting v1 (4 heads) | teacher | 38,400 | 352 | 13.5 |  |
| halting v1 (4 heads) | head evals (6 x 256 seq per head) | 6,144 | 2,821 | 17.3 |  |
| evaluations | bench_heldout, 4 runs (2 invalid) x (A, B at 8 depths, C) x 1024 seq | 4,096 | 5,995 | 24.6 | approximate counts |
| evaluations | halting_stats (A, B; 1024 seq each, slow table) | 2,048 | 2,821 | 5.8 | approximate counts |
| evaluations | halting_free (A, 512 seq): slow table + 6 token-rule passes + flat C | 512 | 6,362 | 3.3 | approximate counts |
| evaluations | halting_eval, final run (A, B; 256 seq: table + 3 change + 3 head passes) + the killed first run | 665 | 5,878 | 3.9 | approximate counts |
| evaluations | flops_score + seq_max_exit (A, B and A; ~1024 seq slow tables) | 1,024 | 2,821 | 2.9 | approximate counts |
| evaluations | bench_fast + autotune pools (small) | 2,000 | 2,821 | 5.6 | approximate counts |
| halting v2 features | feature shards: 10,240 train + 512 check + 1,024 held-out seq (fast one-pass table) | 11,776 | 823 | 9.7 |  |
| halting v2 features | teacher for the shards | 11,776 | 352 | 4.1 |  |
| small-model study | teacher cache build (98,304 + 256 seq, local teacher) | 98,560 | 352 | 34.7 | one pass, reused by all fifteen runs |
| small-model study | A', B' training: 6 runs x 3072 steps x 32 seq | 589,824 | 516 | 304.4 | 1+2+1 blocks, depth 1..8 |
| small-model study | C' training: 3 runs x 3072 steps x 32 seq | 294,912 | 203 | 59.8 | 4 flat blocks |
| small-model study | D' training: 3 runs x 3072 steps x 32 seq | 294,912 | 433 | 127.8 | 10 flat blocks (iso-FLOP control) |
| small-model study | E' training: 3 runs x 3072 steps x 32 seq | 294,912 | 516 | 152.2 | 1+2+1 blocks, depth 1..8, input injection (merge_base anchor) |

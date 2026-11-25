# Halting heads, all looped models

Sequence-level linear head on the mean state (this is `BortModel.halting_head`), trained on 4,096 fresh FineWeb sequences from stored teacher targets and scored exactly on 512 held-out FineWeb and 512 arXiv sequences. Gain = loss improvement over the fixed depth with the same mean loop count (best over thresholds / mean over operating points with 2.5-7.5 loops); ceiling = the best any head could do with hindsight. Token head: held-out accuracy at the stated base rate of 'can stop'.

| model | FineWeb gain best / mean | FineWeb ceiling | arXiv gain best / mean | arXiv ceiling | token head acc (base rate) | checkpoint with the head |
|---|---|---:|---|---:|---|---|
| a_s1 A' looped + merge (4 blocks) | +0.0022 / -0.0002 | +0.0157 | +0.0005 / -0.0045 | +0.0174 | 0.651 (0.47) | `runs\small\a_s1\checkpoint-3072-halting` |
| a_s2 A' looped + merge (4 blocks) | +0.0019 / -0.0002 | +0.0158 | +0.0009 / -0.0087 | +0.0166 | 0.659 (0.47) | `runs\small\a_s2\checkpoint-3072-halting` |
| a_s3 A' looped + merge (4 blocks) | +0.0020 / -0.0000 | +0.0160 | +0.0001 / -0.0075 | +0.0170 | 0.658 (0.47) | `runs\small\a_s3\checkpoint-3072-halting` |
| b_s1 B' looped, no merge (4 blocks) | +0.0019 / -0.0004 | +0.0156 | +0.0010 / -0.0036 | +0.0177 | 0.656 (0.46) | `runs\small\b_s1\checkpoint-3072-halting` |
| b_s2 B' looped, no merge (4 blocks) | +0.0019 / -0.0003 | +0.0156 | +0.0018 / -0.0069 | +0.0151 | 0.657 (0.47) | `runs\small\b_s2\checkpoint-3072-halting` |
| b_s3 B' looped, no merge (4 blocks) | +0.0020 / -0.0001 | +0.0162 | +0.0001 / -0.0081 | +0.0178 | 0.658 (0.47) | `runs\small\b_s3\checkpoint-3072-halting` |
| A8 A looped + merge (8 blocks) | +0.0016 / +0.0003 | +0.0148 | +0.0014 / +0.0000 | +0.0177 | 0.687 (0.56) | `runs\arm_a_merge\checkpoint-12208-halting` |
| B8 B looped, no merge (8 blocks) | +0.0020 / +0.0004 | +0.0143 | +0.0025 / -0.0009 | +0.0183 | 0.691 (0.55) | `runs\arm_b_nomerge\checkpoint-12208-halting` |

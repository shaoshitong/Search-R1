# Full L15/P80 MSE: completed-epoch development Top512 composition

Read-only snapshot `1792705672.3557618`; aggregation `0.380s`. Source uses only atomically published completed evaluation JSON and its closed dev2048 saved predictions. E0 is initial untrained evaluation, not a completed training epoch.

| seed / job | epoch | state | all wrong | Mixed | all correct | Mixed fraction | all-wrong share of remaining errors |
|---|---:|---|---:|---:|---:|---:|---:|
| seed1 / 750189 | 0 | raw | 280 | 175 | 57 | 34.18% | 83.09% |
| seed1 / 750189 | 0 | EMA | 280 | 175 | 57 | 34.18% | 83.09% |
| seed1 / 750189 | 1 | raw | 190 | 284 | 38 | 55.47% | 83.33% |
| seed1 / 750189 | 1 | EMA | 212 | 248 | 52 | 48.44% | 80.30% |
| seed1 / 750189 | 2 | raw | 187 | 310 | 15 | 60.55% | 92.57% |
| seed1 / 750189 | 2 | EMA | 201 | 274 | 37 | 53.52% | 84.45% |
| seed2 / 750190 | 0 | raw | 232 | 155 | 125 | 30.27% | 64.99% |
| seed2 / 750190 | 0 | EMA | 232 | 155 | 125 | 30.27% | 64.99% |
| seed2 / 750190 | 1 | raw | 202 | 286 | 24 | 55.86% | 89.38% |
| seed2 / 750190 | 1 | EMA | 186 | 249 | 77 | 48.63% | 70.72% |
| seed2 / 750190 | 2 | raw | 196 | 310 | 6 | 60.55% | 97.03% |
| seed2 / 750190 | 2 | EMA | 196 | 281 | 35 | 54.88% | 84.85% |
| seed2 / 750190 | 3 | raw | 192 | 317 | 3 | 61.91% | 98.46% |
| seed2 / 750190 | 3 | EMA | 195 | 304 | 13 | 59.38% | 93.75% |

All14 raw/EMA rows exactly reproduce the corresponding evaluation Top512 Mixed counts. Every epoch and both seeds match the same2048 question IDs, ordered own five trajectory IDs and strict EM labels; identitySHA `0520e04dd7fc6e8eb9f66865299d6345c013ca4222141f794255bc57d9954914`. Pool classes remain663 Mixed/1055 all-wrong/330 all-correct. Ranking uses saved scores only, ties by qid, and never the labels or a classification threshold.

At this snapshot seed1 has finished E2 and seed2 E3. Their latest raw selected sets contain187 all-wrong/15 all-correct and192 all-wrong/3 all-correct respectively. Existing aggregate error fractions are92.57% and98.46% all-wrong. Thus the remaining selected nonMixed questions are chiefly all-wrong; fewer all-correct distractors remain. These are different-progress snapshots, not a same-epoch controlled comparison or a causal transition analysis.

This is repeatedly viewed NQ development, not final4096. No epoch/state was selected, no seed was chosen as best, and the prospective seed1-only full-budget selection rule is unchanged. Complete source proof hashes, each K=0..5 strictEM-count histogram and all source evaluation/prediction paths are in [actual-summary.json](actual-summary.json); no per-question output was exported to Mac. No checkpoint/model tensors, final pools, new inference or Slurm commands were used; no remote file or training state was changed.

Published read script source62de4f53975cb3aa3fd296abb4bc12593dc3b8d5 / SHA e2122619efc791e33c9a380a4dc8f19a18f21047aebd6f3ee424c1109a813aa1. Each completed evaluation and prediction artifact was SHA-checked before and after aggregation to reject a concurrently changed artifact.

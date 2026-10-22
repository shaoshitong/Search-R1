# Seven matched pilots: all epochs on viewed validation2048

This ledger reports all58 preplanned entries: seven layer/prefix cells × epochs0/1/2/3 × raw/EMA, plus original E1 raw/EMA. All have identical2048 question IDs, exact ownG5 trajectory IDs/order and strict-EM labels (663 Mixed,1055 all-wrong,330 all-correct). Identity SHA `c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7`. The question pool has already been viewed for model selection; it is not the untouched final4096.

Suggested candidate for a later explicitly frozen confirmation is **L21/P80 epoch3 raw**, because it leads BA and AUC and improves overall ranking versus E1. This recommendation does not freeze a candidate or trigger final scoring. E2 raw selects219 Mixed/512 versus E3 raw215, a four-question difference without reliable evidence of an advantage. Epoch3 has the completed independent CPU model/Adam/two-rank RNG acceptance.

| Model | BA % | AUC % | BCE | Mixed recall % | nonMixed FPR % | Top512 Mixed |
|---|---:|---:|---:|---:|---:|---:|
| E1_raw | 52.28 | 59.28 | 0.64381 | 13.88 | 9.31 | 201/512 (39.26%) |
| E1_EMA | 52.88 | 59.50 | 0.65302 | 21.87 | 16.10 | 205/512 (40.04%) |
| L27P20_E3_raw | 53.88 | 56.95 | 0.77607 | 89.14 | 81.37 | 200/512 (39.06%) |
| L27P20_E3_EMA | 53.90 | 56.73 | 0.67493 | 45.70 | 37.91 | 198/512 (38.67%) |
| L21P20_E3_raw | 57.23 | 58.88 | 0.71192 | 69.83 | 55.38 | 198/512 (38.67%) |
| L21P20_E3_EMA | 56.34 | 58.80 | 0.68501 | 58.67 | 45.99 | 207/512 (40.43%) |
| L27P80_E3_raw | 57.88 | 59.37 | 0.69855 | 66.67 | 50.90 | 197/512 (38.48%) |
| L27P80_E3_EMA | 55.95 | 57.99 | 0.67536 | 52.34 | 40.43 | 200/512 (39.06%) |
| L21P80_E3_raw | 59.94 | 62.21 | 0.68676 | 68.17 | 48.30 | 215/512 (41.99%) |
| L21P80_E3_EMA | 58.75 | 61.23 | 0.69015 | 70.14 | 52.64 | 214/512 (41.80%) |
| L15P20_E3_raw | 58.58 | 60.02 | 0.68767 | 63.95 | 46.79 | 200/512 (39.06%) |
| L15P20_E3_EMA | 57.10 | 58.97 | 0.68406 | 60.48 | 46.28 | 205/512 (40.04%) |
| L18P20_E3_raw | 58.65 | 60.93 | 0.72786 | 77.53 | 60.22 | 206/512 (40.23%) |
| L18P20_E3_EMA | 57.22 | 59.69 | 0.67613 | 54.30 | 39.86 | 205/512 (40.04%) |
| L24P20_E3_raw | 54.21 | 55.82 | 0.72948 | 70.44 | 62.02 | 187/512 (36.52%) |
| L24P20_E3_EMA | 54.19 | 55.73 | 0.69337 | 59.43 | 51.05 | 188/512 (36.72%) |

The fixed0.5 threshold is unchanged. L21/P80 E3 raw raises Mixed recall21.87%→68.17% but also nonMixed FPR16.10%→48.30% relative to E1 EMA. Its errors include490/1055 all-wrong→Mixed,179/330 all-correct→Mixed, and211/663 Mixed→nonMixed. The AUC gain shows more than a threshold shift, but classification remains imperfect and BCE worsens0.65302→0.68676. It is not justified to describe this as calibrated or solved.

## Paired differences

All intervals are1000 paired question-bootstrap percentile95%, using identical resamples and reselecting Top512 with the same exact qid tie rule. They reflect this validation sample, not training-seed variation, and are unadjusted exploratory intervals after examining many candidates.

| Candidate minus baseline | ΔBA pp [95%] | ΔAUC pp [95%] | ΔTop512 percentage pp [95%] | ΔBCE |
|---|---:|---:|---:|---:|
| L21P80_E3_raw − E1_EMA | +7.05 [+4.42,+9.55] | +2.71 [+0.52,+5.18] | +1.95 [-1.76,+6.45] | +0.03375 |
| L21P80_E3_raw − E1_raw | +7.65 [+5.14,+10.17] | +2.94 [+0.62,+5.26] | +2.73 [-1.37,+7.23] | +0.04295 |
| L21P80_E3_raw − L27P80_E3_raw | +2.05 [-0.04,+3.93] | +2.84 [+1.18,+4.55] | +3.52 [+0.20,+7.62] | -0.01179 |
| L21P80_E3_raw − L21P20_E3_raw | +2.71 [+0.88,+4.54] | +3.33 [+1.88,+4.76] | +3.32 [+0.98,+7.04] | -0.02515 |
| L21P80_E3_raw − L18P20_E3_raw | +1.28 [-0.82,+3.22] | +1.28 [-0.34,+2.88] | +1.76 [-1.56,+5.08] | -0.04110 |
| L21P80_E3_raw − L21P80_E2_raw | +1.83 [+0.05,+3.67] | +0.79 [-0.50,+2.12] | -0.78 [-3.12,+2.73] | -0.04969 |
| L21P80_E3_raw − L21P80_E3_EMA | +1.19 [-0.63,+2.92] | +0.98 [-0.38,+2.37] | +0.20 [-2.73,+3.32] | -0.00339 |

The matched L21/P80 versus L27/P80 comparison supports a depth-dependent difference at the same8192-question/192-update budget. L21/P80 versus L21/P20 is a matched prefix comparison. Neither establishes that learning rate caused the effect; learning rate was held fixed. All pilots used the same fixture, ownG5 plans, nominal budget and LR, but each layer received its corresponding fresh official block and newly initialized head.

## Complete epoch ledger

| Model | BA % | AUC % | BCE | Mixed recall % | nonMixed FPR % | all-wrong→Mixed % | all-correct→Mixed % | Top512 Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1_raw | 52.28 | 59.28 | 0.64381 | 13.88 | 9.31 | 8.53 | 11.82 | 201/512 |
| E1_EMA | 52.88 | 59.50 | 0.65302 | 21.87 | 16.10 | 13.93 | 23.03 | 205/512 |
| L27P20_E0_raw | 49.48 | 50.12 | 0.77200 | 89.29 | 90.32 | 91.18 | 87.58 | 164/512 |
| L27P20_E0_EMA | 49.48 | 50.12 | 0.77200 | 89.29 | 90.32 | 91.18 | 87.58 | 164/512 |
| L27P20_E1_raw | 50.72 | 55.95 | 0.64432 | 3.47 | 2.02 | 1.90 | 2.42 | 181/512 |
| L27P20_E1_EMA | 52.06 | 52.31 | 0.71278 | 68.02 | 63.90 | 65.40 | 59.09 | 175/512 |
| L27P20_E2_raw | 54.62 | 56.75 | 0.72624 | 78.13 | 68.88 | 67.68 | 72.73 | 187/512 |
| L27P20_E2_EMA | 53.04 | 55.22 | 0.68340 | 46.00 | 39.93 | 38.77 | 43.64 | 181/512 |
| L27P20_E3_raw | 53.88 | 56.95 | 0.77607 | 89.14 | 81.37 | 78.48 | 90.61 | 200/512 |
| L27P20_E3_EMA | 53.90 | 56.73 | 0.67493 | 45.70 | 37.91 | 35.64 | 45.15 | 198/512 |
| L21P20_E0_raw | 50.00 | 50.10 | 0.80935 | 100.00 | 100.00 | 100.00 | 100.00 | 161/512 |
| L21P20_E0_EMA | 50.00 | 50.10 | 0.80935 | 100.00 | 100.00 | 100.00 | 100.00 | 161/512 |
| L21P20_E1_raw | 51.32 | 56.97 | 0.64588 | 11.01 | 8.38 | 7.30 | 11.82 | 198/512 |
| L21P20_E1_EMA | 50.40 | 52.97 | 0.74730 | 100.00 | 99.21 | 99.05 | 99.70 | 177/512 |
| L21P20_E2_raw | 55.52 | 58.95 | 0.73528 | 81.15 | 70.11 | 67.01 | 80.00 | 210/512 |
| L21P20_E2_EMA | 55.12 | 57.06 | 0.70553 | 77.53 | 67.29 | 65.31 | 73.64 | 201/512 |
| L21P20_E3_raw | 57.23 | 58.88 | 0.71192 | 69.83 | 55.38 | 51.09 | 69.09 | 198/512 |
| L21P20_E3_EMA | 56.34 | 58.80 | 0.68501 | 58.67 | 45.99 | 42.84 | 56.06 | 207/512 |
| L27P80_E0_raw | 49.98 | 49.51 | 0.76465 | 80.39 | 80.43 | 79.72 | 82.73 | 164/512 |
| L27P80_E0_EMA | 49.98 | 49.51 | 0.76465 | 80.39 | 80.43 | 79.72 | 82.73 | 164/512 |
| L27P80_E1_raw | 51.16 | 56.83 | 0.64298 | 5.43 | 3.10 | 3.03 | 3.33 | 191/512 |
| L27P80_E1_EMA | 50.72 | 51.84 | 0.71601 | 65.91 | 64.48 | 64.45 | 64.55 | 179/512 |
| L27P80_E2_raw | 55.60 | 58.58 | 0.72300 | 76.62 | 65.42 | 63.70 | 70.91 | 196/512 |
| L27P80_E2_EMA | 53.92 | 55.70 | 0.68737 | 53.70 | 45.85 | 44.45 | 50.30 | 188/512 |
| L27P80_E3_raw | 57.88 | 59.37 | 0.69855 | 66.67 | 50.90 | 49.29 | 56.06 | 197/512 |
| L27P80_E3_EMA | 55.95 | 57.99 | 0.67536 | 52.34 | 40.43 | 38.48 | 46.67 | 200/512 |
| L21P80_E0_raw | 50.00 | 50.21 | 0.84460 | 100.00 | 100.00 | 100.00 | 100.00 | 168/512 |
| L21P80_E0_EMA | 50.00 | 50.21 | 0.84460 | 100.00 | 100.00 | 100.00 | 100.00 | 168/512 |
| L21P80_E1_raw | 53.66 | 58.97 | 0.64066 | 19.46 | 12.13 | 10.71 | 16.67 | 202/512 |
| L21P80_E1_EMA | 50.18 | 53.93 | 0.76441 | 99.85 | 99.49 | 99.34 | 100.00 | 183/512 |
| L21P80_E2_raw | 58.10 | 61.42 | 0.73645 | 82.20 | 65.99 | 63.51 | 73.94 | 219/512 |
| L21P80_E2_EMA | 56.02 | 58.84 | 0.71299 | 82.81 | 70.76 | 68.06 | 79.39 | 203/512 |
| L21P80_E3_raw | 59.94 | 62.21 | 0.68676 | 68.17 | 48.30 | 46.45 | 54.24 | 215/512 |
| L21P80_E3_EMA | 58.75 | 61.23 | 0.69015 | 70.14 | 52.64 | 49.29 | 63.33 | 214/512 |
| L15P20_E0_raw | 50.00 | 51.06 | 0.76298 | 100.00 | 100.00 | 100.00 | 100.00 | 174/512 |
| L15P20_E0_EMA | 50.00 | 51.06 | 0.76298 | 100.00 | 100.00 | 100.00 | 100.00 | 174/512 |
| L15P20_E1_raw | 50.23 | 56.33 | 0.64439 | 2.26 | 1.81 | 1.80 | 1.82 | 204/512 |
| L15P20_E1_EMA | 49.85 | 53.66 | 0.72924 | 99.70 | 100.00 | 100.00 | 100.00 | 176/512 |
| L15P20_E2_raw | 54.75 | 59.14 | 0.74631 | 87.33 | 77.83 | 75.73 | 84.55 | 204/512 |
| L15P20_E2_EMA | 54.56 | 56.92 | 0.70057 | 76.77 | 67.65 | 66.35 | 71.82 | 194/512 |
| L15P20_E3_raw | 58.58 | 60.02 | 0.68767 | 63.95 | 46.79 | 42.84 | 59.39 | 200/512 |
| L15P20_E3_EMA | 57.10 | 58.97 | 0.68406 | 60.48 | 46.28 | 42.84 | 57.27 | 205/512 |
| L18P20_E0_raw | 50.00 | 51.49 | 0.76006 | 100.00 | 100.00 | 100.00 | 100.00 | 170/512 |
| L18P20_E0_EMA | 50.00 | 51.49 | 0.76006 | 100.00 | 100.00 | 100.00 | 100.00 | 170/512 |
| L18P20_E1_raw | 50.45 | 56.81 | 0.63520 | 2.41 | 1.52 | 1.52 | 1.52 | 203/512 |
| L18P20_E1_EMA | 50.35 | 54.28 | 0.72609 | 99.70 | 98.99 | 98.86 | 99.39 | 183/512 |
| L18P20_E2_raw | 55.74 | 59.91 | 0.75742 | 87.93 | 76.46 | 73.08 | 87.27 | 205/512 |
| L18P20_E2_EMA | 56.59 | 57.69 | 0.69301 | 66.82 | 53.65 | 50.33 | 64.24 | 198/512 |
| L18P20_E3_raw | 58.65 | 60.93 | 0.72786 | 77.53 | 60.22 | 55.64 | 74.85 | 206/512 |
| L18P20_E3_EMA | 57.22 | 59.69 | 0.67613 | 54.30 | 39.86 | 36.49 | 50.61 | 205/512 |
| L24P20_E0_raw | 50.00 | 49.30 | 0.87970 | 100.00 | 100.00 | 100.00 | 100.00 | 162/512 |
| L24P20_E0_EMA | 50.00 | 49.30 | 0.87970 | 100.00 | 100.00 | 100.00 | 100.00 | 162/512 |
| L24P20_E1_raw | 51.21 | 54.72 | 0.65314 | 9.35 | 6.93 | 6.54 | 8.18 | 182/512 |
| L24P20_E1_EMA | 50.00 | 51.04 | 0.78285 | 100.00 | 100.00 | 100.00 | 100.00 | 177/512 |
| L24P20_E2_raw | 54.25 | 56.18 | 0.73633 | 79.03 | 70.54 | 69.19 | 74.85 | 189/512 |
| L24P20_E2_EMA | 53.02 | 54.19 | 0.72180 | 85.97 | 79.93 | 79.62 | 80.91 | 182/512 |
| L24P20_E3_raw | 54.21 | 55.82 | 0.72948 | 70.44 | 62.02 | 57.73 | 75.76 | 187/512 |
| L24P20_E3_EMA | 54.19 | 55.73 | 0.69337 | 59.43 | 51.05 | 48.72 | 58.48 | 188/512 |

Remote complete receipt: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/planned-pilot-all-epochs-viewed2048-20261022-v1/complete.json`, SHA `7ca495c3d2fabba402f28f27ece99914870007bb92dacd1bab518538c3656adb`. Recomputed BCE agrees with every stored actual GPU metric within2e-6. The operation read existing prediction JSONs only and performed no new inference, Slurm mutation, model checkpoint read, threshold tuning, calibrator fit, final4096 read or candidate freeze. Source279bbc416fa6341ffa4e5b3c65e0be3ef98ccdf9 was pushed and ls-remote verified before execution.

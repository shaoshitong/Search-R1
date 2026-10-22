# Viewed2048 P80 two-seed paired comparison

All five models have exactly the same2048 questions, ownG5 trajectory IDs/order and strict EM labels (663 trueMixed). These are already explored development predictions. No checkpoint/model/tensor or final4096 data was opened.

| model | dev Top512Mixed | dev BA | dev AUC | trainB Top512Mixed | trainB BA |
|---|---:|---:|---:|---:|---:|
| E1_EMA | 205 /512 (40.04%) | 52.88% | 0.5950 | not comparable; oldtrainA unused | — |
| CURRENT64_seed1 | 266 /512 (51.95%) | 53.71% | 0.6797 | 384/512 | 56.35% |
| CURRENT64_seed2 | 267 /512 (52.15%) | 59.67% | 0.6890 | 397/512 | 64.78% |
| MSE_M5_seed1 | 278 /512 (54.30%) | 63.51% | 0.7111 | 379/512 | 69.89% |
| MSE_M5_seed2 | 277 /512 (54.10%) | 65.38% | 0.7118 | 371/512 | 71.49% |

The two-seed average is the arithmetic mean of each trained model’s own Top512 yield; scores were not ensembled. Bootstrap resamples2048 questions together across models, then reranks each model and reselects512 for every draw (2000 draws).

- CURRENT64_E5_raw: meanTop512 52.05%; versus E1EMA delta +12.01percentagepoints, paired-question95% [+8.01,+16.50]points.
- MSE_M5_E7_raw: meanTop512 54.20%; versus E1EMA delta +14.16percentagepoints, paired-question95% [+9.77,+18.55]points.

MSE-minus-current64 meanTop512 delta +2.15points, paired-question95% [-0.29,+4.10]points. This interval includes zero. Epochs, global batch/update budget and objectives differ; this is not an isolated causal loss comparison.

These are conditional paired-question intervals for already selected models, **not training-seed confidence intervals**. They do not establish untouched-pool generalization. FixedtrainB is listed separately; no oldE1trainA comparison is made. Current64 sigmoid scores are ranking scores, not calibrated Mixed probabilities; its0.5 BA threshold is diagnostic only. MSE uses predeclared predictedvariance0.08 threshold.

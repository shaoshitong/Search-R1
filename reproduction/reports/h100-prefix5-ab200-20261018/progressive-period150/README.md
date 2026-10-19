# B150 paired probe metrics and actual151 prefix expansion

Original actor149 generated2560 training questionsG5 and the fresh fixed512 heldout questionsG5 using its native40-token hidden/head313. Qid and normalized-question overlaps are zero. Paired before/after metrics use the exact same trajectories and labels; periods use changing actor snapshots and are not a fixed-label learning curve. These classification metrics are not QA EM/F1.

All four original probe journals record50 real Adam writes2050→2100 atLR5e-6, EMA.99, trajectoryBCE+5mixedBCE. Actual151 initial candidate scoring uses actor150 and the migrated EMA150 at80 tokens. The migration binds before/after payloads and preserves the Adam/EMA/RNG clocks without reported resets, with new linear coordinates zero. This is producer-receipt/actual-score binding, not an independent tensor-preservation audit. Later outcome-assisted final512 reranking remains the authorized separate selection stage.

| Head / original distribution | Before BA | After BA | Before trajectory BCE | After trajectory BCE | Before mixed BCE | After mixed BCE |
|---|---:|---:|---:|---:|---:|---:|
| raw / train | 50.3897% | 67.4992% | 0.853177 | 0.512642 | 0.943627 | 0.865177 |
| raw / test | 46.6890% | 55.7860% | 1.146663 | 0.775089 | 0.821101 | 1.349506 |
| ema / train | 53.2566% | 56.8600% | 0.572094 | 0.548429 | 1.052215 | 1.009143 |
| ema / test | 50.4849% | 53.0268% | 0.821076 | 0.796687 | 1.245829 | 1.294077 |

Heldout BA improves for both heads, but ordinary heldout accuracy falls and mixed BCE worsens for both heads. Trajectory BCE improves. This is not uniform probe generalization improvement; no algorithm, loss, threshold or learning rate was changed.

Post-update EMA heldout:52/512 actually mixed (10.1562%), mean predicted q 64.9455%; ordinary accuracy 33.9844% versus majority 89.8438%. Mixed recall 76.9231%, nonmixed recall 29.1304%; TP40/FN12/TN134/FP326. The predicted mixed probability remains much higher than actual prevalence.

Original training pool:313mixed/2247nonmixed. Balanced fitting draws1280+1280 with replacement:1284unique groups, 49.8438% repeated draws. Train evaluation uses all original2560 groups. Selected313/512 mixed is outcome-assisted periodic selection, not probe classification accuracy.

| Changing balanced-batch window | Trajectory BCE | Mixed BCE | Weighted total |
|---|---:|---:|---:|
| first10 | 0.892328 | 0.949890 | 5.641780 |
| last10 | 0.703271 | 0.671215 | 4.059344 |

These are global-question-weighted means over changing batches, not fixed-data loss. All50 scalar steps are retained in the JSON. Probe train loop 7.331887s is not whole-outer time; the original recorded training-wall phase is1310.962974s, with QA/refit closure separately recorded.

No weights, hidden, full trajectories or predictions were copied to Mac; no new forward/rollout/GPU test ran. Actual complete151 grounding150 reward use was not yet available at this read. Both200/final51713 and native/resource closure remain incomplete.

# B110: original paired probe metrics and actual111 adoption

Original actor109 produced this periodic2560-questionG5 pool and fresh fixed512-questionG5 diagnostic set. Both use actual40-token actor hidden (head313); qid and normalized-question overlap are zero. The same trajectories and labels are used before/after updating. These are probe classification/loss metrics, not QA EM/F1.

Raw Adam uses LR5e-6, trajectoryBCE + 5 × mixedBCE, EMA0.99. All four original rank journals recorded50 real writes,1490→1540; each actual write updated EMA. Actual unlabelled selection111 used the bound EMA110 payload and actor110. No model/hidden tensor, complete rollout or full predictions were brought to Mac; no forward, training mutation or new test occurred.

| Head / original distribution | Before balanced accuracy | After balanced accuracy | Before trajectory BCE | After trajectory BCE | Before mixed BCE | After mixed BCE |
|---|---:|---:|---:|---:|---:|---:|
| raw / train | 55.0190% | 72.1568% | 0.698920 | 0.518170 | 0.879745 | 0.800848 |
| raw / test | 55.4239% | 53.4657% | 1.029593 | 0.822606 | 1.025536 | 1.337526 |
| ema / train | 54.7668% | 59.1168% | 0.526068 | 0.520562 | 0.942712 | 0.914624 |
| ema / test | 53.9330% | 53.5826% | 0.800897 | 0.803251 | 1.321763 | 1.334666 |

Training-side raw and EMA balanced accuracy improve, while heldout balanced accuracy decreases in this same-data paired comparison. EMA heldout predicts mixed too often:84/512 are actually mixed (16.40625%), but mean predicted q is68.8613%; post-update mixed recall83.3333%, nonmixed recall23.8318%, FP326/TN102/TP70/FN14. Ordinary accuracy33.59375% is below the always-nonmixed majority baseline83.59375%. This is weak generalization, not evidence of quality success.

The original training pool has322mixed/2238nonmixed. Fitting uses1280+1280 replacement sampling,1271unique groups and50.3515625% repeated draws. Training-side post-update evaluation uses the original2560 groups, not the resampled sequence. Final selected322/512 mixed is outcome-assisted periodic selection, not classifier accuracy.

| Actual changing-batch loss window | Trajectory BCE | Mixed BCE | Weighted total |
|---|---:|---:|---:|
| first10 | 0.794552 | 0.908051 | 5.334807 |
| last10 | 0.610411 | 0.630991 | 3.765363 |

Windows are weighted by actual global question counts and involve different batches; they are not a fixed-data loss curve. The complete50-step record is in the JSON. This periodic round performs exactly50 updates. Its probe update loop is 7.361962seconds, which is not whole-outer runtime.

Source probe payload SHA is producer metadata only, not an independent tensor hash/reload. Both200-outer goals, later150/200 QA/final51713 and actual exit/resource closure remain incomplete. Healthy training continues unchanged.

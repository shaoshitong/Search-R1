# B140: original paired probe metrics and actual141 EMA adoption

Original actor139 generated the2560-questionG5 pool and fresh fixed512-questionG5 diagnostic using actual40-token actor hidden, head313. Qid and normalized-question overlaps are zero. Before/after compare exactly the same trajectories and labels. Different periodic sets use changing actors/labels; this is not a fixed-label longitudinal curve. Classification metrics are not QA EM/F1.

LR5e-6, EMA.99, trajectoryBCE + 5 × mixedBCE remain unchanged. All four original journals record50 real writes,1910→1960. Actual unlabelled selection141 uses EMA140 with actor140. This is original receipt adoption evidence, not independent tensor preservation or GPU reload.

| Head / original distribution | Before BA | After BA | Before trajectory BCE | After trajectory BCE | Before mixed BCE | After mixed BCE |
|---|---:|---:|---:|---:|---:|---:|
| raw / train | 49.9067% | 64.0605% | 1.172229 | 0.544552 | 0.814978 | 0.976477 |
| raw / test | 57.0747% | 51.0003% | 1.507860 | 0.832514 | 0.646513 | 1.197407 |
| ema / train | 51.2978% | 54.7562% | 0.701291 | 0.600774 | 0.974101 | 0.972794 |
| ema / test | 57.4921% | 59.0951% | 0.960228 | 0.853586 | 1.112747 | 1.259683 |

Raw train BA improves, while raw heldout BA falls57.0747%→51.0003%. EMA heldout BA improves57.4921%→59.0951%, but its ordinary accuracy falls43.5547%→37.6953% and heldout mixed BCE worsens1.112747→1.259683. Both heldout heads improve trajectory BCE but worsen mixed BCE. This is not uniform quality improvement, and no loss/threshold/sampling change was made.

Post-update EMA heldout:68/512 actually mixed (13.2812%), mean predicted q 65.8535%; ordinary accuracy 37.6953% versus majority 86.7188%. Mixed recall 88.2353%, nonmixed recall 29.9550%; TP60/FN8/TN133/FP311. It continues to overpredict mixed.

Original train pool:293mixed/2267nonmixed. Fitting draws1280+1280 with replacement: 1287unique groups, 49.7266% repeated draws. Post-update train metrics use all original2560 groups, not balanced repeated rows. Selected293/512 mixed is outcome-assisted periodic selection, not classifier accuracy.

| Changing balanced-batch window | Trajectory BCE | Mixed BCE | Weighted total |
|---|---:|---:|---:|
| first10 | 1.060276 | 1.102948 | 6.575013 |
| last10 | 0.762809 | 0.768141 | 4.603515 |

These are global-question-weighted means over changing balanced batches, not fixed-data evaluation. Full50-step scalar records are in the JSON. The actual train loop takes 8.870684s, not whole-outer time.

No model, hidden, full trajectory or prediction tensors were copied to Mac; no new forward/rollout/GPU test ran. Frozen original training runtimes remain unchanged. Both200, B150/both200 QA/final51713 and native/resource closure remain incomplete.

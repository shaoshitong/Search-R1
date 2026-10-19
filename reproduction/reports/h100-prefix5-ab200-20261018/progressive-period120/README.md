# B120: original paired probe metrics and actual121 adoption

The original actor119 produced the periodic2560-questionG5 pool and fresh fixed512-questionG5 diagnostic set using actual40-token actor hidden (head313). Qid and normalized-question overlap are zero. Before/after metrics use the same original trajectories and labels. These classification metrics are not QA EM/F1, and changing actor snapshots make different periodic sets a changing-label series.

LR5e-6, EMA0.99, trajectoryBCE + 5 × mixedBCE are unchanged. All four actual original probe journals recorded50 writes,1630→1680. The actual unlabelled selection121 used the corresponding EMA120 payload with actor120. This is metadata adoption evidence, not an independent tensor preservation audit.

| Head / original distribution | Before BA | After BA | Before trajectory BCE | After trajectory BCE | Before mixed BCE | After mixed BCE |
|---|---:|---:|---:|---:|---:|---:|
| raw / train | 53.9209% | 68.6406% | 1.190355 | 0.554335 | 0.723603 | 0.923795 |
| raw / test | 55.9762% | 55.7889% | 1.569123 | 0.833540 | 0.590818 | 1.403713 |
| ema / train | 55.5617% | 57.6561% | 0.780214 | 0.679505 | 0.908367 | 0.922808 |
| ema / test | 57.5264% | 57.9312% | 1.096361 | 0.974866 | 0.950852 | 1.128209 |

Raw and EMA train balanced accuracy improve. EMA heldout BA rises slightly (57.5264%→57.9312%), raw heldout BA declines slightly (55.9762%→55.7889%); mixed BCE worsens for both heads on both original distributions. The result is not a uniform improvement or quality-success claim.

Post-update EMA heldout: 63/512 actually mixed (12.3047%), mean predicted q 59.6893%; ordinary accuracy 42.9688% versus majority 87.6953%. Mixed recall 77.7778%, nonmixed recall 38.0846%, TP49/FN14/TN171/FP278. The classification remains biased toward predicting mixed.

Original training pool:284mixed/2276nonmixed. Fitting uses1280+1280 replacement draws, 1274 unique groups and 50.2344% repeated draws. Post-update train evaluation is over the original2560 groups. Periodic selected284/512 mixed is outcome-assisted selection, not classifier accuracy.

| Actual changing-batch window | Trajectory BCE | Mixed BCE | Weighted total |
|---|---:|---:|---:|
| first10 | 0.934442 | 0.991772 | 5.893301 |
| last10 | 0.721009 | 0.662722 | 4.034619 |

These window means are weighted by global question counts over changing balanced batches, not a fixed-data loss curve. Full50-step records are in the JSON. The update loop takes 7.298056s; this is not whole-outer runtime.

Only code and small scalar metadata are local. No model/hidden/complete trajectories/full predictions were downloaded and no forward, baseline or new training job was run. Current original jobs continue unchanged. Both200, future150/200 QA/final51713 and actual terminal/resource closure remain incomplete.

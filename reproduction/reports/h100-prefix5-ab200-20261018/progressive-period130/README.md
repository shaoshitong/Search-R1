# B130: original paired probe metrics and actual131 EMA adoption

Actual actor129 produced the original2560G5 pool and fresh fixed512G5 diagnostic using40-token current-actor hidden, head313. Qid and normalized-question overlaps are zero. Before/after use the same trajectories/labels. Later periodic sets use changing actor snapshots and labels; they are not a fixed-label learning curve. These classification metrics are not QA EM/F1.

LR5e-6, EMA.99, trajectoryBCE + 5 × mixedBCE are unchanged. Each of four original journals records50 writes,1770→1820. Actual unlabelled selection131 uses the corresponding EMA130 payload with actor130. This is original receipt adoption evidence, not independent tensor preservation or GPU reloading.

| Head / original distribution | Before BA | After BA | Before trajectory BCE | After trajectory BCE | Before mixed BCE | After mixed BCE |
|---|---:|---:|---:|---:|---:|---:|
| raw / train | 54.0505% | 70.5954% | 0.783864 | 0.531329 | 0.987642 | 0.834491 |
| raw / test | 53.0405% | 50.6547% | 1.112857 | 0.822079 | 0.898398 | 1.323644 |
| ema / train | 55.6971% | 58.6901% | 0.549374 | 0.543769 | 0.997659 | 0.966778 |
| ema / test | 49.4555% | 49.7716% | 0.814743 | 0.814397 | 1.354694 | 1.347641 |

Raw train BA improves substantially, but raw heldout BA falls (53.0405%→50.6547%). EMA heldout BA barely changes (49.4555%→49.7716%). There is no verified heldout improvement beyond chance-level balanced discrimination at this point. No threshold/loss/sampling/training change was made.

Post-update EMA heldout:71/512 actually mixed (13.8672%), mean predicted q 67.2433%; ordinary accuracy 32.8125% versus majority 86.1328%. Mixed recall 73.2394%, nonmixed recall 26.3039%; TP52/FN19/TN116/FP325. The probe still overpredicts mixed.

Original train pool:342mixed/2218nonmixed; fitting uses1280+1280 replacement draws, 1305unique groups, 49.0234% repeated draws. Train evaluation uses all original2560groups, not balanced repeated rows. Selected342/512 mixed is outcome-assisted periodic selection, not classifier accuracy.

| Changing balanced-batch window | Trajectory BCE | Mixed BCE | Weighted total |
|---|---:|---:|---:|
| first10 | 0.808083 | 0.926073 | 5.438450 |
| last10 | 0.670591 | 0.687687 | 4.109026 |

Window means are global-question weighted over changing batches, not fixed-data loss evaluation. Full50-step scalar records are in the JSON. The actual train loop takes 7.525941s, not whole-outer time.

No weights/hidden/full trajectories/full predictions were downloaded. No new model forward, rollout or GPU test ran. Original A/B leaders continue unchanged; both200, B150/both200 QA/final51713 and actual terminal/resource closure remain incomplete.

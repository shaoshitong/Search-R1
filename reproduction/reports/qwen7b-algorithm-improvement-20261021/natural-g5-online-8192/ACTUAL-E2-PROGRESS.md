# Actual natural G5 control through L21 epoch 2

Original jobs remain 749459 (L21/P20) and 749460 (L27/P20). No new job submission or scheduling mutation was made by this observer. Both are two GPU jobs. L21 completed 128 real optimizer updates and began epoch 3; L27 started and is loading. This is partial evidence, not final acceptance.

The actual paired evaluator recomputed saved raw and EMA logits on the same fixed training 2048 and previously viewed independent 2048 question/G5 records. Initial raw and EMA logits match exactly between natural and conditional sampling. Final 4096 data was not read.

| L21 epoch 2 independent 2048 | Mixed BCE | Balanced accuracy | NonMixed false-positive rate | Real Mixed in top 512 |
| --- | ---: | ---: | ---: | ---: |
| Conditional G5 raw | 0.73528 | 55.52% | 70.11% | 210 (41.02%) |
| Natural G5 raw | 0.70388 | 56.14% | 57.55% | 193 (37.70%) |
| Conditional G5 EMA | 0.70553 | 55.12% | 67.29% | 201 (39.26%) |
| Natural G5 EMA | 0.71041 | 53.66% | 72.78% | 188 (36.72%) |

Natural G5 improves raw thresholded balanced accuracy and reduces false positives at this epoch, but has not improved top-512 ranking. Continue the originally submitted three epochs and perform actual checkpoint audits after clean Slurm exit before concluding. Exact counts, source references and paired deltas are in `actual-matched-metric-summary-E2.json`.

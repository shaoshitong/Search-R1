# B190 same-rollout paired probe results

Sampling actor189, actual actor hidden prefix80/head353. All original2560 training groups and fresh fixed512 heldout groups were evaluated before/after against the same per-trajectory strictEM labels. Qid and normalized-question overlap are0. These are classification metrics, not QA EM/F1.

Four ranks each completed50 actual raw/EMA writes2610→2660 at LR5e-6, EMA.99, trajectoryBCE+5mixedBCE. Actual191 scoring used this EMA with actor190; producer metadata and actual-score receipts are not independent tensor readback.

| Head / original distribution | Before BA | After BA | Before trajectory BCE | After trajectory BCE | Before mixed BCE | After mixed BCE |
|---|---:|---:|---:|---:|---:|---:|
| raw / train | 52.0125% | 60.3112% | 0.992849 | 0.646980 | 0.819356 | 0.976126 |
| raw / test | 48.6237% | 49.5581% | 1.300034 | 0.842807 | 0.697725 | 1.347263 |
| ema / train | 53.1567% | 55.5096% | 0.649184 | 0.609036 | 1.089698 | 1.095884 |
| ema / test | 49.7601% | 49.6338% | 0.889908 | 0.831810 | 1.106269 | 1.233362 |

EMA heldout BA remains near chance (49.6338%). Its ordinary accuracy is33.3984%, versus majority85.9375%; actual mixed72/512=14.0625% versus mean predicted q64.2520%. TP52/FN20/TN119/FP321. Both heads improve trajectory BCE while worsening heldout mixed BCE. Do not interpret in-sample improvement or periodic outcome-assisted selection as generalization success.

Training original330mixed/2230nonmixed; fitting resamples1280+1280,1283unique groups,49.8828% repeated draws. Train post-update metrics use all original groups, not those repeated fitting draws. Periodic selected330/512 mixed is outcome-assisted selection, not classifier accuracy.

| Changing balanced-batch window | Trajectory BCE | Mixed BCE | Weighted total |
|---|---:|---:|---:|
| first10 | 0.909998 | 0.998719 | 5.903591 |
| last10 | 0.680706 | 0.738656 | 4.373987 |

The50 changing-batch steps are retained in JSON. No new generation, GPU test, tensor load or active-runtime edit was performed. Labels and actor snapshots differ between periods; these are not fixed-label learning curves. All200/final51713/terminal closure remain pending.

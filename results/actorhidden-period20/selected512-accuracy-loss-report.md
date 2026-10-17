# Selected512 classification and probe fitting curves

Snapshot: completed outers 1–92, run `a008d7ce983347bbb6e81051b25a7c4e`.

This report uses selection-time probabilities, before actor/probe updates, joined by original trajectory IDs to eventual own strict-EM labels. It is not a post-update evaluation of selected512. Threshold remains 0.5; mixed means 0 < correct count < 5. All 92 completed rounds have all 512 selected probabilities >= 0.5 and both actual classes. Thus selected-subset balanced accuracy is exactly 0.5 (mixed recall 1, nonmixed recall 0). This statistic on the selected tail does not establish random ranking on the full candidate pool.

Latest outer92: TP136/FN0/TN0/FP376; ordinary accuracy and actual mixed fraction 136/512 = 0.265625; mean predicted probability 0.7567661295780292. Exact per-outer metrics are in `curves-through92/selected512-all-outers.csv`.

## Paired full-pool loss before → after one natural training epoch

| Outer | Train total BCE | Held-out total BCE | Held-out balanced accuracy |
|---|---|---|---|
| 20 | 1.3359 → 1.2313 | 1.4495 → 1.4967 | 54.60% → 58.80% |
| 40 | 1.5700 → 1.3183 | 1.6735 → 1.5598 | 54.59% → 51.20% |
| 60 | 1.5039 → 1.4575 | 1.5686 → 1.6440 | 52.10% → 52.75% |
| 80 | 1.6944 → 1.3910 | 1.8080 → 1.6468 | 51.94% → 54.10% |

All four same-data training totals decrease, but held-out totals increase at20 and60 and decrease at40 and80. The individual trajectory and group BCE components can move in opposite directions. No stable held-out improvement is established.

Only outers20/40/60/80 performed probe fitting, each100 synchronous Adam updates, LR1e-5. Online batch loss is before that batch update and uses different examples at each step: faint raw curves plus trailing10-step means are descriptive, not a fixed-set optimization curve. Paired whole-pool before/after evaluations are the appropriate direct fitting comparison. At different periods the actor snapshot and generated labels differ; they are not a fixed validation learning curve.

## Artifacts

- `curves-through92/selected512-accuracy.png` and PDF: all completed iterations.
- `curves-through92/probe-adam-loss-curves.png` and PDF: trajectory/group/total loss for all400 actual updates.
- `curves-through92/probe-paired-fit-evaluation.png` and PDF: same-data pre/post fulltrain and heldout evaluations.
- `selected-accuracy-all-complete.json`: small metric rows, period loss rows, source SHA references; no tensors or full trajectories.

CPU metadata only on the source node. No model forward, new rollout, training stop, code deployment, threshold change, or training runtime modification. Models/hidden states/full trajectories remain remote.

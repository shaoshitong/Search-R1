# Progressive B: actual periodic90

Same sampling actor0089 produced native question+20 prefixes and labels for the original2560-question training pool and fresh fixed512-question heldoutG5. Qid/content overlaps are zero. Before/after metrics use the same original trajectories; these are mixed-classification metrics, not QA EM/F1.

| Model / original distribution | Mixed BA before → after | Trajectory BCE before → after | Mixed BCE before → after |
|---|---:|---:|---:|
| raw / train | 50.4946% → 64.6451% | 0.776642 → 0.564694 | 1.002378 → 0.918775 |
| raw / test | 53.1636% → 54.0943% | 1.081401 → 0.833091 | 0.883669 → 1.201870 |
| ema / train | 53.8693% → 57.0549% | 0.561209 → 0.543562 | 1.032483 → 0.996304 |
| ema / test | 54.8996% → 51.8588% | 0.804784 → 0.776616 | 1.254616 → 1.314165 |

EMA heldout BA falls54.90%→51.86%, while original-train BA rises53.87%→57.05%. Its heldout mixedBCE worsens1.254616→1.314165 despite trajectoryBCE improving0.804784→0.776616. This update improves training fit without heldout improvement; raw heldout BA rises only53.16%→54.09%. It is not quality success.

Original train contains372mixed/2188nonmixed. Balanced resampling uses1280+1280 groups,1325 unique,48.2422% repetitions. Heldout contains86mixed/426nonmixed, majority83.2031%. Post-update EMA ordinary accuracy is33.7891%, mixed recall79.0698%, nonmixed recall24.6479%, with321 false positives. The threshold remains0.5. Different periodic actor versions and labels do not form a fixed-label learning curve.

| Actual changing-batch window | Trajectory BCE | Mixed BCE | Weighted total |
|---|---:|---:|---:|
| first10 | 0.817041 | 0.855537 | 5.094727 |
| last10 | 0.706844 | 0.722664 | 4.320164 |

First/last10 windows use different balanced mini-batches, weighted by actual global question counts; they are not repeated fixed-data evaluations. All four original rank journals show50 probe writes1210→1260, one EMA update per write; all four actor journals show20 writes1780→1800. LR5e-6, EMA.99 and trajectoryBCE+5×mixedBCE are unchanged. The50 probe updates use40 global255-trajectory and10 global260-trajectory steps.

Probe loop:7.306881s. Entire outer training wall:1237.464019s (20.62min), including rollout, actor and other phases. The probe loop is not the whole iteration.

Actual91 unlabelled EMA scoring adopted head90 payload `61c1b02d443a0b729daf712da3af515b02212e145fd19d82daea4ee6e2bdb895`, EMA1260 and actor0090. These are producer checkpoint identities plus actual score records, not independent model tensor hashing/deserialization.

Periodic90 selects372/512=72.65625% mixed by the existing actual-label-assisted full-pool rule; this is not classifier accuracy. Post50 ordinary B also uses expanded-prefix-admission continuation and true-mixed-first reranking, so selected-ratio differences cannot isolate EMA effects.

Frozen A/B training runtimes are unchanged. Actual100/150/200 normal-savepoint, QA and final51713 obligations remain pending for B; both complete200 objectives remain active.

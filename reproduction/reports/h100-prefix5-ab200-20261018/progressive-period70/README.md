# Progressive B, actual periodic70

Frozen training runtime remains `c5f1a07e6a9f4ccee35f45ddf69ab4aa3d28e910`. This result/control publication is not a runtime deployment.

The original sampling actor0069 supplied the native question+20 hidden and the complete G5 for all2560 original training questions and the fresh fixed512-question diagnostic. Actual qid/content intersections are both0. The same original data and labels are used before/after the50 probe updates. This is prefix classification, not QA EM/F1.

| Model / original distribution | Mixed balanced accuracy before | After | Trajectory BCE before → after | Mixed BCE before → after |
| --- | ---: | ---: | ---: | ---: |
| raw / train | 49.5981% | 70.9037% | 0.972677 → 0.483441 | 0.946173 → 0.825310 |
| raw / test | 49.8828% | 54.1065% | 1.264842 → 0.767267 | 0.774152 → 1.324317 |
| ema / train | 54.7160% | 58.3593% | 0.543844 → 0.512911 | 1.039934 → 0.986429 |
| ema / test | 50.0524% | 51.9561% | 0.792687 → 0.764615 | 1.307633 → 1.345497 |

The EMA diagnostic balanced accuracy rises50.05%→51.96%, but its mixedBCE worsens1.307633→1.345497. Raw diagnostic mixedBCE also worsens0.774152→1.324317 despite a balanced-accuracy increase. Ordinary EMA diagnostic accuracy31.25% is below the85.55% majority baseline (74mixed/438nonmixed). Its mixed recall81.08% and nonmixed recall22.83% show substantial false-positive preference. These are weak heldout results, not quality success. Different periodic actors produce different G5 labels;60→70 is not a fixed-label learning curve.

Original train distribution is378mixed/2182nonmixed. The balanced current-round pass samples1280+1280 groups,1338 unique with47.7344% repetitions; no history or diagnostic data is trained. All four original rank journals confirm50 writes930→980 with EMA once/write, LR5e-6, EMA.99, trajectoryBCE+5×mixedBCE. The50 global-step loss points are retained in the JSON. Actual global batches are40×255 and10×260 trajectories.

| Global-step window, actual changing training batches | Trajectory BCE | Mixed BCE | Weighted total |
| --- | ---: | ---: | ---: |
| First10 | 0.877437 | 1.021115 | 5.983014 |
| Last10 | 0.595952 | 0.608745 | 3.639679 |

These windows are different balanced mini-batches, not repeated evaluations on one fixed batch. The paired full original train/heldout table above supplies the fixed-data before/after comparison.

The actual71 unlabelled predicted selection uses the completed70 EMA payload `54f946df22f1cdb6c3c370597dabc04dd0d77c990bc506ab090604c08a4889f4`, EMA980 and actor0070. This adoption was read from the original selection JSON, not inferred from process liveness.

The final selected378/512=73.8281% mixed is label-assisted: all378 actual mixed groups from the2560 full periodic pool are admitted, with134 nonmixed fallbacks. It is not probe accuracy. Ordinary post50 B rounds also rerank an expanded predicted pool; do not attribute the entire final mixing fraction to the probe or prefix.

Training-wall time1109.228830s≈18m29, probe phase55.649254s, actual50-update loop7.161282s. The7.16s loop is not the iteration runtime. Actor1400/probe980 per rank at70; following71 complete is actor1420/probe990. No independent70 tensor hash, CPU/GPU reload or backup was performed by this read-only report.

Raw small evidence SHA256: `88c75d6680bab26100a313863abf07aca3c9294c3eec20bdbb04fb95ebc639ce`. Query source SHA256: `18949c390bcdf9578f9b779b6b25f887eb85224e10b2f83c5d80b1f7da23bae9`. Both actual native trainers continue;200, future scheduled QA and final51713 remain incomplete.

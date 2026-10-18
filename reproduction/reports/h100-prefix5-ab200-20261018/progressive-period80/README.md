# Progressive B, actual periodic80

Sampling actor0079 produced the native question+20 prefixes, all2560 original trainingG5 and the fresh fixed512-question diagnosticG5. Actual qid and normalized-content overlap is0. The same original data and labels are compared before/after50 probe updates. These are prefix classification metrics, not QA EM/F1.

| Model / original distribution | Mixed BA before → after | Trajectory BCE before → after | Mixed BCE before → after |
|---|---:|---:|---:|
| raw / train | 50.6395% → 65.3922% | 1.062872 → 0.490587 | 0.851306 → 0.868476 |
| raw / test | 50.3043% → 52.6234% | 1.480679 → 0.819111 | 0.699536 → 1.218652 |
| ema / train | 54.8083% → 57.8249% | 0.624360 → 0.551262 | 0.922911 → 0.885158 |
| ema / test | 55.3698% → 54.9358% | 0.932363 → 0.850038 | 1.118873 → 1.232957 |

EMA heldout BA falls55.37%→54.94%, and its mixedBCE worsens1.118873→1.232957. Raw heldout BA rises modestly while mixedBCE worsens. This is not stable heldout improvement or quality success. Original-distribution metrics and losses on changing balanced training batches measure different distributions.

Original train has415mixed/2145nonmixed. The current-round balanced pass contains1280+1280 groups,1360 unique,46.875% repetitions. Heldout has87mixed/425nonmixed; majority accuracy is83.0078%. EMA post-update ordinary accuracy is38.0859%, mixed recall80.4598%, nonmixed recall29.4118%, showing a substantial false-positive preference. Different periodic actors/labels mean70→80 is not a fixed-label learning curve.

| Actual changing-batch window | Trajectory BCE | Mixed BCE | Weighted total |
|---|---:|---:|---:|
| first10 | 0.914787 | 1.025296 | 6.041265 |
| last10 | 0.636373 | 0.684062 | 4.056683 |

The first/last10 windows use different balanced mini-batches and are weighted by actual global question counts. Their downward trend is not a repeated evaluation on fixed data. Full original-distribution paired results are shown above.

All four original rank journals confirm50 writes1070→1120, one EMA step per write, LR5e-6, EMA.99, trajectoryBCE+5×mixedBCE. Actual global batches are40×255 and10×260 trajectories. The real50-update loop is6.942258s; the full80 training wall is1206.587188s≈20.11min, not6.94s.

Actual81 unlabelled selection used EMA80 payload `2253ecf032bf94ec8308d5bdde8644bff7ae06c039bd0442bee5ca045eb50c98`, EMA1120 and actor0080, as read from the original selection JSON. Source payload SHA is producer metadata only; this read did not perform independent tensor hashing, checkpoint deserialization or GPU forward.

Final selected mixing at80 is415/512=81.0547% because the periodic rule admits actual mixed groups from the full2560 pool; it is not probe classification accuracy. The following ordinary81 selected153/512=29.8828%. Post50 ordinary B also reranks an expanded pool, so the final ratio cannot isolate EMA/probe effects.

Frozen runtimes A2e397/Bc5f1 are unchanged; this is results publication only. Both200-outer goals, scheduled100/150/200 QA and final51713 remain incomplete.

Raw small evidence SHA256: `3e61644bddb4e6835a09ede2f0c99b5e6aec7383306fbd305106b04238a90a42`.

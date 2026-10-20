# B180 same-rollout paired probe results

Original actor179 generated2560 training questionsG5 and fresh fixed512 diagnostic questionsG5, using actual sampling-actor question+80 hidden and head353. Qid and normalized-question intersections are0. Before/after evaluations use exactly the same trajectories and own strict-EM labels. The training metrics evaluate all original2560 groups, not the repeated balanced draws. These metrics are group classification, not QA EM/F1.

Four original journals each record50 real probe/EMA updates2470→2520, LR5e-6/EMA.99, trajectoryBCE+5mixedBCE. Actual181 score used EMA180 with actor180; this is producer-receipt and actual-score binding, not a new tensor readback or forward test.

| Head / original distribution | Before BA | After BA | Before trajectory BCE | After trajectory BCE | Before mixed BCE | After mixed BCE |
|---|---:|---:|---:|---:|---:|---:|
| raw / train | 49.3228% | 65.1630% | 1.062285 | 0.548666 | 0.886512 | 1.000449 |
| raw / test | 49.4598% | 54.7255% | 1.293195 | 0.800301 | 0.816618 | 1.288740 |
| ema / train | 49.3903% | 55.2817% | 0.656357 | 0.583820 | 1.004375 | 1.028255 |
| ema / test | 53.8558% | 51.0260% | 0.852113 | 0.789308 | 1.294527 | 1.384468 |

Raw heldout BA improves, while EMA heldout BA declines. Both heads improve trajectory BCE but worsen heldout mixed BCE; training BA improvement therefore does not establish generalization improvement. The configured EMA selector, losses and threshold remain unchanged.

Post-update EMA heldout: actual mixed66/512=12.8906%, mean predicted q 69.6149%; ordinary accuracy29.2969% versus majority87.1094%. Mixed recall80.3030%, nonmixed recall21.7489%; TP53/FN13/TN97/FP349. Predicted mixing remains substantially above observed prevalence.

Original fitting pool:314mixed/2246nonmixed; balanced fitting draws1280+1280 with replacement, 1270 unique groups and50.3906% repeated draws. The selected314/512 mixed result is outcome-assisted periodic selection, not probe classification accuracy.

| Changing balanced-batch window | Trajectory BCE | Mixed BCE | Weighted total |
|---|---:|---:|---:|
| first10 | 0.923178 | 1.086160 | 6.353978 |
| last10 | 0.747286 | 0.779609 | 4.645333 |

The50 scalar steps are in JSON. Window means weight by actual global question count and use changing batches, not a fixed-data loss trace. Actual head train loop5.131186s is not whole-outer time; the recorded full training wall is977.873428s.

Only existing small metadata was read. No new rollout, model forward, tensor load or frozen-repo edit. Current actor snapshots and labels change between periods, so cross-period values are not a fixed-label learning curve. B200, final51713 and native/resource closure remain incomplete.

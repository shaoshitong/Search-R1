# Matched full169028 L15 Prefix80 MSE — both first epochs

Both actual jobs completed1321 first-epoch optimizer updates and the fixed diagnostics, then continued into the next epoch. This is a fixed-epoch descriptive comparison, not the family checkpoint choice or an untouched-data result.

| Seed | State | Train B Mixed BA | Train B Mixed@512 | Development Mixed BA | Development Mixed@512 | Development variance MSE |
|---|---|---:|---:|---:|---:|---:|
| seed1 | raw | 67.3295% | 359/512 (70.1172%) | 66.5479% | 284/512 (55.4688%) | 0.00745031 |
| seed1 | EMA | 56.2982% | 307/512 (59.9609%) | 56.3624% | 248/512 (48.4375%) | 0.00834641 |
| seed2 | raw | 66.6663% | 367/512 (71.6797%) | 66.0406% | 286/512 (55.8594%) | 0.00714388 |
| seed2 | EMA | 62.2655% | 316/512 (61.7188%) | 60.5228% | 249/512 (48.6328%) | 0.00850443 |

Original E1 EMA on the same viewed development pool:205/512(40.0391%),Mixed BA52.8846%. Both full-data raw first epochs improve on this point estimate; this is not final generalization confirmation. Same budget/geometry/pool are used, with independent explicit initialization and training-view seeds. EMA remains a separate state.

At the fixed0.08 population-variance threshold on development, seed1 raw predicts420/663 trueMixed and falsely predicts81/330 all-correct plus338/1055 all-wrong asMixed; seed2 raw predicts352/663 trueMixed, with32/330 all-correct and259/1055 all-wrong false positives. Similar Top512 yields do not imply identical threshold calibration.

All10 epochs remain required. The predeclared seed1-only maximumTop512 rule selects one epoch/state, then applies that exact choice to seed2. Actual selected checkpoints still need CPU acceptance, trajectory-order audit and legacy E1 known32 adapter replay before final candidate freeze and the blind4096 confirmation. No checkpoints or final inputs were opened for this report.

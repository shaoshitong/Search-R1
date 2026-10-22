# Full169028 L15 Prefix80 MSE — first two completed epochs, seed2

These are existing fixed trainB and repeatedly viewed development results. This seed never chooses the family epoch/state. Both jobs must finish10 epochs; the published rule selects from seed1 only. The same choice is then applied to seed2 before any blind4096 evaluation.

| Epoch | State | TrainB Mixed BA | TrainB Mixed@512 | Dev Mixed BA | Dev Mixed@512 | Dev variance MSE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | raw | 66.6663% | 367/512 (71.6797%) | 66.0406% | 286/512 (55.8594%) | 0.00714388 |
| 1 | EMA | 62.2655% | 316/512 (61.7188%) | 60.5228% | 249/512 (48.6328%) | 0.00850443 |
| 2 | raw | 71.5626% | 383/512 (74.8047%) | 68.8872% | 310/512 (60.5469%) | 0.00685546 |
| 2 | EMA | 63.3182% | 358/512 (69.9219%) | 61.8333% | 281/512 (54.8828%) | 0.00757876 |

For raw E1→E2 on development: true Mixed predictions352→429 of663, all-correct false positives32→29 of330, all-wrong false positives259→344 of1055. Recall rises53.09%→64.71%, while nonMixed FPR rises21.01%→26.93%. BA and Top512 improve, but error components do not all improve. Variance score is not a Mixed probability; the fixed population-variance threshold remains0.08.

Original E1 baseline: development BA52.8846%,Top512205/512(40.0391%). Its old trainA score must not be substituted for trainB. Unified E1 trainB supplement is staged and waits for complete Zseed2 acceptance. No CP tensor or final data read by this report.

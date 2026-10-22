# Full169028 L15/P80 MSE: first two completed epochs, both seeds

Same fixed trainB and repeatedly viewed development2048 ownG5. All epoch/state results below are descriptive; the published seed1-only selection rule is unchanged. Final4096 remains blind.

| Seed | Epoch | State | TrainB BA | TrainB Mixed@512 | Dev BA | Dev Mixed@512 | Dev variance MSE |
|---|---:|---|---:|---:|---:|---:|---:|
| seed1 | 1 | raw | 67.3295% | 359/512 (70.1172%) | 66.5479% | 284/512 (55.4688%) | 0.00745031 |
| seed1 | 1 | EMA | 56.2982% | 307/512 (59.9609%) | 56.3624% | 248/512 (48.4375%) | 0.00834641 |
| seed1 | 2 | raw | 68.5710% | 381/512 (74.4141%) | 67.1582% | 310/512 (60.5469%) | 0.00687763 |
| seed1 | 2 | EMA | 62.6522% | 353/512 (68.9453%) | 61.9312% | 274/512 (53.5156%) | 0.00764121 |
| seed2 | 1 | raw | 66.6663% | 367/512 (71.6797%) | 66.0406% | 286/512 (55.8594%) | 0.00714388 |
| seed2 | 1 | EMA | 62.2655% | 316/512 (61.7188%) | 60.5228% | 249/512 (48.6328%) | 0.00850443 |
| seed2 | 2 | raw | 71.5626% | 383/512 (74.8047%) | 68.8872% | 310/512 (60.5469%) | 0.00685546 |
| seed2 | 2 | EMA | 63.3182% | 358/512 (69.9219%) | 61.8333% | 281/512 (54.8828%) | 0.00757876 |

At E2/raw, both independently initialized/training-plan seeds select310/512 trueMixed questions (60.5469%). Their thresholded BA differs (67.16% and68.89%), so identical Top512 counts do not mean identical scores, selected question identities or calibration. Raw/EMA are both reported. Original E1 EMA development baseline is205/512 (40.0391%), BA52.8846%; these are development comparisons, not unknown-question confirmation.

Seed1 completed E2 at2642 actual Adam/EMA updates and is genuinely in E3; seed2 is genuinely in E4. No training change, extra GPU work or final-data read was used to summarize these existing evaluations.

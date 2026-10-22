# Full169028 L15 Prefix80 MSE — seed2 first three epochs

Fixed trainB and repeatedly viewed development; no final inference or candidate epoch choice. Both states are reported. The published seed1-only rule still decides epoch/state only after the full10-epoch runs complete.

| Epoch | State | TrainB BA | TrainB Mixed@512 | Dev BA | Dev Mixed@512 | Dev variance MSE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | raw | 66.6663% | 367/512 (71.6797%) | 66.0406% | 286/512 (55.8594%) | 0.00714388 |
| 1 | EMA | 62.2655% | 316/512 (61.7188%) | 60.5228% | 249/512 (48.6328%) | 0.00850443 |
| 2 | raw | 71.5626% | 383/512 (74.8047%) | 68.8872% | 310/512 (60.5469%) | 0.00685546 |
| 2 | EMA | 63.3182% | 358/512 (69.9219%) | 61.8333% | 281/512 (54.8828%) | 0.00757876 |
| 3 | raw | 73.2149% | 402/512 (78.5156%) | 70.2279% | 317/512 (61.9141%) | 0.00676982 |
| 3 | EMA | 68.3597% | 380/512 (74.2188%) | 66.4080% | 304/512 (59.3750%) | 0.00696118 |

Raw development threshold errors (fixed population variance0.08; not a Mixed probability):

| Epoch | True Mixed recalled /663 | All-correct false Mixed /330 | All-wrong false Mixed /1055 |
|---:|---:|---:|---:|
| 1 | 352 | 32 | 259 |
| 2 | 429 | 29 | 344 |
| 3 | 453 | 25 | 361 |

E3 training/evaluation completed3963 actual optimizer and EMA updates; both ranks subsequently progressed into E4. No claim that the E3 checkpoint is selected or independently CPU accepted yet. Original E1 EMA development baseline205/512(40.0391%),BA52.8846%. The untouched4096 is still blind.

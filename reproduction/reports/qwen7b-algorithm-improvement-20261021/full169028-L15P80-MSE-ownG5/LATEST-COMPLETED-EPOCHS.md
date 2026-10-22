# Full L15/P80 MSE: all completed epoch diagnostics

Same fixed trainB2048 and viewed development2048 ownG5. Both raw/EMA and all completed epochs reported. These are development results; final4096 is still blind.

| Seed | Epoch | State | TrainB BA | TrainB Mixed@512 | Dev BA | Dev Mixed@512 | Dev variance MSE |
|---|---:|---|---:|---:|---:|---:|---:|
| seed1 | 1 | raw | 67.33% | 359/512 (70.12%) | 66.55% | 284/512 (55.47%) | 0.00745031 |
| seed1 | 1 | EMA | 56.30% | 307/512 (59.96%) | 56.36% | 248/512 (48.44%) | 0.00834641 |
| seed1 | 2 | raw | 68.57% | 381/512 (74.41%) | 67.16% | 310/512 (60.55%) | 0.00687763 |
| seed1 | 2 | EMA | 62.65% | 353/512 (68.95%) | 61.93% | 274/512 (53.52%) | 0.00764121 |
| seed1 | 3 | raw | 71.42% | 389/512 (75.98%) | 69.09% | 318/512 (62.11%) | 0.00681917 |
| seed1 | 3 | EMA | 67.63% | 373/512 (72.85%) | 66.07% | 298/512 (58.20%) | 0.00705294 |
| seed2 | 1 | raw | 66.67% | 367/512 (71.68%) | 66.04% | 286/512 (55.86%) | 0.00714388 |
| seed2 | 1 | EMA | 62.27% | 316/512 (61.72%) | 60.52% | 249/512 (48.63%) | 0.00850443 |
| seed2 | 2 | raw | 71.56% | 383/512 (74.80%) | 68.89% | 310/512 (60.55%) | 0.00685546 |
| seed2 | 2 | EMA | 63.32% | 358/512 (69.92%) | 61.83% | 281/512 (54.88%) | 0.00757876 |
| seed2 | 3 | raw | 73.21% | 402/512 (78.52%) | 70.23% | 317/512 (61.91%) | 0.00676982 |
| seed2 | 3 | EMA | 68.36% | 380/512 (74.22%) | 66.41% | 304/512 (59.38%) | 0.00696118 |
| seed2 | 4 | raw | 73.16% | 410/512 (80.08%) | 70.31% | 318/512 (62.11%) | 0.00662176 |
| seed2 | 4 | EMA | 71.43% | 396/512 (77.34%) | 68.42% | 317/512 (61.91%) | 0.00675485 |
| seed2 | 5 | raw | 71.46% | 408/512 (79.69%) | 67.99% | 320/512 (62.50%) | 0.00666226 |
| seed2 | 5 | EMA | 72.78% | 405/512 (79.10%) | 69.37% | 319/512 (62.30%) | 0.00662849 |

Seed2 E4→E5/raw: Top512318→320, BA70.3088%→67.9876%, Mixed recall63.6501%→51.2821%, nonMixed FPR23.0325%→15.3069%. Thresholded predictions became more conservative while the ranking count slightly improved. The primary selection rule remains seed1 development Top512 over all20 completed epoch/state choices, then the same epoch/state on seed2. No early selection or optimization change uses seed2.

Population-variance diagnostic threshold0.08 corresponds to raw4v score0.32 and is not a calibrated Mixed probability. Full10E/13210 budget per seed, selected CP acceptance, order audit, actual freeze and independent final confirmation remain required.

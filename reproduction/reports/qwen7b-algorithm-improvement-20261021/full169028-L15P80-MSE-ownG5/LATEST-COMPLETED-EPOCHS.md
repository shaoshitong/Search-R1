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
| seed1 | 4 | raw | 72.15% | 394/512 (76.95%) | 70.19% | 312/512 (60.94%) | 0.00679006 |
| seed1 | 4 | EMA | 70.39% | 389/512 (75.98%) | 68.18% | 313/512 (61.13%) | 0.00680714 |
| seed1 | 5 | raw | 70.19% | 402/512 (78.52%) | 67.79% | 314/512 (61.33%) | 0.00667131 |
| seed1 | 5 | EMA | 71.60% | 397/512 (77.54%) | 69.28% | 312/512 (60.94%) | 0.00669189 |
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
| seed2 | 6 | raw | 74.64% | 408/512 (79.69%) | 70.95% | 325/512 (63.48%) | 0.00658111 |
| seed2 | 6 | EMA | 73.78% | 410/512 (80.08%) | 70.53% | 325/512 (63.48%) | 0.00654615 |
| seed2 | 7 | raw | 75.09% | 418/512 (81.64%) | 70.90% | 318/512 (62.11%) | 0.00657380 |
| seed2 | 7 | EMA | 74.39% | 413/512 (80.66%) | 71.05% | 328/512 (64.06%) | 0.00649732 |

Seed1 E3→E4/raw: Top512 **318→312**, while BA **69.0950%→70.1886%**. The saved Top512 composition check reports all-wrong191→197 (net+6), Mixed318→312 (net−6), all-correct3→3; these are net class-count changes, not six identified question swaps. Ranking and the fixed-threshold BA need not move together. No epoch/state is selected from this partial-run change.

Seed2 E4→E5/raw: Top512318→320, BA70.3088%→67.9876%, Mixed recall63.6501%→51.2821%, nonMixed FPR23.0325%→15.3069%. Thresholded predictions became more conservative while the ranking count slightly improved. The primary selection rule remains seed1 development Top512 over all20 completed epoch/state choices, then the same epoch/state on seed2. No early selection or optimization change uses seed2.

Seed2 E6/raw and EMA both select325/512 (63.4765625%) Mixed. Their fixed-dev BA values are70.9539% and70.5254%. These are already completed development diagnostics; no seed2-based epoch/state selection is made. Local live snapshot `1792708398.986121`.

Population-variance diagnostic threshold0.08 corresponds to raw4v score0.32 and is not a calibrated Mixed probability. Full10E/13210 budget per seed, selected CP acceptance, order audit, actual freeze and independent final confirmation remain required.

## Seed1 E4: complete fixed trainB/development diagnostics

Copied from local `actual-live.json`, snapshot `1792707990.0928564`, epoch4/cursor5284. Both pools contain2048 fixed ownG5 groups. TrainB has888 Mixed/950 all-wrong/210 all-correct (43.359375% Mixed); viewed dev has663/1055/330 (32.373046875% Mixed). The objective is MSE on population variance; all saved mean scores below are4v, threshold0.32 (=v0.08), never fitted.

| Pool | State | BA | Mixed recall | nonMixed FPR | all-wrong FPR | all-correct FPR | Mixed@512 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TrainB | raw | 72.152066% | 68.355856% | 24.051724% | 28.315789% | 4.761905% | 394/512 (76.953125%) | 0.807832887 |
| TrainB | EMA | 70.392203% | 60.698198% | 19.913793% | 23.263158% | 4.761905% | 389/512 (75.976562%) | 0.792711731 |
| Viewed dev | raw | 70.188564% | 68.174962% | 27.797834% | 34.691943% | 5.757576% | 312/512 (60.937500%) | 0.784244028 |
| Viewed dev | EMA | 68.177140% | 60.180995% | 23.826715% | 28.720379% | 8.181818% | 313/512 (61.132812%) | 0.771347284 |

| Pool | State | Variance MSE(v) | Target4variance MSE | Variance MAE(v) | Variance Spearman |
|---|---|---:|---:|---:|---:|
| TrainB | raw | 0.0065113148 | 0.1041810360 | 0.0642594023 | 0.5238045774 |
| TrainB | EMA | 0.0068991914 | 0.1103870623 | 0.0694178609 | 0.5052556779 |
| Viewed dev | raw | 0.0067900645 | 0.1086410322 | 0.0655387409 | 0.4636316966 |
| Viewed dev | EMA | 0.0068071363 | 0.1089141804 | 0.0684271777 | 0.4474582979 |

| Pool | State | TP | FN | FP | TN | Mean4v: Mixed | Mean4v: all-wrong | Mean4v: all-correct | Predicted Mixed: Mixed/AW/AC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainB | raw | 607 | 281 | 279 | 881 | 0.4137928386 | 0.2253443700 | 0.0910411215 | 607 / 269 / 10 |
| TrainB | EMA | 539 | 349 | 231 | 929 | 0.3632223563 | 0.2194801946 | 0.1163413344 | 539 / 221 / 10 |
| Viewed dev | raw | 452 | 211 | 385 | 1000 | 0.4118943362 | 0.2522608355 | 0.0950272345 | 452 / 366 / 19 |
| Viewed dev | EMA | 399 | 264 | 330 | 1055 | 0.3658838251 | 0.2402586589 | 0.1194097771 | 399 / 303 / 27 |

All values are copied from already completed fixed diagnostics, not online training batches. Existing E1–E5 seed1 and E1–E7 seed2 rows above are preserved. Final4096 remains blind and the seed1-only full-budget checkpoint selection rule is unchanged.

## Seed1 E5: fixed trainB/development diagnostics

Local snapshot `1792709138.225533`; epoch5/cursor6605. Same ownG5 pools, variance threshold0.08 (4v score0.32), no fitted threshold.

| Pool | State | BA | Mixed recall | nonMixed FPR | all-wrong FPR | all-correct FPR | Mixed@512 | AUC | Variance MSE(v) | Variance MAE(v) | Variance Spearman |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainB | raw | 70.189500% | 55.292793% | 14.913793% | 17.263158% | 4.285714% | 402/512 (78.515625%) | 0.8200557238 | 0.0068473475 | 0.0648161460 | 0.5309417142 |
| TrainB | EMA | 71.601041% | 63.288288% | 20.086207% | 23.368421% | 5.238095% | 397/512 (77.539062%) | 0.8068499534 | 0.0066607067 | 0.0662652559 | 0.5230842712 |
| Viewed dev | raw | 67.787053% | 52.036199% | 16.462094% | 20.568720% | 3.333333% | 314/512 (61.328125%) | 0.7909082989 | 0.0066713075 | 0.0631782686 | 0.4685346760 |
| Viewed dev | EMA | 69.275474% | 62.594268% | 24.043321% | 29.668246% | 6.060606% | 312/512 (60.937500%) | 0.7823779887 | 0.0066918926 | 0.0660570021 | 0.4619808844 |

These are completed fixed diagnostics, not online batch metrics. All earlier completed rows remain above. No epoch/state selection or final4096 access.

## Seed2 E7: fixed trainB/development diagnostics

Local snapshot `1792709474.336931`; epoch7/cursor9247. Same ownG5 pools, variance threshold0.08 (4v score0.32), no fitted threshold.

| Pool | State | BA | Mixed recall | nonMixed FPR | all-wrong FPR | all-correct FPR | Mixed@512 | AUC | Variance MSE(v) | Variance MAE(v) | Variance Spearman |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainB | raw | 75.090090% | 67.680180% | 17.500000% | 20.421053% | 4.285714% | 418/512 (81.640625%) | 0.8389105701 | 0.0061300775 | 0.0607579967 | 0.5685300265 |
| TrainB | EMA | 74.390727% | 67.229730% | 18.448276% | 22.000000% | 2.380952% | 413/512 (80.664062%) | 0.8325256291 | 0.0062443194 | 0.0615472446 | 0.5556673225 |
| Viewed dev | raw | 70.903235% | 62.745098% | 20.938628% | 25.402844% | 6.666667% | 318/512 (62.109375%) | 0.7966213089 | 0.0065737987 | 0.0625030268 | 0.4808648068 |
| Viewed dev | EMA | 71.053302% | 64.705882% | 22.599278% | 27.962085% | 5.454545% | 328/512 (64.062500%) | 0.8012594541 | 0.0064973227 | 0.0625321861 | 0.4873464434 |

Completed fixed diagnostics only. All earlier completed rows are preserved; no seed2-based epoch/state selection or final4096 access.

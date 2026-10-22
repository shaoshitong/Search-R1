# Full169028 MSE: actual fixed-pool epoch metrics

Existing MSE job749568 only. Each epoch uses the full169028-question pool, per-epoch independent shuffled order and own randomM3..10 trajectory subset. No final4096 read. Fixed train2048 and viewed independent validation2048 are held constant across epochs; raw updates EMA.9995. Variance loss is not binary-Mixed BCE. The fixed G5 classification threshold0.08 was specified before fitting and is not tuned.

| Epoch | Pool | State | Variance MSE | MAE | Spearman | Mixed BA % (G5) | FPR % (G5) | Top512 Mixed |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | fixed_train2048_G5 | raw | 0.010707 | 0.096419 | -0.0021 | 50.00 | 100.00 | 231/512 (45.12%) |
| 0 | fixed_train2048_G5 | EMA | 0.010707 | 0.096419 | -0.0021 | 50.00 | 100.00 | 231/512 (45.12%) |
| 0 | fixed_validation2048_G5 | raw | 0.011985 | 0.104457 | 0.0412 | 50.00 | 100.00 | 175/512 (34.18%) |
| 0 | fixed_validation2048_G5 | EMA | 0.011985 | 0.104457 | 0.0412 | 50.00 | 100.00 | 175/512 (34.18%) |
| 0 | fixed_validation2048_M3 | raw | 0.013679 | 0.116487 | 0.0082 | — | — | 124/512 (24.22%) |
| 0 | fixed_validation2048_M3 | EMA | 0.013679 | 0.116487 | 0.0082 | — | — | 124/512 (24.22%) |
| 0 | fixed_validation2048_M5 | raw | 0.011980 | 0.104416 | 0.0118 | — | — | 164/512 (32.03%) |
| 0 | fixed_validation2048_M5 | EMA | 0.011980 | 0.104416 | 0.0118 | — | — | 164/512 (32.03%) |
| 0 | fixed_validation2048_M10 | raw | 0.010804 | 0.096659 | 0.0050 | — | — | 214/512 (41.80%) |
| 0 | fixed_validation2048_M10 | EMA | 0.010804 | 0.096659 | 0.0050 | — | — | 214/512 (41.80%) |
| 1 | fixed_train2048_G5 | raw | 0.008852 | 0.084336 | 0.2404 | 56.26 | 7.41 | 307/512 (59.96%) |
| 1 | fixed_train2048_G5 | EMA | 0.008836 | 0.089019 | 0.1681 | 56.75 | 27.50 | 279/512 (54.49%) |
| 1 | fixed_validation2048_G5 | raw | 0.008313 | 0.081194 | 0.1834 | 52.79 | 11.62 | 219/512 (42.77%) |
| 1 | fixed_validation2048_G5 | EMA | 0.008625 | 0.087922 | 0.1469 | 55.91 | 26.50 | 211/512 (41.21%) |
| 1 | fixed_validation2048_M3 | raw | 0.008815 | 0.079117 | 0.1609 | — | — | 178/512 (34.77%) |
| 1 | fixed_validation2048_M3 | EMA | 0.009206 | 0.089289 | 0.1244 | — | — | 167/512 (32.62%) |
| 1 | fixed_validation2048_M5 | raw | 0.008465 | 0.081714 | 0.1968 | — | — | 241/512 (47.07%) |
| 1 | fixed_validation2048_M5 | EMA | 0.008747 | 0.088389 | 0.1648 | — | — | 228/512 (44.53%) |
| 1 | fixed_validation2048_M10 | raw | 0.007844 | 0.076357 | 0.2317 | — | — | 295/512 (57.62%) |
| 1 | fixed_validation2048_M10 | EMA | 0.008193 | 0.080787 | 0.1879 | — | — | 283/512 (55.27%) |

## Actual epoch coverage

| Epoch | Questions | Unique | Trajectory visits | Plan SHA256 |
|---:|---:|---:|---:|---|
| 1 | 169028 | 169028 | 1098580 | `1b54acb15a9e90a55d7339ee082565deb02ce734773ac822a94ea209b70c4143` |

Latest actual sample epoch1792686128.402759. Each metric is copied from the completed epoch evaluation receipt; an epoch absent from the table is not claimed complete. Terminal model/Adam/RNG acceptance is separate.

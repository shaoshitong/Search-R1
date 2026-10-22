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
| 2 | fixed_train2048_G5 | raw | 0.008294 | 0.083342 | 0.2947 | 62.29 | 23.62 | 329/512 (64.26%) |
| 2 | fixed_train2048_G5 | EMA | 0.008830 | 0.085579 | 0.2427 | 53.91 | 6.03 | 317/512 (61.91%) |
| 2 | fixed_validation2048_G5 | raw | 0.008304 | 0.083652 | 0.2031 | 57.30 | 28.23 | 221/512 (43.16%) |
| 2 | fixed_validation2048_G5 | EMA | 0.008322 | 0.082650 | 0.1836 | 53.32 | 8.45 | 226/512 (44.14%) |
| 2 | fixed_validation2048_M3 | raw | 0.008787 | 0.081670 | 0.1732 | — | — | 166/512 (32.42%) |
| 2 | fixed_validation2048_M3 | EMA | 0.008835 | 0.080650 | 0.1602 | — | — | 175/512 (34.18%) |
| 2 | fixed_validation2048_M5 | raw | 0.008416 | 0.083887 | 0.2143 | — | — | 239/512 (46.68%) |
| 2 | fixed_validation2048_M5 | EMA | 0.008467 | 0.083150 | 0.2031 | — | — | 243/512 (47.46%) |
| 2 | fixed_validation2048_M10 | raw | 0.007869 | 0.078229 | 0.2535 | — | — | 294/512 (57.42%) |
| 2 | fixed_validation2048_M10 | EMA | 0.007859 | 0.077574 | 0.2355 | — | — | 297/512 (58.01%) |
| 3 | fixed_train2048_G5 | raw | 0.008354 | 0.080779 | 0.3315 | 60.71 | 12.93 | 329/512 (64.26%) |
| 3 | fixed_train2048_G5 | EMA | 0.008621 | 0.083427 | 0.2901 | 57.40 | 8.97 | 336/512 (65.62%) |
| 3 | fixed_validation2048_G5 | raw | 0.008177 | 0.079873 | 0.2163 | 54.18 | 18.19 | 212/512 (41.41%) |
| 3 | fixed_validation2048_G5 | EMA | 0.008228 | 0.081147 | 0.2071 | 54.62 | 12.92 | 225/512 (43.95%) |
| 3 | fixed_validation2048_M3 | raw | 0.008719 | 0.079664 | 0.1849 | — | — | 168/512 (32.81%) |
| 3 | fixed_validation2048_M3 | EMA | 0.008764 | 0.078919 | 0.1768 | — | — | 168/512 (32.81%) |
| 3 | fixed_validation2048_M5 | raw | 0.008323 | 0.080267 | 0.2271 | — | — | 233/512 (45.51%) |
| 3 | fixed_validation2048_M5 | EMA | 0.008375 | 0.081598 | 0.2204 | — | — | 245/512 (47.85%) |
| 3 | fixed_validation2048_M10 | raw | 0.007707 | 0.074143 | 0.2606 | — | — | 294/512 (57.42%) |
| 3 | fixed_validation2048_M10 | EMA | 0.007739 | 0.076195 | 0.2560 | — | — | 294/512 (57.42%) |

## Actual epoch coverage

| Epoch | Questions | Unique | Trajectory visits | Plan SHA256 |
|---:|---:|---:|---:|---|
| 1 | 169028 | 169028 | 1098580 | `1b54acb15a9e90a55d7339ee082565deb02ce734773ac822a94ea209b70c4143` |
| 2 | 169028 | 169028 | 1100933 | `28c5fecd05219f79b0b3f2bdad708ed4b517497705d4b015dd48c7776e009bd2` |
| 3 | 169028 | 169028 | 1098015 | `d69d31e109e93823a81a8cb1976c8a6493e349ed448ad004f6679a64c7485e26` |

Latest actual sample epoch1792688712.637779. Each metric is copied from the completed epoch evaluation receipt; an epoch absent from the table is not claimed complete. Terminal model/Adam/RNG acceptance is separate.

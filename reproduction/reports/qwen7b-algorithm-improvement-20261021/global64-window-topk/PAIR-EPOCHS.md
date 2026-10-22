# Actual matched global64 TopK window-pair epochs

Root-owned jobs current64=749889, window2018=749890. No final4096 read. Primary Top512 uses the same viewed independent2048 ownG5; fixed train2048 is reported separately. Rank sigmoid scores are uncalibrated; BA threshold0.5 is predeclared and secondary.

| Cell | Epoch | Pool | State | BA% | AUC% | Top512 Mixed | FPR% | rho |
|---|---:|---|---|---:|---:|---:|---:|---:|
| current64 | 0 | fixed_train2048_G5 | EMA | 50.00 | 52.55 | 246/512 (48.05%) | 0.00 | 0.0490 |
| current64 | 0 | fixed_train2048_G5 | raw | 50.00 | 52.55 | 246/512 (48.05%) | 0.00 | 0.0490 |
| current64 | 0 | fixed_validation2048_G5 | EMA | 50.00 | 54.85 | 175/512 (34.18%) | 0.00 | 0.0835 |
| current64 | 0 | fixed_validation2048_G5 | raw | 50.00 | 54.85 | 175/512 (34.18%) | 0.00 | 0.0835 |
| current64 | 1 | fixed_train2048_G5 | EMA | 50.00 | 52.66 | 248/512 (48.44%) | 0.00 | 0.0510 |
| current64 | 1 | fixed_train2048_G5 | raw | 50.00 | 67.98 | 325/512 (63.48%) | 0.00 | 0.3077 |
| current64 | 1 | fixed_validation2048_G5 | EMA | 50.00 | 55.03 | 178/512 (34.77%) | 0.00 | 0.0863 |
| current64 | 1 | fixed_validation2048_G5 | raw | 50.00 | 63.02 | 239/512 (46.68%) | 0.00 | 0.2135 |

Only actual epoch receipts appear above; process start and queue status are not training acceptance. Each epoch audit is preserved in the small latest JSON and covers only its exact training-cache window, not a future or final pool.

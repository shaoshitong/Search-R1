# Completed matched current64 versus detached-window2018 development results

Both jobs completed10 epochs with the same8192 training questions, fixed ownG5 validation, initial head, layer15/Prefix80, global64, LR5e-6 and EMA.9995. Raw/EMA epoch/state choice has not been frozen. The primary metric is Top512 true Mixed; these are viewed development results, not final results.

| Epoch | State | Current64 devTop512 | Window2018 devTop512 | Current64 AUC% | Window2018 AUC% | Current64 BA% | Window2018 BA% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | raw | 175 (34.18%) | 175 (34.18%) | 54.85 | 54.85 | 50.00 | 50.00 |
| 0 | EMA | 175 (34.18%) | 175 (34.18%) | 54.85 | 54.85 | 50.00 | 50.00 |
| 1 | raw | 239 (46.68%) | 219 (42.77%) | 63.02 | 60.47 | 50.00 | 50.00 |
| 1 | EMA | 178 (34.77%) | 175 (34.18%) | 55.03 | 54.98 | 50.00 | 50.00 |
| 2 | raw | 251 (49.02%) | 241 (47.07%) | 65.01 | 63.88 | 50.04 | 50.00 |
| 2 | EMA | 176 (34.38%) | 173 (33.79%) | 55.45 | 55.18 | 50.00 | 50.00 |
| 3 | raw | 265 (51.76%) | 239 (46.68%) | 68.49 | 64.61 | 62.21 | 49.86 |
| 3 | EMA | 182 (35.55%) | 178 (34.77%) | 56.46 | 55.67 | 50.00 | 50.00 |
| 4 | raw | 265 (51.76%) | 254 (49.61%) | 68.89 | 67.44 | 55.80 | 61.21 |
| 4 | EMA | 196 (38.28%) | 183 (35.74%) | 58.09 | 56.55 | 50.00 | 50.00 |
| 5 | raw | 266 (51.95%) | 249 (48.63%) | 67.97 | 67.82 | 53.71 | 62.01 |
| 5 | EMA | 209 (40.82%) | 193 (37.70%) | 59.80 | 57.91 | 50.00 | 50.00 |
| 6 | raw | 264 (51.56%) | 263 (51.37%) | 67.20 | 68.95 | 62.43 | 64.13 |
| 6 | EMA | 212 (41.41%) | 211 (41.21%) | 61.46 | 59.67 | 50.00 | 50.00 |
| 7 | raw | 261 (50.98%) | 261 (50.98%) | 67.13 | 68.18 | 60.75 | 60.97 |
| 7 | EMA | 226 (44.14%) | 218 (42.58%) | 62.61 | 61.61 | 50.00 | 50.00 |
| 8 | raw | 253 (49.41%) | 252 (49.22%) | 66.19 | 67.67 | 57.91 | 59.18 |
| 8 | EMA | 235 (45.90%) | 232 (45.31%) | 63.64 | 63.14 | 50.00 | 50.00 |
| 9 | raw | 260 (50.78%) | 261 (50.98%) | 67.75 | 68.14 | 61.18 | 53.06 |
| 9 | EMA | 241 (47.07%) | 240 (46.88%) | 64.46 | 64.26 | 50.00 | 50.00 |
| 10 | raw | 257 (50.20%) | 269 (52.54%) | 67.09 | 68.61 | 57.67 | 54.84 |
| 10 | EMA | 242 (47.27%) | 244 (47.66%) | 64.95 | 65.16 | 50.00 | 50.00 |

Current64 used1280 Adam updates. Window2018 used1278 updates and correctly skipped2 cached-only constraints with zero current gradient. Both used1280 batch attempts and81920 question visits. CPU checkpoint acceptance is separate and root-owned. Rank scores are uncalibrated; BA at the fixed0.5 sigmoid threshold is secondary. No final pool was read and no candidate was frozen.

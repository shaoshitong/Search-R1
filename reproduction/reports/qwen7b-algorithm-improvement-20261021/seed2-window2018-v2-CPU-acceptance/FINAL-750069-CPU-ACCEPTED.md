# Corrected window2018 seed2: completed and CPU accepted

Actual job750069 completed0:0. Ten8192-question epochs produced1280 batch attempts,1273 Adam/EMA updates and7 cached-only zero-current-gradient skips. The previous job750023 failed argument parsing before any update and was superseded; its failure and original files remain recorded.

E10 CP SHA256: `2887a89b75824d3fcae2bdec3cdd46244c7499432653f5e6f044d8c505d81c50`. Terminal CPU acceptance SHA256: `b710c5cb816463edf50a9cfbfc0df4f76bcf09c9c6561646fefeadb3484eae69`. The shared8GiB CPU mutex serialized this loader;45 raw/EMA tensors,45 Adam states/moments/clocks, two-rank CPU RNG restoration plus16byte serializedCUDA RNG, ten exact explicit-seed own-G5 plans, source hashes and2018-row graph-free queue roundtrip all passed. Peak CPU RSS 4.166GiB. This does not claim a CUDA next-update restore test.

| epoch | Adam updates | trainB raw BA | dev raw BA | dev rawTop512 | trainB EMA BA | dev EMA BA | dev EMA Top512 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 49.96% | 50.00% | 155 | 49.96% | 50.00% | 155 |
| 1 | 128 | 50.80% | 50.28% | 224 | 49.96% | 50.00% | 157 |
| 2 | 256 | 52.51% | 51.43% | 227 | 49.96% | 50.00% | 161 |
| 3 | 377 | 59.63% | 57.77% | 245 | 49.96% | 50.00% | 162 |
| 4 | 505 | 58.48% | 54.97% | 252 | 49.96% | 50.00% | 168 |
| 5 | 633 | 52.88% | 52.14% | 265 | 49.97% | 50.00% | 186 |
| 6 | 761 | 59.38% | 56.49% | 258 | 49.98% | 50.11% | 193 |
| 7 | 889 | 54.25% | 53.53% | 258 | 51.04% | 50.21% | 212 |
| 8 | 1017 | 57.04% | 57.10% | 255 | 54.16% | 51.56% | 220 |
| 9 | 1145 | 51.54% | 50.79% | 266 | 60.69% | 55.81% | 230 |
| 10 | 1273 | 50.73% | 50.51% | 273 | 66.05% | 59.65% | 238 |

FixedtrainB has2048 questions/888Mixed; viewed development has2048/663Mixed. E10 raw trainB Top512 is419 versus dev273; EMA348 versus238. Raw dev AUC 0.69235, EMA 0.63402.

TopK-ramp trains unbounded ranking logits; the recorded sigmoid scores are not calibrated Mixed probabilities. The fixed0.5 BA diagnostic can collapse toward50% under a common score shift even while ranking/Top512 improve. It was not used to select questions or fit a threshold. E10 raw predicts nearly all questions above0.5, so its BA50.51% must be read alongside the actual273/512 ranking yield; no probability-calibration claim is made.

The durable coordinator reused the already accepted current64E5/MSE-M5E7 seed2 checkpoints and accepted only the corrected window-v2 E10; all three selected receipts are in `observer-complete.json`. Both coordinator and read-only observer exited normally. No checkpoint/rollout/model tensor was transferred locally. No Slurm action, training mutation, new inference or final4096 read was performed by this acceptance.

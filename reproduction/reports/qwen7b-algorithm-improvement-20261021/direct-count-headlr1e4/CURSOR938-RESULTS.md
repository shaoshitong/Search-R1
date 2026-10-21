# Head-LR-only matched candidate: final 938 updates

Job **743809 completed normally (0:0)** after the fixed 938-update single pass. All four scheduled checkpoints and fixed validations were accepted in the new namespace and configuration. This is an engineering completion and a scientific negative: the model still recalls no Mixed questions at the fixed 0.5 threshold.

| Update | Mixed BA | Mixed recall | NonMixed recall | AUC | True Mixed in Top512 |
|---:|---:|---:|---:|---:|---:|
| 256 | 50.00% | 0.00% | 100.00% | 0.552572 | 188/512 (36.72%) |
| 512 | 50.00% | 0.00% | 100.00% | 0.561871 | 195/512 (38.09%) |
| 768 | 50.00% | 0.00% | 100.00% | 0.571551 | 190/512 (37.11%) |
| 938 | 50.00% | 0.00% | 100.00% | 0.580316 | 199/512 (38.87%) |

The unchanged fixed 2048 pool has 663 true Mixed questions. At the final node, all-wrong and all-correct false-positive rates are both 0, and selected K=0…5 counts are `[225,45,43,45,66,88]`. Top512 contains 199 true Mixed questions, slightly above the same-HF E4 baseline’s 196, but balanced accuracy remains 50%; the joint gate fails. No significance or external-generalization claim is made, and no candidate is promoted.

Final CP938 is 3,754,035,012 bytes; SHA256 `14de7223746a741399e45662d56bfecf33a00b15727ba0ac6121008566eb76f0`. Legacy Adam reaches 11402 and the fresh head Adam reaches 938; actor updates remain 0. The original four-rank history/RNG archive is preserved.

At 256/512/768, independent full-nine-component CPU reloads ran inside the existing allocation. For terminal 938, the separate audit streamed the actual full file SHA and independently rejoined fixed IDs/own EM, recomputed q and all metrics, and inherited the producer’s saved full-nine-component CPU-reload proof. It did **not** independently load the 3.75GB checkpoint tensors on the login node. This boundary is explicit in the audit receipt.

The finite milestone observer exited normally after actual Slurm completion and all four acceptances. The allocation has ended; artifacts and checkpoints remain remote. No new GPU job, threshold fitting, bootstrap, or final-pool access was introduced. The separate frozen-E1 cache/head diagnostic continues under job 743978.

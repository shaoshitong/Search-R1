# Unique G5 class assignment versus label-blind G5

Read-only analysis of all **169,028** accepted questions and their ten real strict-EM labels. No model, feature tensor, checkpoint payload or GPU was used.

| Statistic | Label-blind own10→G5 expectation | Current class-conditional unique sweep |
|---|---:|---:|
| Mixed questions | 67157.016 (39.7313%) | 84,514 (50%) |
| Homogeneous G5 drawn from a G10-Mixed question | 19142.984 | 1,786 |
| Those hard negatives as fraction of all negative G5 | 18.7914% | 2.1133% |

Current question coverage is unique, but assigning 84,514 Mixed targets from only 86,300 G10-Mixed questions makes **97.9305%** of these questions positive. The hard homogeneous-G5 exposure from such questions is **90.6702% lower** than a label-blind five-of-ten draw. G10-pure questions account for 97.8867% of current negatives. This can reward identifying whether the question was ever mixed among ten trajectories instead of distinguishing its currently selected five. This is a mechanism hypothesis; its effect on independent accuracy remains unproved.

For K10=1 or9, a random G5 is Mixed with probability50%; forced assignment makes that97.9305%. For K10=5, the natural probability is99.2063%; forced assignment makes it97.9305%. Thus the distortion depends strongly on K10, not just a uniform class-prior shift.

Exact distribution uses P(K5=k|K10=K)=C(K,k)C(10−K,5−k)/C(10,5), with C(10,5)=252. Current conditional expectation then assigns Mixed with probability84,514/86,300 to each eligible question.

| K5 | Label-blind expected questions | Forced expected questions |
|---|---:|---:|
| 0 | 77632.738 | 69143.367 |
| 1 | 17614.405 | 25328.140 |
| 2 | 14778.833 | 15561.589 |
| 3 | 15667.913 | 16568.404 |
| 4 | 19095.865 | 27055.867 |
| 5 | 24238.246 | 15370.633 |

The independent control preserves one question visit per epoch and selects the five indices without reading EM. EM then supplies the actual target and global epoch class counts. Each class receives BCE weight N/(2N_class), divided by the actual batch size128 or68. This balances global loss mass without resampling questions or conditioning which five trajectories are selected. The same E1 raw/EMA/two Adam, learning rates, frozen E3 block and fixed train/independent pools are retained. Raw weighted and unweighted training BCE are reported separately; independent scores use the unchanged unweighted protocol.

Seed2026102203 metadata-only simulation produced67,191 Mixed/101,837 nonMixed with label-blind sampling; the forced control produced84,514/84,514. This simulation is not the actual training order.

Remote receipt: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/unique-vs-labelblind-G5-real169028-20261022.json`.

No improvement is claimed before the independent control has trained and been evaluated.

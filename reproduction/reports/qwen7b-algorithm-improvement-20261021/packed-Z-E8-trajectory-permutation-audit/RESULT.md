# Actual E8 trajectory-order audit complete

Job749929 completed0:0 onACD1-23 in1m45s. Same fixed viewed dev2048 questions and ownG5 labels/tokens, exact immutable E8 raw+EMA CP. Original predictions exactly reproduced saved E8 scores: maximum absolute difference0 for both raw andEMA. No final4096 access or optimizer/model update.

| State | Order | Top512 true Mixed | BA at variance.08 | Variance MSE | Rank correlation with original | Top512 overlap | Mean absolute variance change |
|---|---|---:|---:|---:|---:|---:|---:|
|raw|original|227|59.32%|0.00963508|1.0|512|0|
|raw|reverse|235|59.29%|0.00962483|0.987885|475|0.006091|
|raw|cyclic1|232|58.94%|0.00964160|0.990101|483|0.005610|
|EMA|original|235|57.53%|0.00866859|1.0|512|0|
|EMA|reverse|232|58.43%|0.00866918|0.984564|473|0.005207|
|EMA|cyclic1|239|57.83%|0.00865024|0.986830|472|0.004796|

This packed causal architecture is measurably order-sensitive. Depending on state/order,70–102 of2048 predictions cross the already declared.08 threshold; maximum absolute variance change is.0383–.0553. Yet the rank correlations remain.9846–.9901 and top512 overlap472–483. These three predefined orders change trueMixed selections by−3…+8 questions versus original. This audit does not establish the cause of the training/validation gap. Choosing the best observed order would be additional viewed-dev tuning; no such selection or final candidate freeze is made here.

Complete score differences, exact per-class allwrong/Mixed/allcorrect confusion counts, all source/identity hashes and remote-only prediction file proof are retained in actual-terminal-proof.json. The6144 packed actor rows (12288 raw/EMA scalar scores) took29.67s after model load; the complete Slurm wall clock was105s.

# Matched peer8192 ranking results

Both independent jobs completed256/256 real updates (Adam10464→10720), COMPLETED0:0 in30m24s. Their first next-update GPU replay and independent CPU full checkpoint reload passed all six components; final checkpoint SHA and ownEM/fixed-ID predictions were independently accepted. No actor update or optimizer reset occurred.

Every row below uses the same frozen2048 questions, real label-blind fixedG5, qualifiedHF L26/Prefix20 backend, and threshold0.5. PoolMixed663/2048=32.3730%.

|Branch|Mixed BA|AUC|Top512 Mixed|Mixed recall|NonMixed recall|AllWrong FPR|AllCorrect FPR|
|---|---:|---:|---:|---:|---:|---:|---:|
|Original peer/no rank|54.6555%|55.9899%|182/512|67.7225%|41.5884%|50.2370%|84.5455%|
|inter1|55.4754%|56.7254%|189/512|68.9291%|42.0217%|49.7630%|84.2424%|
|bothrank1|55.0855%|56.3177%|186/512|67.5716%|42.5993%|49.0995%|83.9394%|
|E4/same HF|55.7714%|57.6476%|196/512|67.5716%|43.9711%|51.1848%|71.5152%|

Inter1 improves the matched no-rank control by0.8199 percentage points ofBA and7 Top512 Mixed questions; bothrank1 improves by0.4300 points and4 questions. Neither exceeds the precommitted sameHF frontier: E2BA56.2505% and E4Top196. Neither branch is promoted. Per the rule frozen before their results, the next larger-data continuation starts from the original no-rank peer CP256, preserving its raw/EMA/Adam10720/RNG; it does not continue either failed ranking fork.

- inter1 job743466: complete SHA `3faee039b6bc17e11e89ce9665f219760a6d4cc27407018cc48a4e38f26b8ce3`; CP256 SHA `667b4cff22ba92f72c88f34f5eee320f30be7445c3ac7323fcc7e814eee27463`; independent audit SHA `528b2154738b19f2250f3795fc196342245cf1aef81b9d877cadfb943b9210cc`.
- bothrank1 job743467: complete SHA `04f3560b28cd12666122575f191c81f241999da3192cca402e6db39dff4b72a1`; CP256 SHA `2041c3e8ab24ecc9b5c04dd8f55fa3ed2f7ec5aaf839bb9f985915107cab841d`; independent audit SHA `668be9209f80fb52d26875c8af9cfb243fdd0eca6f8e6c1d61974fb73439cef8`.
- Runtime commit3c6faad3b821015b36b6c92ba51308898ac1b381 is immutable and separate from these reports. Detailed remote paths, bytes, actual learning curves, all256 grouping agreement, fixed ownEM checks, and final Slurm states are in the two small final-audit JSON files. Raw predictions, model/optimizer state, and features remain remotely stored.
- The finite observer exited normally. Jobs released their allocation naturally; sources and checkpoints remain intact. Final confirmation/stability pools were not read. This result closes these two experiments, not the overall improvement goal.

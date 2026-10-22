# Packed Z E8 trajectory-order audit

Independent viewed-dev audit only. The exact published E8 Z raw and EMA states use the frozen official Qwen2.5-7B actor, runtime block index21, Prefix20, one question plus the same five own prefixes. Orders are original, full reverse, and fixed one-step cyclic shift. Own IDs, ID-to-EM mapping, prompt, prefix tokens and population-variance targets are unchanged. Strict EM never enters model features.

Reports include original-variance/logit score deltas, score Spearman, Top512 overlap and trueMixed, predeclared variance threshold.08 BA and actual allwrong/Mixed/allcorrect confusion counts. Original-order outputs are paired to the already saved E8 dev predictions. Random trajectory order during training does not guarantee order-invariant predictions.

CPU tests exercised real pack_group/reorder functions and short-padding masks; no model/CP/GPU was loaded. The independent stage script pins E8 SHA2d81f15cd4af00fe192963cba6f6db86aba22bf47523bd3e10982b86f7178483, exact accepted fixed-dev G5/causal Prefix20 sources and architecture source. Stage does not hash or load the 3.77GB CP; the owned audit job fresh-verifies SHA before loading immutable E8. The currently running Z training is untouched.

Stage only: `python3 -m reproduction.operations.qwen7b_warmup.deploy_packed_trajectory_permutation_audit --commit COMMIT`. Root alone may submit the staged audit.sbatch after checking actual same-name jobs and locks. Requested resources:2GPU/12CPU/128GiB, normal/acd_u,25minutes. Both states share each packed actor prefill. Nothing accesses final4096, no training or optimizer is created, and no new candidate is frozen.

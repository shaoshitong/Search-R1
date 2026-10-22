# Five matched L15/Prefix80 entries and real CPU fixture accepted

Source pin: `0436c2c1e3c1b38a2abcbc252bf3524eff0e6deb`. All five source manifests and batch entries were staged and freshly SHA-verified. No Slurm job was submitted by this controller.

Remote suite: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-matched-L15P80-8192-20261023-v1`. The five separate `training.sbatch` entries are in `mse-Mrandom`, `mse-M5`, `mixed-bce-Mrandom`, `mixed-bce-M5`, and `topk-ramp-M5`. Each declares two GPUs,12CPUs,128GiB,2h,acd_u/normal. Root owns all submissions.

The CPU fixture actually completed in9.846seconds. Independent CPU acceptance then took11.171seconds: all8192×10 training and2048×5 fixed-validation rows match accepted source identities, strict EM and actual retained Prefix80 tokens, with fresh source and compact SHA checks. Of92160 trajectories,43567 genuinely end before80 tokens and are masked/padded. No Prefix20 source was extended to80. Fixture SHA: `a269f9d4bf22bef0e30afc7cbc0e9a708fb83cb73230fefa95106b56ec54d936`. Fixed independent G5 identity SHA: `c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7`.

These are token-data and source acceptance results. GPU training, initial-head/prediction equality across arms, optimizer execution and checkpoint acceptance remain unrun until root submits the entries. Final4096 remains untouched.

Prefix20 recovery of unaccepted replay qualification

Job742009 passed four-rank migrated raw/EMA forward qualification, then failed exact next-update replay before any accepted continuation updates. SourceCP2 unchanged. Job742046 uses immutable sourcee6b2a9f4319a with explicit deterministic FlashAttention, deterministic torch/CUBLAS, gradient-only DDP bucket warmup, and numeric replay diagnostics. Actual HPC3 CPU state-invariance test passed. Exact GPU replay, training progress and five-epoch quality acceptance remain pending.

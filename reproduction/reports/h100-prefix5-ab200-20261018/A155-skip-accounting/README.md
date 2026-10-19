# A155: actual writes and original guard dispositions

All four original A155 journals processed20 opportunities:6 ordinary writes,1 retained write with a post-update guard, then13 skipped opportunities. Each rank made7 actual Adam writes,3077→3084. All four original ledger hashes match the independently read completion observation.

After the seventh write (zero-based opportunity6), the original negative-advantage-tail guard rejected the frozen monitor observation. It recorded1 tail token among59 negative tokens: tail weight fraction0.01694915254237288, exceeding the unchanged reference1% limit. Sampled old-policy KL was0.0032950861857104683 (below0.2); invalid/overflow counts were0. The subsequent13 opportunities kept the unchanged3084 state, with the same explicit tail reason. The seventh successful write was retained; no rollback occurred.

This was the existing reference opportunity guard, not a newly added audit, CUDA/OOM failure or native training exit. The guard record's `fatal_reasons` field rejects an opportunity; it does not establish native-process failure. No threshold, algorithm, safety setting or running code was changed.

The original next outer156 completed20 actual writes on all four ranks, to3104. Its original leader stayed the same birth and157 had started. There are16 unwritten opportunities cumulatively through156 (3 earlier historical plus13 at155). Reports must use3104, not theoretical3120 or a sum of physical ranks. Normal outer counting follows the reference20-processed/at-least-one-write rule; it does not claim20 writes in every completed outer.

Only scalar reasons and hashes were returned; full logs, weights and tensors remain remote. This publication did not load a model or perform a new forward. Both200/no201, later QA/final51713 and actual terminal/resource closure remain incomplete.

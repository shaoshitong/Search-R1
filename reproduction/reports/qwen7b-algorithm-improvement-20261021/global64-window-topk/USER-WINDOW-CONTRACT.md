# 2018 total scores, 64 differentiable question groups

The user's current specification is a **total window of 2018 questions**,
including the current **64 questions**. Each question is one group of five
trajectories. At capacity the other **1954 question scores are detached**.

For one optimizer batch the ranking loss receives

\[
\widetilde s = [\operatorname{stopgrad}(s_{\mathrm{history},1:1954}),
                  s_\theta(x_{\mathrm{current},1:64})].
\]

The loss sorts the joint window and uses the labels of the actual selected G5
groups. Historical entries influence the ranking and its threshold, but have
no autograd graph. Actor forward and probe forward/backward work is performed
for the current 64 groups; the actor itself remains frozen. The detached queue stores scores, group identity, own-G5
labels and the actual model-update version, not hidden features or graphs.
The current entries enter the FIFO after the batch; oldest entries leave.
Repeated question IDs are removed before a new view is added.

This does not freeze a separate set of model parameters for old questions:
all samples share the probe. It freezes only their cached scores within this
batch. It also does not claim equality to a fresh differentiable forward of
all 2018 questions: historical scores can be stale. The implementation records
their model age and separately measures score drift. A batch whose loss depends
only on historical scores must not move Adam or EMA.

The existing implementation matches this contract:

- `detached_score_window.py`: `capacity` is the total capacity; history room is
  `capacity - current_batch`.
- `online_packed_window_topk_seeded_v2.py`: historical scores are materialized
  as constant tensors; only the current global64 logits carry gradient.
- The two completed window replicas used the same total2018/current64 contract.
  Seed2 job750069 completed 1280 batch attempts with 1273 real Adam/EMA updates
  and seven legitimate historical-only zero-current-gradient skips. Its saved
  2018-entry queue passed exact CPU restoration without hidden features/graphs.

The training window uses Top500 of 2018 as specified. Development evaluation
continues to use Top512 of 2048, and the final4096 confirmation remains blind.

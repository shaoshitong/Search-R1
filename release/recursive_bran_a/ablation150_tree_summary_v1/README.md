# Sparse tree and recorded cost summaries

This independent CPU collector completes the tree/cost fields from existing
production output. It does not change or restart the training or aggregate EM
watcher. Run from a published repository copy on the same node:

```sh
python -B -m release.recursive_bran_a.ablation150_tree_summary_v1.collect \
  --base /dataset/shaoshitong/recursive-ablation150-20261014 \
  --output /dataset/shaoshitong/recursive-ablation150-20261014/analysis/tree-summary-v1
```

`--once` reads currently closed points and exits. Otherwise it checks for new
5/50/100/150 boundaries every 60 seconds. Already persisted summaries are reused
after small file SHA checks, so a growing live log does not reparse accepted
trees. It never writes within a training group, takes training locks, calls
models/HTTP/GPU interfaces, opens weights/tensors, or generates new evaluation.

Each summary requires its immutable run/config/scientific source, a native
closed dev512, and the actual complete eight-rank post-evaluation manifest.
Tree counts reuse the published `report.tree_events` over exactly 512 unique
closed trees. Reward composition, contrast acceptance, actual generations,
branch action indices, candidates, selected count, shortfalls and recorded
branch costs come from existing sparse events. Incomplete or conflicting
coverage is reported rather than generalized to the full cohort.

`raw_inherited_leaf_tokens` counts repeated sampled policy-token occurrences
across selected leaves, using the native total-minus-unique(sample_id,offset)
definition. Its denominator is the sum of already saved root `model_mask`
counts and selected child `branch_end.cost.full_training_model_tokens`. This is
an occurrence duplication fraction, **not** mean child inherited prefix length.
If either source is absent, the fraction is null. `response_length/mean`
includes observations and is never substituted for a policy-mask denominator.
Only root JSON is read for the mask count; token/text/probability data stay on
the node and are excluded from the small aggregate.

Existing generation/discarded/prefill/retrieval/RPC/stage scalar sums retain
per-field outer coverage. Timers overlap; summed child elapsed time is not GPU
wall. Recorded GPU hours mean eight configured GPUs times recorded outer wall,
not active GPU utilization or lease billing. A zero-write replay remains
explicitly missing its original sampling cost.

Per keypoint, `summary.json` is the small publishable aggregate;
`provenance-private.json` stays remote with original event paths/SHA and native
log-line origins; `complete.json` binds both by SHA. `tree-summary-state.json`
provides their pointers. Four matched labels reuse their actual condition;
the A aggregate-only pack does not fabricate missing A tree statistics. Twenty
eight unique sparse reports cover seven new runs, but the collector always
leaves `goal_complete=false`: actual complete150 training/exit/release and the
other required study reports are separate requirements.

CPU fixtures:

```sh
python -B -m unittest release.recursive_bran_a.ablation150_tree_summary_v1.test_collect
```

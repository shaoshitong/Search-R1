# Full169028 L15/P80 MSE viewed-G5 trajectory order audit

This independent source-stage is deliberately unselected and not executable until root binds an actual epoch/state after selection within the full-data family. It does not load checkpoints, read saved predictions or score the final4096 pool while staging. Existing jobs and source roots remain unchanged.

The published training `7a8ff6ae8a33324846d85f77ffc28e9ab699c028` provides the exact original pack function and copied block/head architecture: L15, real Prefix80, 2048 head positions, one question prompt per packed group. Both full training seeds use the same later selected epoch and raw-or-EMA state. Exact checkpoint, CPU acceptance, source publication, complete10epoch receipt, epoch evaluation, saved dev prediction and actual physical-v2 fixture proofs are required before inference.

The input remains the original viewed NQ development2048 ownG5 (663 Mixed), never the final4096. Every view retains the same question, five own trajectory IDs, strict EM labels and real Prefix80 tokens. Only `original`, `reverse` and `cyclic1` are allowed, all three are reported, and the audit cannot choose a favorable order. Short genuine answers keep the original masked padding. Source-v2 accepted chunk metadata identifies the two dev shards, so the audit does not rescan all training tokens.

Original-order question/trajectory/label identity is checked against the same checkpoint's saved evaluation. Four-variance score differences are reported and must be <=0.001 absolute for an aligned-complete status. This numerical tolerance is fixed in source before audit execution. Failure retains an explicit alignment-failed status rather than attributing unrelated inference differences to order. Saved source metrics are retained alongside rerun original metrics.

Each permutation reports variance MSE/MAE, fixed-threshold Mixed BA/confusion/classwise errors, Top512 true-Mixed count and selected-QID hash. Reverse/cyclic1 also report mean/max/p95 score drift, logit drift, tie-aware Spearman ranking correlation, Top512 overlap, changed threshold decisions and true-Mixed count/fraction difference versus original. These are viewed-development sensitivity diagnostics, not a new holdout or seven-benchmark conclusion.

CPU tests cover ownID/EM/token invariance, exact P80 order and masked padding, label-free inputs, malformed geometry and mixed-epoch rejection, deliberately invalid future descriptors, and saved-score alignment using synthetic2048 only. Training packing is AST-identical to the published full-data source. No optimizer or actor/probe updates are created during the eventual audit.

## Source-stage and later root-owned invocation

Publish the source branch, then run `deploy_packed_full169028_permutation_audit.py --commit SOURCE_SHA --action stage`. This writes immutable source, `audit.sbatch`, an intentionally invalid `future-unbound-audit-template.json`, and `staged.json`; it never calls Slurm.

After explicit root selection and authorization only, populate a separate bound descriptor satisfying `descriptor()` with the actual two selected CP/source/CPU/evaluation proofs. Root may then submit exactly once:

```sh
sbatch REMOTE_AUDIT_ROOT/audit.sbatch REMOTE_BOUND_DESCRIPTOR_JSON SHA256_OF_BOUND_DESCRIPTOR
```

Resources are normal/acd_u, two H100,12CPU,128GiB,1hour. Both seeds × original/reverse/cyclic1 ×2048 questions imply12288 frozen-actor prefill rows. The selected state is fixed before audit, and no audit result is used to alter checkpoint, state, threshold or trajectory order.

# Root-only actual binding and submit-once control

This independently staged controller binds the already-published5899b504 fixed-trainB scorer. Source-stage runs only CPU reconciliation tests and `--help`: no actual binding, real prediction read, CP payload, GPU or Slurm command is executed during staging.

Later root invocation defaults to `--action bind` (small metadata only). Explicit `--action submit` first requires the actual749935 terminal CPU coordinator complete receipt, its exact accepted-E10 CP proof, and fresh sacct `COMPLETED|0:0`. Both Z E10/raw models must have accepted CPU proofs and actual source complete/evaluation/saved-trainB predictions. The main scorer's existing exact-G5 identity matcher and source-fixture e6e7 gate are reused. No independent review framework is added.

The bound descriptor preserves E1 EMA, fixed2048 trainB, both Z E10/raw replications, .5/.08 diagnostics and the predeclared first32 sorted existing dev QIDs for E1 adapter replay. It contains the existing E1 original evaluation/prediction SHA, exact native adapter source, input fixtures, same-CP acceptance metadata and source-publication proofs. Checkpoint SHA is inherited from actual independent CPU acceptance; the controller only stats CP sizes and does not open tensor payloads. The owned GPU scorer freshly checks E1's full immutable CP SHA before inference. Old trainA is not read or substituted. P80 is not appended before final state selection.

Before sbatch, root-bound JSON passes the original descriptor validator and is persisted with fsync, followed by an exclusive durable submit intent. Same-name active jobs are reconciled using exact owner, command path and config-derived Comment. If a previous submit outcome is unknown and no live same-name job is found, accounting is queried and the controller returns `PRIOR_SUBMISSION_OUTCOME_UNKNOWN_NO_RESUBMISSION`; it never guesses that retry is safe. Existing receipt/config/source hashes are checked on reuse. No training process, partition or existing task is changed.

After source-stage, root may invoke the immutable entry explicitly:

```sh
bash ROOT_CONTROL_CODE/root-bind-and-submit-once.sh --action bind
bash ROOT_CONTROL_CODE/root-bind-and-submit-once.sh --action submit
```

The stage task invokes neither command. Root owns actual execution, later progress verification and any resolution of an unknown submit outcome. The job uses the original normal/acd_u2H100/12CPU/128GiB/1hour entry, without dependencies, excluding the previously failed ACD1-13 host.

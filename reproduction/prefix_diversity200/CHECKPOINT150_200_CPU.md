# Actual150/200 joint-component CPU reader

Prepared locally only. This preparation did not contact H100, submit a reader,
read checkpoints, run a new unit/heavy test, or repeat5/50/100.

The two configs are pinned to run `f3ee1435e5014ba1969387c91878893a`, resume4,
runtime `e3463cfddca71a22ac684d75c875dbe123e8bc43`, config SHA
`384cf0f516e7bf9ce3623bdf48e87a078f8eceaa5af5e03f238e04e86ae59ad8`.
If these identities change, publish an explicitly updated config first.

## When the actual milestone is available

Run from the independent helper checkout:

```sh
cd /Users/shaoshitong/Documents/Search-G1/prefix-diversity-checkpoint5-cpu-20261016
python3 -m reproduction.prefix_diversity200.checkpoint_joint_components_control check \
  --config reproduction/prefix_diversity200/checkpoint150_joint_components_control.json \
  --receipts /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/search-g1-prefix1024-diversity512/acceptance150
```

`check` reads small joint/config/writer receipts only. A candidate is not a PASS.
Only after real joint150 and its successful stopped actor/probe writers exist,
use the same command with `deploy`, once. It stages published code in
`BASE/acceptance150-joint-components-cpu-v1`, then runs a separate owned systemd
reader. It never edits `BASE/repo`, checkpoint files or training processes.

Use the same command with `observe` to follow actual stage/file progress and
the terminal proof/exit. Observe at least twice, including completion; a
submission or live PID alone is not acceptance. Do not rerun `deploy`, change
the directory name or clear an existing intent/proof after a failure. Preserve
it and diagnose missing coverage before any explicitly authorized continuation.

For200, replace the config filename with
`checkpoint200_joint_components_control.json` and receipt directory with
`acceptance200`. That stages `BASE/acceptance200-joint-components-cpu-v1`.

## Coverage and interpretation

- Only150/200 are accepted by the new entry. Expected actor/probe Adam counters
  are3000/4000 per rank, scheduler150/200, next outer151/201.
- Before its once intent, the reader rechecks source/config/joint identity,
  exact writer receipts and stable immutable file signatures. Missing joint or
  active writer yields pending without a new intent or large-file read.
- It fully SHA-reads and CPU-deserializes the actual milestone HF435 FP32
  tensors/tokenizer, all8 native actor Adam/scheduler/RNG states, small driver,
  and current probe with its private8-rank RNG. It retains actual rank write
  logs, sameG5/native token/oldLP/reward inputs, Top1024 scores, strictEM variance
  Top512, all1024/5120 probe20-step mean normalization and data permutation/cursor
  checks from the shared reader. It SHA-verifies pinned runtime source files
  and train/validation/final splits. Old model weights are not read; the retained
  warmup100 small proof is reused.
- Driver identity and RNG remain checked. Actual joint pool snapshot and
  selection-time preceding joint/probe receipt define recovery bindings. The
  recognized stale resume3 pool next54/probe52 descriptors are recorded as
  standalone-driver FAIL and excluded from authoritative recovery data. Unknown
  mismatches fail the audit. No driver/checkpoint is silently repaired.
- The shared reader gained an explicit optional driver-validation callback;
  its default strict interpretation remains. Historical5/50/100 proofs and
  their executed source versions are preserved. The new proof records both
  entry/shared source SHAs and reports component status separately from the
  standalone-driver status.

Successful proof status is
`JOINT_MILESTONE_FULL_SHA_CPU_COMPONENTS_VERIFIED_EXPLICIT_RECOVERY_BINDINGS`.
Inspect `details.explicit_joint_driver_components.standalone_driver_status`
and `recovery_bindings`; do not report a stale standalone driver as PASS.
Proof, intent, stage log, process identity, exit and CPU/RSS cost live only in
the owned control directory. All weights/tensors/raw trajectories stay remote.
Copy only small proof/cost summaries locally and commit/push those receipts.

The reader uses emptyCUDA, Nice19, CPUQuota100%, MemoryMax12GiB, IOWeight10,
OMP/MKLthreads1 and RuntimeMax3600s. It performs no GPU forward, QA, short
training, CUDA RNG application, independent backup or native CUDA executor
exact-restore test. A component proof requires explicit joint-bundle recovery
binding; it is not an executed recovery or overall200-outer training completion.

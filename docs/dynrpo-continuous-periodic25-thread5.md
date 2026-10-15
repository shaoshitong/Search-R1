# Continuous native Qwen probe, independent HPC3 experiment

This revision retains the user-pinned native Qwen last decoder block and its
existing readout. The student still starts from official Qwen2.5-3B-Instruct
revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, with the canonical train/dev/
final split and existing actor/reward recipe. No other chat's output is written.

- Student budget: 200 complete outer iterations, four GPUs.
- Probe phases: 0,25,50,75,100,125,150,175. Outer200 saves and evaluates only.
- Probe data: 5120 distinct questions, five real strict-EM rollouts each,20epochs.
- Initial accepted10epochs/200Adam steps continue to20epochs/400steps.
- Subsequent phases inherit this same probe/readout/Adam/private RNG. There is
  no student parameter copy, reinitialization, per-outer fit, or cross-outer replay.
- After each finished phase, cumulative probe Adam steps are400,800,...,3200.
- Layer27 all-token BF16 features come from the live FSDP actor. Encoding and
  fitting do not update the student. Online feature tensors remain in remote RAM.
  The old169028 cache is read for question/token metadata only.
- Full checkpoints at25/50/75/100/125/150/175/200 supplement existing checkpoints
  and dev512 evaluation. Save and actual full-state verification precede refit.

## State and recovery

`PROBE_PENDING` pairs the completed student with the previous probe. Its full
actor model, four rank-local Adam/scheduler/RNG states, source cursor, and probe
Adam/RNG are authenticated by a pending-pair receipt before extra rollouts.
`PROBE_FITTING` stores the epoch, active permutation, minibatch cursor and true
head state. `STUDENT_READY` has a separate immutable post-probe snapshot paired
with the same student checkpoint. The original pre-probe checkpoint is retained.

Rollout labels and worker/driver RNG are checkpointed after each512-question
chunk. Recomputed online features must match the original per-question hashes.
Probe checkpoints are saved every epoch and after caught exceptions. A small
in-flight Adam ledger detects unpersisted possible updates after SIGKILL/OOM;
the program refuses to silently label an older epoch a lossless continuation.
Such a hard-kill ambiguity requires explicit recovery disposition.

Existing paired actor restoration remains responsible for actual CUDA model,
Adam/scheduler and RNG readback. CPU tests are not a four-GPU restart acceptance.
After a real launch/resume, inspect five complete student outers and the start
of the next outer. Training remains active toward200 after that acceptance.

## Paths and entrypoints

New namespace:
`/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016`.
Accepted initial assets remain read-only at
`/data/user/jxu957/experiments/qprobe-teacher-warmup200-native-thread5-20261015`.

The adapter is `reproduction/hpc3_dynrpo_periodic`. `control.py` publishes and
fetch-verifies source blobs, deploys immutable source, and reconciles Slurm jobs.
`observe.py` reads actual job/progress evidence on demand; no recurring Codex
monitor is installed. Current phase source is never edited in place. Stage-only
deployment leaves any running preparation/runtime source unchanged.

Try `acd_u` first, then `acd_ue` when actual scheduling evidence warrants it.
`emergency_acd` is the last tier, never the default. Do not create dependencies
on another experiment. Reconcile owned job identity and terminal state before
replacing a submission. Shared packages, data, models, and FAISS assets are
read-only; all generated states and caches are remote in this namespace.

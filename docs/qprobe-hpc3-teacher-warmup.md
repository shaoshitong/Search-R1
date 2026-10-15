# HPC3 question probe: current native4 execution contract

This companion describes the current launch and acceptance contract. It is not
a receipt that formal training, outer200, or the final CUDA restoration has
completed. The algorithm design remains in [question-diversity-probe.md](question-diversity-probe.md).

## Owner, allocation and immutable paths

| Setting | Current contract |
| --- | --- |
| Login owner | `jxu957`, actual UID `14090` |
| Slurm account / QOS | `jxu957` / `formal-user` |
| Partition | `emergency_acd` |
| Training allocation | One node, **four reserved GPUs**, 48 CPUs, `--mem=512G` |
| Training topology | `training_world_size=4`, actor microbatch 4 per rank |
| Recovery/final-audit exclusions | `ACD1-31,ACD1-67` |
| Storage admission budget | User-authorized provisional 2 TB, `2 * 10**12` bytes |
| Publication/control root | `/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014` |
| Immutable source | `ROOT/source/<full 40-character commit>`, with full-file publication SHA proof |
| Producer outputs | A distinct owned `qprobe-teacher-warmup200-*` remote base |

Four cards share the unchanged global training budget. Each rank handles 640
trajectories, with actor microbatch 4 and local minibatch 32; the global
minibatch remains 128. One successful outer has 20 optimizer opportunities per
rank. Actual accepted optimizer writes and rejected opportunities must be
measured rather than inferred from the requested budget.

The official initial/reference model is Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Model, canonical common data,
original retrieval assets and the existing Python3.10/Torch2.4/vLLM0.6.3 runtime
are reused through verified remote bindings. Native BF16 computation retains
FP32 master parameters, optimizer state and reductions. Python headers, private
runtime wrappers, library paths and node-local JIT/cache roots remain bound to
the actual runtime. CoSearch, TOP-D and shared model/retrieval assets retain
their own state and ownership.

Requested GRES is insufficient. Native admission must match Slurm device
assignments, NVML device minors/indices and actual CUDA UUIDs; all worker and
frozen-state checks use **four actual ranks**. Historical `common8` package names
and `eight-rank-*` artifact names do not change the required world size.

Before native model work and each archive reservation, the actual owner must
successfully measure shared experiment usage and reserve the required complete
checkpoint/output or archive/target copies within the provisional budget.
Filesystem free space does not prove personal quota. Permission failures block
the measurement; an observer's partial read cannot become an owner admission.
The 2 TB authorization is not a claim that the filesystem quota has been
independently confirmed. Large model, tensor, optimizer, FAISS, rollout and
evaluation files stay remote; local work keeps code, documentation and small
audit summaries.

## Extra frozen-teacher collection before formal outer1

Draw the fixed seeded pool (`seed=20261013`) of **5120 unique canonical training
questions**. Collect five complete real search trajectories per question using
the frozen official actor. Ten immutable chunks contain all 25600 trajectories,
their unique identities and five binary strict terminal EM labels. For correct
count `K`, preserve the mean `K/5` and sample standard deviation
`sqrt(K * (5-K) / 20)`, with `ddof=1`. Keep the `K=0` and `K=5` groups. A failed
generation, missing response or provenance failure cannot become a zero label.

The cache namespace pins exact questions, initial model files, sampling and
scorer sources/configuration, feature namespace and retrieval version. Reuse
requires fresh full SHA and binding checks. Accepted chunks retain their real
labels and trajectories; a verified partial cache can be copied independently
to a fresh recovery base when its teacher namespace is unchanged.

Fit the six-count question head on **all 5120 complete groups before the first
formal actor update**, including the existing mean/std auxiliary objective.
The legacy historical bootstrap cap of 512 does not apply to this warmup.
This is one extra collection outer, separate from the 200 successful training
outers. Head bootstrap may write its own CPU Adam state; it does not write actor
Adam, advance the actor scheduler, or complete an actor outer.

Actual four-rank before/after readbacks must prove unchanged actor parameters,
actor Adam and scheduler with zero actor updates. Restore and read back the
pre-warmup worker RNG boundary. Persist and read back the driver RNG, actual
candidate stream and private selector RNG before/after warmup, and compare the
saved initial head with the actual fitted head. Warmup does not consume the
formal candidate stream. A trained-actor recovery uses its complete paired
state and does not run teacher warmup or initialize a replacement head.

## Formal training and complete-goal acceptance

Each formal attempt predicts all 5120 candidates, draws `E=64` uniformly for
exploration, then selects 448 remaining questions with the greatest predicted
expected sample standard deviation. The 512 selected groups produce five real
trajectories each. Retain the original Search-G1 reward, GRPO/PPO, actor LR
warmup57, numerical checks and policy/provenance gates. No post-rollout EM
filter or refill is permitted. Commit one online head update only after the
corresponding actor outer succeeds; incomplete attempts cannot commit the head.

Completion requires all of the following evidence:

- The real frozen-teacher 5120/G5 collection, all5120 initial fit and actual
  four-rank/driver/data/RNG boundary checks described above.
- **200 successful formal actor outers**, with linked attempt, optimizer,
  scheduler and head receipts; no update201.
- Ordered canonical fixed512 validation at **5, 50, 100, 150 and 200**. At step5,
  the separate real inline dev512 evaluation is required; the historical
  stage64 evaluation is not a replacement for it.
- The final canonical **51713 native questions**, with exact ordered identities,
  seven-source counts, per-source scores and both macro/weighted EM and F1.
  Retain repeated canonical questions and remove only producer shard padding;
  do not assume the historical eight-card padding count applies to native4.
- Complete paired checkpoints at **all five** milestones, including HF weights,
  four-rank Adam/scheduler/RNG, driver/data state, question head/Adam/private RNG,
  replay, selector state and pinned reward dependencies. Each original pair
  must have separate archive and target copies with fresh full-file SHA and
  complete file-set checks. A later receipt's historical pointer does not
  archive an earlier checkpoint's weights.
- A distinct final200 native4 **restore-only** allocation that actually loads
  the archived target pair. Actual actor/model/Adam/scheduler/RNG and head/
  replay/selector/driver/data readbacks must agree at two boundaries, with zero
  new writes, generation, fitting, validation or update201. The question head
  uses constant-LR Adam and has no scheduler object.
- Actual owned-process cleanup and successful job terminal accounting. CPU
  structural/SHA reports, a Slurm submission, `RUNNING`, or an initial model
  smoke alone cannot accept the final recovery or the full goal.

The ordinary training restore binder continues to reject `step >= 200`.
Final200 verification uses the separate `final_restore_audit.sh` entry and
never enters the training loop. `verify_goal.py` is the final CPU consumer,
called after the native audit is actually `COMPLETED|0:0`; it must fail early
when completed200 evidence is absent.

## Once controls for archives and final native audit

Use an explicitly published helper source, even when the producer uses an older
immutable source. The producer's default environment/source link may not contain
new control modules. Populate these variables from verified actual receipts:

The producer job ID comes from its original submission and native allocation
admission receipts. The scientific run manifest need not contain a flat
`slurm_job_id` field. Controllers bind the admission's run ID, actual four GPU
UUIDs/indices, account, source command and work directory, preserve its full SHA
in the once-control binding, and check fresh Slurm accounting. Do not edit a live
run manifest to manufacture this field.

```bash
QP_HELPER_COMMIT='FULL_40_CHARACTER_PUBLISHED_HELPER_COMMIT'
QP_SOURCE="/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/source/$QP_HELPER_COMMIT"
QP_PRODUCER='/data/user/jxu957/experiments/ACTUAL_OWNED_PRODUCER_BASE'
```

For each actually published pair at 5/50/100/150/200, invoke once, substituting
the real milestone:

```bash
bash "$QP_SOURCE/reproduction/hpc3_qprobe/final_control.sh" "$QP_SOURCE" archive-start \
  --base "$QP_PRODUCER" --commit "$QP_HELPER_COMMIT" --step 5
bash "$QP_SOURCE/reproduction/hpc3_qprobe/final_control.sh" "$QP_SOURCE" archive-inspect \
  --base "$QP_PRODUCER" --step 5
```

`archive-start` writes an exclusive intent before starting a detached CPU
supervisor. Actual PID/start-time/session/owner binding authenticates the worker;
the supervisor retains its child and records its real wait status. CPU workers
use threadcaps1 and an empty CUDA mask, request no GPU, and load no main actor.
Archive reservations/copies serialize under the project storage lock. The
inspector reports actual process state, copied/partial bytes and the observed
child exit code; it does not launch another attempt.

Only after the producer is actually `COMPLETED|0:0` with final51713 and cleanup:

```bash
bash "$QP_SOURCE/reproduction/hpc3_qprobe/final_control.sh" "$QP_SOURCE" audit-start \
  --base "$QP_PRODUCER" --commit "$QP_HELPER_COMMIT"
bash "$QP_SOURCE/reproduction/hpc3_qprobe/final_control.sh" "$QP_SOURCE" audit-inspect \
  --base "$QP_PRODUCER"
```

The short `audit-start` entry detaches the heavy CPU verification and submission
work from the SSH window. It freshly checks all five source/archive/target file
sets, performs actual scheduler `--test-only`, then submits once with
`jxu957/formal-user/emergency_acd`, four GPUs, 48 CPUs, `512G`, exclusions31/67,
job name `qprobe-final200audit` and a new independent audit base. The GPU entry
is bound to the verified **target receipt** and complete transport proof.
The CPU submit controller does not access GPU devices or load the actor, but is
explicitly allowed to request this native GPU audit; requesting or allocating
resources is not CUDA restoration acceptance. Inspectors expose CPU process
births/IO, real child wait status and the distinct Slurm outcome.

An unknown Popen/SSH/sbatch outcome preserves the original intent. Never delete
its claim, rerun a start command, or fabricate a successful exit code. Once the
actual job is identified, explicit reconciliation checks its owner, source,
resources and nonce workdir, plus uniqueness among the matching active jobs:

```bash
bash "$QP_SOURCE/reproduction/hpc3_qprobe/final_control.sh" "$QP_SOURCE" audit-reconcile \
  --base "$QP_PRODUCER" --job ACTUAL_JOB_ID
```

Reconciliation observes the original job; it does not submit another one.
The final CPU goal check remains an explicit follow-up after successful native
audit completion. These controls create no automation and send no messages.

## Historical evidence retained separately

Earlier eight-card assumptions, 1 TB admission limits, account/QOS trials and
ACL failures are historical records, not current launch defaults. Account
transition ownership, shared-group access and scheduling evidence remain under
`ROOT/preparation/account-transition-*.json`; any earlier
`d_huiqimai_team/normal` settings are superseded by the current contract above.

Actual failed-startup originals and SHA manifests remain in the corresponding
remote producer bases, including:

- `qprobe-teacher-warmup200-4gpu-jxu957-recovery717827-20261014/failed-job-717938`:
  initial actor log-probability consistency failure, before formal work.
- `qprobe-teacher-warmup200-4gpu-jxu957-recovery717938-20261014/failed-job-718288`:
  owner storage measurement blocked by the two private checkpoint directories;
  the specifically authorized group-access repair retains content evidence.
- `qprobe-teacher-warmup200-4gpu-jxu957-recovery718288-20261014/failed-job-718322`:
  teacher policy-identity metadata transport failure, before committed chunks
  or formal work.
- `qprobe-teacher-warmup200-4gpu-jxu957-recovery718322-20261014/failed-job-718509`:
  NVML telemetry timeout after two committed teacher chunks, with no formal
  actor updates. Verified chunk reuse preserves this historical origin.

These bases are children of `/data/user/jxu957/experiments`. Failure archives do
not claim successful training or final recovery. The separate old-node31
process-birth/private-cache readback is required by final goal acceptance; it
does not claim that the unavailable GPU context query succeeded. Current
telemetry records bounded query failures as durable gaps, requires genuine
successful samples, and retains strict device/UUID/schema/durability and
training-evidence checks. Persistent query failure remains a failure.

## Independent final consumer after an eligible terminal CPU failure

The independent final reader and its once-only holder accept an explicit pair
of `--recovery-commit` and `--recovery-followthrough-commit` pins. Omitting both
keeps the original successful-producer route. Passing only one is rejected.
Recovery admission first requires actual outer200, the narrowly qualified
original producer failure, the separately held structural-recovery consumer,
and the separately held recovery followthrough with its new goal receipt and
actual native4 audit. The failed producer and original EA result retain their
failed status; the new consumer cannot label them successful.

An original RAW success is consumed directly. Only an explicitly qualified
terminal original RAW read failure can use the fresh pure-CPU reaggregation;
live, unknown and unrelated failures refuse this route. Its original failure
bytes are copied into the independent once scope and verified before launch,
after the child returns and when consuming the held result. Both supervisor
and child must be terminal, with the actual integer child wait status0 and
unchanged complete stdout/stderr hashes. The final reader rechecks the typed
producer reference and all recovery/native/raw anchors at its ending boundary.

This route retains the full teacher, raw/private, checkpoint transport, native
restore and no-update201 conjunction. It does not submit training, regenerate
labels, change any healthy run or mark the root goal complete. A missing200
refuses before creating its scope. The same once path covers normal and
explicit recovery intents; an existing unknown scope is never replayed.

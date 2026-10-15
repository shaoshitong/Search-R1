# Search-G1: A800 four-GPU formal200

This adapter runs the existing formal200 objective from the official
Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`,
with fresh Adam, scheduler, RNG and sampler. It does not load earlier A800
checkpoints or continue the separate H100 run.

| Quantity | A800 placement |
| --- | --- |
| Training GPUs | Any four actually idle cards outside auxiliary5/6 |
| Retrieval / probe GPUs | Retrieval 5–6, probe 6; actual excluded complement bound at launch |
| Outer batch | 512 questions × 5 responses = 2560 trajectories |
| PPO mini-batch | Global 128, local 32 |
| Actor micro-batch | Local 4, actual global 16 |
| Accumulation per mini-batch | 8 forwards/backwards per rank |
| Old/reference log-prob micro-batch | Global 32, local 8 |
| Optimizer opportunities | 20 per outer; successful Adam writes counted separately |
| Learning rate | 1e-6, 57 completed-outer warmup, then constant |
| Saves | 5, 50, 100, 150, 200 |

One trainer processes fresh0 through200. Checkpoint5 and its actual acceptance
are inline; passing that acceptance continues to6 in the same process. All20
opportunities must be processed and at least one Adam write must return for an
outer and one warmup advance to count. All-skipped or abandoned attempts do not
count; partial writes are retained and explicitly reported.

The canonical common-data manifest is
`0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`.
The original train169028/dev512/stage64/final51713 files are copied without
repartitioning. Training and validation refer to their real file hashes and
ordered IDs. The actual full2560 sampling-context/log-prob gate precedes Adam;
the launch plan labels it prospective until that real gate runs.

`prepare_run.py` binds a fresh namespace to published source, official local
model bytes and canonical data. `run_pipeline.py` starts actual local retrieval,
initial probe calibration, the frozen endpoint, and `train.sh`. Assigned GPUs,
host and process births replace Slurm identity; `slurm_job_id` is null. Historical
eight-GPU smoke tests are not synthesized as successful local evidence.

The local deployment/observation entry is `deploy.py`. It first requests the
published GitHub branch; a direct-download failure can use a Git bundle of the
same already-published commit through the Mac. Credentials, model weights and
environments are never copied into the repository. Existing A800 environment,
official model and retrieval assets remain shared resources; only the new run's
private files and recorded process births are eligible for cleanup.

Model execution uses the existing `/data/search-g1/env/bin/python`, headers in
`/data/search-r1/bootstrap-tools/python-dev`, and isolated per-run Ray,
Triton/Inductor/HF caches. The original model and retrieval hashes are verified
at deployment/runtime. The Mac controller dependency is the user's existing
`Documents/remote_codex_control/control.py` SSH route; deployment changes no
SSH/VPN settings or authentication files.
# Local startup recovery

Ray uses the manifest-bound private `/tmp/g-<execution_id>` root so its Unix
socket path stays within the system limit. The old failed namespace remains
unchanged. A new deployment may specify `--initial-probes-source-base` for a
completed official-model initial0 calibration. Its real labels, fitted heads,
model/data fingerprints and unchanged scientific producer bytes are verified;
copied heads receive a new run version with the original provenance. This does
not inherit trained weights or optimizer state, or claim a new calibration.

## Prospective question-probe teacher warmup

Use `deploy_qprobe.py` for this branch. Its `stage`, `prepare`, and `bind`
actions cannot start the pipeline. The official actor stays fresh; one extra
frozen-teacher outer generates all 5120 canonical question groups × 5 actual
responses, fits the new question-selection head on all 5120 groups, and restores
the actor RNG boundary. The actor then performs 200 successful training outers
with candidate5120/selected512/G5/E0, taking the top512 predicted diversity
scores with the existing identity-based tie ordering. The extra teacher outer is outside that
budget and performs no actor optimizer or scheduler writes.

The existing G1 binary reward probes remain separate from this new question
selection head. Preparation defaults to the original frozen initial0 source
`/data/shaoshitong/Search-G1-a800-fresh200-7895330a2c584bba9940720e98f006ef`;
`--initial-probes-source-base` can select an explicit source only at `stage`.
The existing adoption verifier still checks its real model/data/producer
provenance before the pipeline can use those reward probes. The new adapter
does not claim that this later GPU-stage adoption has already passed. Reward
probe refresh and save schedules remain 0/50/100/150 and 5/50/100/150/200.

Publish the exact source commit first, then use one small local handle throughout:

```text
python -m reproduction.a800_common.deploy_qprobe stage --handle HANDLE.json \
  --source-commit FULL_PUBLISHED_SHA \
  --published-ref refs/heads/codex/qprobe-a800-async-top512-20261014 \
  --baseline-intent-sha256 21ab1fc56365501b2082808f660dc7f3f130bedc9bb4748c63ada21c6032de3a \
  --teacher-warmup-cache /data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter
python -m reproduction.a800_common.deploy_qprobe prepare --handle HANDLE.json
python -m reproduction.a800_common.deploy_qprobe feature-plan --handle HANDLE.json
python -m reproduction.a800_common.deploy_qprobe inspect --handle HANDLE.json
python -m reproduction.a800_common.deploy_qprobe bind --handle HANDLE.json
python -m reproduction.a800_common.deploy_qprobe launch --handle HANDLE.json --review-receipt REVIEW.json
```

`stage` verifies the pushed generic branch ref, creates a Git bundle below32MiB
that reuses the original A800 checkout's baseline objects, and installs an
independent source namespace. It sends no untracked data/model/environment
files. `prepare` creates a new fresh200 namespace, copies and SHA-verifies the
six canonical inputs plus original official model manifest and formal method
template directly from the original completed1db base. The original deployment
intent must match the supplied exact SHA. Only source commit, source-map hash
and creation time change in the formal method. The old baseline's ninth input,
its historical source bundle, is excluded from the eight copied inputs.
Preparation uses CPU with CUDA
hidden and preserves BF16, actor local4/global16, accumulation8, reference
local8, formal guards and LR warmup57.

The fresh intent and prepared manifest explicitly mark GPU placement as
unbound. At final launch, the selector measures actual NVML inventory,
compute applications, memory and utilization, then binds four idle training
cards outside auxiliary5/6. GPU4/7 may be selected when idle. Every assigned
card, including auxiliary5/6, must have no compute application, at most256MiB
used and zero utilization. The immutable binding receipt records the selected
indices, ordered UUIDs and actual excluded complement. The final qprobe gate
binds before its launch claim; the deployment environment and launch receipt
derive from that actual manifest. Pipeline startup rechecks the same binding
and its120-second freshness without selecting different cards. Busy cards
cause a failure requiring observation; no process is stopped or shared.

`feature-plan` only prints the separate remote feature job's argv, environment
and progress paths. Run that job on A800 from the new run's `repo`, using the
printed environment and explicit auxiliary physical GPU5 or6 after checking
availability. `precompute_qprobe.py` invokes the existing feature implementation,
checks the actual single visible GPU and refuses an occupied device. It freezes
`CUDA_VISIBLE_DEVICES` to the admitted NVML UUID before importing Torch, then
checks the actual CUDA-runtime UUID against that physical GPU. It writes
`preparation/question-features-progress.json` with completed question counts,
PID birth and exit status, then an actual `features-ready.json` after full cache
verification. `inspect` returns these bounded summaries plus the existing
deployment observation. `bind` follows completed feature preparation and fixes
the immutable teacher cache path from the stage intent and output paths under
the new remote run.

For the snapshot-metadata repair, prepare a fresh independent run from the
published repair commit and reuse the completed BF4 encoding instead of running
`feature-plan` and a GPU encoding job:

```text
python -m reproduction.a800_common.deploy_qprobe feature-adopt --handle NEW_HANDLE.json \
  --feature-source-base /data/shaoshitong/Search-G1-a800-fresh200-27bab88929e048b4869f3889b99d8e18
python -m reproduction.a800_common.deploy_qprobe bind --handle NEW_HANDLE.json
```

`feature-adopt` only accepts that explicitly selected successful encoding. It
verifies original run/source and process birth, zero exit status, model/data
identity and unchanged encoding producers, then copies the three immutable
cache files on A800 and checks their complete SHA values before and after the
copy. The new readiness record says `copied_verified_existing_encoding`, records
the original GPU encoding separately from the current CPU copy process, and
does not claim fresh GPU encoding. Its exclusive feature claim prevents a
competing encoding job; failures keep the claim and partial output for diagnosis.
Use the new handle for launch; the failed BF4 handle remains historical evidence.

The repair seals generation metadata from the returned complete behavior
identity and every actual turn's matching snapshot. It rejects absent or mixed
turn evidence and a conflicting existing top-level value, then retains all
original strict checks. The current bounded-context and asynchronous sampling
protocol has a separate teacher namespace; the old v2 identity, chunks and
failed-run evidence remain preserved for their original protocol.

The fixed feature producers retain their exact bytes, so the completed BF4
encoding is copied into the new run without re-encoding. Rollout source changes
are recorded separately and cannot silently authorize reuse of old teacher labels.

Actor local4/global16 changes the training micro-batch only. The teacher identity
binds the complete rollout configuration, whose global log-prob micro-batch32
and all effective sampling fields. The first asynchronous v3 launch generates
5120 questions/25600 trajectories under its new protocol. Each fresh official actor
gets a newly fitted5120-group head with40 head optimizer updates; no old actor, head checkpoint or Adam state
is inherited. Initial0 frozen reward-head adoption permits exactly this
local2→4/global8→16/two-row→four-row budget transition alongside the existing
FP32→BF16 transition and retains every other identity/budget check.

The teacher cache defaults to the stable remote namespace
`/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter`, so a later fresh
actor launch with the exact same qualified protocol can reuse accepted trajectories. Choose another explicit
`--teacher-warmup-cache` only when creating a new intent. Its real path and
parent must be owned by the current remote user and contain no symlink. The
schema2 scientific identity binds model/data, question sampling/configuration,
retrieval and relevant producer source. Existing identity/chunks are never
overwritten or erased; a mismatch requires an explicitly selected new namespace.
Reused chunks receive the new run's lineage while the teacher cache preserves
its original provenance. `inspect` reads completion evidence from that bound
shared path.

The final `launch` reuses `deploy.py`'s original `REMOTE_LAUNCH_PREPARED` and its
compatible intent/receipt schema. Local launch requires the same published HEAD,
a clean checkout, the full committed and disk runtime source maps, and exact
committed bytes of both deployment modules. The imported launcher text must
also equal the committed snippet; the remote gate compares the submitted
snippet/hash with its verified published deployment module before executing it.
It requires an actual independent code-review
JSON receipt with role `a800_qprobe_independent_code_review`, `approved: true`,
`independent: true`, distinct nonempty `reviewer` and `implementation_actor`,
the exact `source_commit` and complete `source_sha256` map, a past positive
`reviewed_unix`, and `checks` containing exactly these seven keys set to true:

```json
{
  "official_model_and_canonical_data": true,
  "frozen_teacher_5120_x_5": true,
  "bootstrap_all_5120_groups": true,
  "exploration_0_top512": true,
  "independent_trajectory_phase_v1": true,
  "formal_200_not_201": true,
  "four_rank_bf16_local4_global16_accum8_ref8": true,
  "recoverable_guards_lr_warmup57": true
}
```

The adapter never manufactures that review. Its launch gate independently
checks the bound E0/top512 and asynchronous phase configuration, all5120 bootstrap limit, remote teacher cache, canonical
feature cache and four-rank recipe before the original launcher performs its
actual process/GPU admission. Every mutation has an exclusive local/remote
claim. A timeout or failure preserves the same namespace and forbids automatic
retry; use `inspect` and diagnose the real state. The local source bundle is
deleted only after A800 returns the matching SHA receipt. Keep a failed relay
bundle until its target is verified. No SSH configuration or credentials change.

Saved CP5 restoration uses a separate explicit intent, not the fresh0 route.
After the new published source's CPU qualifier has authenticated the original
checkpoint, pass `--restore-receipt REMOTE_SOURCE_RECEIPT.json` and
`--restore-receipt-sha256 FULL_SHA256` to `stage`. Then use the same `prepare`,
`feature-adopt`, `bind`, review and `launch` actions with that new handle.
Feature adoption copies the existing immutable encoding; `feature-plan` is
forbidden for this route. The original source/checkpoint/logs remain unchanged.

The distinct `question_probe_saved_checkpoint` state restores the original run
lineage into a new execution and namespace, including CP5 actor weights, master
optimizer state, scheduler, saved worker/driver RNG, sampler cursor and question
head. Logical training continues at6 through200. Its RNG boundary is the saved
checkpoint before the later development evaluation; post-evaluation RNG was
not persisted and bitwise post-evaluation continuation is not claimed. Existing
development512 measurements are linked without repeating evaluation, teacher
generation or head bootstrap fitting. CPU qualification is not CUDA/FSDP reload
acceptance; the single resumed trainer must perform the existing actual model,
Adam, scheduler and RNG readbacks before its first write.

The independent deployment review additionally requires
`saved_cp5_full_actor_head_optimizer_scheduler_rng_sampler`,
`logical6_to200_no_repeated_teacher_or_evaluation`, and
`saved_rng_not_post_evaluation_boundary`, all true for the exact published source.

For the sigma3 loss experiment, create a new handle with `--fresh-sigma3` at
`stage`. This flag is exclusive with saved-checkpoint restoration and fixes
the original initial0 reward calibration and the strict v4 teacher namespace. Every
later action reads that immutable handle; do not add the flag to later actions.

```sh
python -m reproduction.a800_common.deploy_qprobe stage --fresh-sigma3 \
  --handle SIGMA3_HANDLE.json --source-commit PUBLISHED_FULL_COMMIT \
  --published-ref refs/heads/codex/qprobe-a800-sigma3-teacher-v4-20261015 \
  --baseline-intent-sha256 ORIGINAL_BASELINE_INTENT_SHA256
python -m reproduction.a800_common.deploy_qprobe prepare --handle SIGMA3_HANDLE.json
python -m reproduction.a800_common.deploy_qprobe feature-adopt --handle SIGMA3_HANDLE.json \
  --feature-source-base /data/shaoshitong/Search-G1-a800-fresh200-27bab88929e048b4869f3889b99d8e18
python -m reproduction.a800_common.deploy_qprobe bind --handle SIGMA3_HANDLE.json
python -m reproduction.a800_common.deploy_qprobe launch --handle SIGMA3_HANDLE.json \
  --review-receipt SIGMA3_REVIEW.json
```

The explicit question-probe fields are `loss_mode: sigma3_std_aux6_v1`,
`mean_loss_weight: 0.0`, `std_loss_weight: 3.0`, and
`count_aux_loss_weight: 0.1`. The loss is CE3 over
`(p0+p5, p1+p4, p2+p3)`, plus three times the squared error between predicted
and actual **sample standard deviation**, normalized by `sqrt(0.3)`, plus
`0.1 * CE6`. It does not supervise variance or the mean. The six-count output
and expected-sample-std Top512 acquisition stay unchanged, with E0 and5120
candidates. The new official actor/reference, actor Adam/scheduler, all worker
and driver RNG, sampler and question head start fresh; the cached labels fit
a new head on all5120 groups. Existing runs and historical checkpoints retain
their legacy loss unless their immutable configuration explicitly selects the
new mode. The launch gate rejects inherited state and question-head paths.

The teacher's schema2 identity includes relevant rollout/scoring source bytes,
model/data, sampling, retriever and question bindings. It excludes question-head
loss source and settings. The older v3 cache is incompatible with the required
full-outer guard repair in `verl/workers/fsdp_workers.py`: its source SHA changes
from `c59cf8d4a2a92e72eaeb63e598d2dfef8fbd1ce7fc2b9e3336d1245411714483`
to `6dbb1b384fa56905f5db06d57c8accf4806dcb981e98dce7fda0b339adf62aab`.
The guard repair remains in force; the v3 cache stays immutable. A fresh v4
namespace generates the required 25,600 frozen-teacher trajectories and fits
a fresh head on all5120 groups. This cost is recorded outside formal200.
Later identical teacher identities can reuse qualified v4 chunks. No source
compatibility alias or identity exception is introduced. Full identity equality,
chunk and raw-rollout SHA checks, complete5120/G5 labels and frozen-worker
evidence still apply. A relevant rollout/scorer change remains incompatible.
Frozen initial0 reward-head adoption is separate from question-head state:
after all edits, regenerate `qprobe-initial0-adoption-source-delta.json` with
`python -m reproduction.a800_common.qprobe_adoption_source_delta` before the
final commit and publication. Its exact source map and changed-source pairs
preserve the original frozen G1 calibration/scoring bytes.

The independent sigma3 review needs the eight common checks above plus
`fresh_official_actor_reference_optimizer_scheduler_rng_sampler_head`,
`sigma3_ce_normalized_sample_std_mse3_aux6_ce_point1`, and
`teacher_namespace_full_identity_and_sha_no_old_question_head`, all true for the
exact published commit and source map. No extra evaluation or GPU preflight is
introduced. The four-rank BF16 local4/global16/accum8, LR1e-6/warmup57,
20 opportunities per complete outer, existing speed optimizations and guards,
and the formal200-outer target are retained.

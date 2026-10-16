FINAL200 RESTORE-ONLY — INTEGRATED CONTROLLER, NO CUDA ACCEPTANCE

The minimal patch preserves the mainline JIT cache repair. Existing step<200 training binder is deliberately untouched. Current training allocations/source snapshots remain immutable; this operator requires an independent final200 restore-only target.

New files:
  reproduction/prefix20_joint4/restore_only_assets.py
    Copies the complete CP200 actor/probe state, all boundary RNG/driver files,
    historical small evidence, reward heads+encoder, official frozen encoder,
    exact input data and epoch68 bootstrap files into a distinct target.
    Each copied file has original SHA, distinct inode identity and fresh target
    SHA, plus file/directory fsync. No tensor file may be a source hardlink or
    symlink. Passive source paths remain provenance metadata only.
    The rebound paired receipt and fresh live reward manifest point exclusively
    at target objects. Immutable source checkpoint files remain byte-for-byte
    copies; existing qstate internal rank paths are read ONLY as basenames under
    the copied target training_state directory, matching production restore.
    This is an explicit restore-only binder; it never loosens train bind_restore.

  search_g1/prefix20_restore_only.py
    Mutation guards, independent collective actual FSDP/HF tensor comparisons,
    rank-local Adam/scheduler/counter/boundary-RNG readback, probe private RNG
    activation/readback, and partial-pending canonical data membership checks.
    Scheduler guard overrides its class method, never adds an instance .step
    entry that would corrupt exact scheduler.state_dict() equality.

  reproduction/prefix20_joint4/restore_only.py
    Dedicated CPU Ray driver + real four actor workers, no trainer.fit() and no
    validation/generation/probe prediction/update/training checkpoint save call.
    Worker init loads copied HF model, restore_mixed_full_state loads actual
    copied Adam/scheduler/boundary RNG. Driver/stream/probe are restored and
    bound to actual trainer attributes at outer200. Two pure readback passes A/B
    compare against source copies and each other, with no reload between passes.
    After private Ray shutdown, recorded driver/actor births must terminate and
    allocated UUID compute contexts must release before completion publication.

  reproduction/prefix20_joint4/restore_only.sh
    Native4 offline entry; outputs/caches stay within the explicit remote base.
    Mainline process_jit_cache.configure() in the FSDP constructor remains used.

Existing-file minimal hunks:
  fsdp_workers.py: mutation guards after init; readonly readback endpoint;
    explicit target-only assertion before restore; guard actor update/logprob,
    generation APIs, capture and model checkpoint save.
  prefix20_probe_worker.py: allowlisted restore-only operations, guarded Adam,
    native private-RNG readback endpoint.
  ray_trainer.py: restore-only fit rejection at entry.
  claim_training.py: restore-only training claim rejection.

Preparation/execution sequence (requires actual final200; not executed):
  1. Preserve the five original source/archive/source full-SHA acceptances and
     their exact archive-receipt-import.json wrappers. Run the independent CPU
     final audit on completed1..200, fixed5/50/100/150/200 and native51713.
     Its RECORDED_EVIDENCE_PASS may have goal_complete=false; actual recovery
     is a separate required gate. Source Slurm must be COMPLETED0:0 and its
     owned cleanup accepted. Extra201 history or artifacts reject preparation.
  2. Deploy the reviewed, pushed restore commit to source-base/source/FULL_SHA
     as an additional immutable operator source; never replace the training
     repo or its source snapshot. Obtain a fresh complete personal-storage
     permission-partition observation, with its original measurement time.
  3. Invoke the CPU controller under the bound Python environment:
     python -B -m reproduction.prefix20_joint4.restore_controller prepare
       --source-base SOURCE --source-commit TRAINING_SHA --run-id RUN_ID
       --restore-commit RESTORE_SHA --base TARGET
       --cpu-final-audit CPU_JSON --cpu-final-audit-sha CPU_SHA256
     TARGET must be a fresh, globally unique namespace under
     /data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-restore200-ID.
     A real storage upper bound younger than6h plus conservative target/source
     copies and8GiB concurrent growth must fit2,000,000,000,000 bytes. The
     controller exclusively registers the one target, bootstraps its repo/env,
     and issues the copy permit before copying regular files with fresh SHA.
  4. Run restore_controller submit --base TARGET exactly once. It rechecks the
     actual final source, budget and copied bytes, then submits only
     gpu:4 / cpu48 / mem512G / jxu957 / formal-user / emergency_acd / export NONE,
     running restore_only.sh TARGET. A rejected or unknown sbatch result keeps
     its exclusive claim and has no retry API. Inspect it without resubmitting.
     The actual job checks its registered execution permit before CUDA/model
     construction, then independently validates allocation and exact command.
  5. Use restore_controller observe --base TARGET for measured read-only
     progress, and accept --base TARGET only after actual COMPLETED0:0.
     Required actual outputs: native-allocation-admission, copied file seal,
     full-state-restored-rank0..3/workers/driver, restore-only-readback-A/B,
     restore-only-owned-release, restore-only-complete. The coordinator checks
     real owner/account/queue/CPU/memory and four distinct process births/UUIDs.
     The accepted adapter is logs/actual-recovery-for-cpu-consumer.json; pass
     its exact path/SHA to the independent CPU consumer for final closure.

Acceptance semantics:
  - Actor writes equal receipt's actual cumulative writes, NOT assumed4000.
  - Four probe Adam states each have2000 steps and completed_outer200.
  - Scheduler last_epoch200 and _step_count201 are preserved saved counters;
    _step_count includes constructor initialization and does not mean outer201.
  - Candidate consumed is source actual count, NOT assumed200*5120; failed
    attempts can have consumed extra pools. All permutation, cursor, epoch,
    partial pending, canonical indices, eligible mapping and RNG remain exact.
  - Optimizer/scheduler/probe/data-consumption deltas0, no generation/save/fit,
    no next attempt, no new201 artifact. Both readbacks are actual reads, not
    reloads that could hide drift.
  - Pinned reward heads AND their encoder are copied and target-bound. No new
    reward calibration/refit/RPC is run; fixed-alpha manager version is restored.

CPU verification proves ONLY copy/path guards, target tamper rejection,
canonical partial-pending membership, counter semantics, forbidden API routing
and preservation of real CPU Adam/scheduler loading. It does not prove actual
FSDP/Ray/vLLM/encoder coexistence, GPU model equality, CUDA/private RNG restore,
or final200 archival completeness. Those remain mandatory real HPC3 tests.

Required real execution checks:
  - Deploy the exact reviewed/published operator source into an independent target; current training source is unchanged.
  - Source CP200 and native51713 evaluation must actually exist and be accepted;
    this component alone does not verify the whole final evaluation result.
  - All five external archive transport receipts must be authentic and accepted;
    controller receipt validation alone cannot establish a transport or CUDA run.
  - Validate rebound receipt with real final200 file layout and current installed
    Ray/vLLM/FSDP versions. Real collective/OOM failure handling must be observed;
    local tests cannot establish it. No live training source has been edited.

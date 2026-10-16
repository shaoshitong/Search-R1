FINAL200 RESTORE-ONLY — CPU-TESTED IMPLEMENTATION, NO CUDA ACCEPTANCE

Delivery: restore-only.patch, files/, BASE.json, delivery-manifest.json,
cpu-tests.txt, cpu-test-result.json.

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

Preparation/execution sequence (prospective only; not executed):
  1. Admit exactly one additional restore target under the unchanged 2TB budget,
     using the REAL total copy plan bytes, including encoder dependencies. The
     copy module intentionally does not submit or invent storage admission.
  2. Create a published/bootstrap-only target namespace:
     /data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-restore200-ID
     Deploy reviewed committed source and bootstrap its real repo/env. It must
     contain no run manifest or restore-copy claim. Bootstrap data symlinks are
     replaced only in this new target by verified regular data copies.
  3. Under the target's reviewed source, invoke:
     python -m reproduction.prefix20_joint4.restore_only copy --base TARGET
       --source-receipt VERIFIED_CP200_PAIR_RECEIPT --commit FULL_PUSHED_SHA
     This is a CPU file-copy operation. It accepts ONLY final200/native4/schema3.
     Source receipt must be the verified source or archive intended for recovery;
     the later GPU process loads target copies only. Inspect any failed claim/
     partial copy; never replay blindly or delete original evidence.
  4. Submit once, native gpu:4 / cpu48 / mem512G / jxu957 / formal-user /
     emergency_acd / export NONE, running restore_only.sh TARGET from TARGET/repo.
     This draft does not implement a scheduler submit/claim controller; reuse
     the mainline once-submit discipline and verify exact actual command/target.
  5. Required actual outputs: native-allocation-admission, copied file seal,
     full-state-restored-rank0..3/workers/driver, restore-only-readback-A/B,
     restore-only-owned-release, restore-only-complete. The coordinator checks
     real owner/account/queue/CPU/memory and four distinct process births/UUIDs.

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

Known integration work/limits before invoking:
  - Deploy the exact reviewed/published operator source into an independent target; current training source is unchanged.
  - Source CP200 and native51713 evaluation must actually exist and be accepted;
    this component alone does not verify the whole final evaluation result.
  - External scheduler/controller must own job submission and CPU copy process,
    durable failure evidence and the existing one-target storage reservation.
  - Validate rebound receipt with real final200 file layout and current installed
    Ray/vLLM/FSDP versions. Real collective/OOM failure handling must be observed;
    local tests cannot establish it. No live training source has been edited.

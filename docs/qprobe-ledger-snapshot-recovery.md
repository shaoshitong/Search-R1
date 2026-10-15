# Coherent process-ledger verification

The active HPC3 qprobe producer is job719225, with immutable training source
1843babdd96b387dd10d91e3207c9ea9489b6941 and four GPU ranks. Its owned-process
observer atomically replaces the process ledger while training and while waiting
for its built-in final CPU acceptance. Actual readonly observations at outers69,
73 and75 saw transient `FileNotFoundError` during a ledger read; the same original
verifier subsequently completed0. These were observer failures, not actor failures.

The new `reproduction/common8/ledger_snapshot.py` reads one regular, owned file
through a single no-follow FD. It checks file identity/content metadata before and
after the complete read, and binds the parsed object and all rank proof hashes to
the same bytes. Atomic replacement of the path may leave a valid older opened
snapshot; this is explicitly historical evidence and makes no current-liveness
claim. Every original run, leader/worker birth, UID/SID, scheduler, LR and actual
write-count check remains in the scheduler verifier.

Only missing/stale read errors (ENOENT/ESTALE) trigger bounded reopen retries.
Malformed JSON, wrong ownership, symlinks, non-regular files, permission/storage
errors, in-place content changes and invalid birth/scheduler evidence remain fatal.
The reader never edits evidence or signals a process.

This change is for separately published future verifier/source invocations. It
does not hot-modify the healthy producer, EA controller or RAW795 controller, and
cannot retroactively repair job719225's already loaded built-in verifier.

## Terminal recovery boundary

There is no shortcut that accepts a FAILED producer as COMPLETED0. If job719225
actually fails only in its final CPU acceptance after
all200 authentic updates and final51713 are complete, recovery requires a new
typed proof bound to the original FAILED accounting, source/run/base/claims,
precise ledger-read traceback, actual trainer exit0, final-acceptance exit1,
successful owned cleanup, all five paired checkpoints, canonical final metrics,
all actual attempts and absence of update201. Preserve the original failed logs,
ledger and pipeline receipt. Other training, numerical or cleanup failures must
not enter this recovery route.

A separately held CPU-only structural acceptance may then revalidate existing
evidence without training, generation or actor reconstruction, publishing to a
new output rather than overwriting the failure. Any later recovery controller
must preserve existing archive claims and authentic held0 results, require all
five source/archive/independent-target SHA proofs, run the actual native4
restore-only audit with two equal readbacks, and consume original teacher/raw
proofs through the independent final verifier. The original FAILED accounting
must stay FAILED; no fabricated pipeline/EA/RAW PASS receipts are permitted.

The first recovery stage is implemented in
`reproduction/hpc3_qprobe/final_cpu_ledger_recovery719225.py`. It admits only the
actual terminal `FAILED|1:0` producer with a Slurm end time, the pinned original
submission/allocation/source, authentic original1..5/current6..200, original
trainer exit0/final-acceptance exit1, stopped owned births, successful cleanup
and the precise original `FileNotFoundError` for this producer's process ledger.
Bare read-stage ENOENT without a filename requires the exact original runtime
line181/pathlib-read or runtime315/verify_data-line17 digest/lambda tail chain;
unrelated bare ENOENT remains fatal. Original source SHA is verified in full.
A healthy/pending producer or missing200 rejects before any once directory,
source scan, child or claim. Other exceptions and cleanup/training failures
reject. Existing unknown scopes are inspected without replay.

The once scope preserves complete bounded copies of the original failure files
and verifier source, binds their full SHA to the original accounting and claims,
and rechecks both original and preserved bytes before/after its held CPU child.
The child invokes `verify_full(..., publish_receipt=False)`, independently checks
all original attempts/head/actor/scheduler evidence, canonical fixed/final
metrics and five paired-checkpoint metadata, then writes a new separate
`structural-acceptance.json`. Its supervisor records an actual child wait,
PID/start/session/UID, source, stdout/stderr full SHA and error. Its readonly
consumer requires both processes terminal and equal original/preserved evidence.
`verify_full` retains its original publish-on-success default. The new separate
stage keeps the producer FAILED and explicitly reports `goal_complete=False`.

`final_control` now has an explicit `recovery_commit` branch through final CPU
submission, its held worker, full preflight, actual Slurm intent, readonly inspect
and unique-job reconciliation. Default calls still require actual COMPLETED0.
The branch privately loads the exact published stage1 consumer and reconsumes
its actual held0 proof, original/preserved bytes and precise FAILED1:0 evidence;
the same stable reference is bound through every subsequent claim. It retains
the original once paths, so any prior/unknown native intent still prevents a
second submission. All five completed source/archive/independent-target full
SHA checks and canonical final51713 remain required before native submission.

CPU and native source pins are intentionally distinct on recovery. The repaired
CPU verifier may differ in `common8`, while native loading/training-state code
must match original1843 exactly. The native entry and commit are therefore the
already published EA source `ea4601109969e358b65ab8251d64798926748b7d`, whose
protected scientific source prefixes match1843. Its publication is freshly
verified, `bind_audit` still checks every protected hash, and submit/reconcile
both bind the actual EA native command. The original native four-rank restore
and two-readback checks are not relaxed.

`verify_goal` also accepts an explicit consumed recovery commit. It reads the
separate structural receipt, keeps original pipeline FAILED, retains every
teacher/attempt/fixed/archive/native/metric gate, reconsumes the typed failure at
the end, and publishes only into an already owned separate recovery-controller
once directory. Normal receipt paths and default finished-pipeline checks remain
strict. Neither adapter claims that a native job or the final goal has run.

The third stage is `recovery_followthrough719225.py`. Its `start --commit <new
published helper> --recovery-commit <actual held stage1 helper>` entry refuses
healthy or missing200 producers before creating a scope. It verifies and copies
the pinned EA's exact producer-failure traceback, claim/source, stopped birth,
self-reported exit1 and failure marker. EA has no external parent wait receipt;
the new proof explicitly preserves that distinction. It does not infer current
completion from EA's previous state, which can be stale when one_tick raises.

The new supervisor retains a CPU worker with an actual wait status. That worker
reconsumes existing serial fixed archives and requires full claim/controller,
runner/task births, token, wait0 and task-result SHA binding. Both processes must
be dead before accepting a completed archive. Transport path and actual SHA are
checked before fresh source/archive/independent-target verification. Only a
missing control scope with no payload may create a new archive intent. Unknown,
partial, failed or orphan scopes are never replayed.

After all five archives it uses the explicit typed final-control branch and the
EA-pinned native4 restore-only entry. Final CPU submission also requires the
complete held runner/task identity chain. Known queue/registration states wait
on the same job; unknown outcomes require inspection. Actual native COMPLETED0,
no new actor/head/scheduler/generation work, readback/accepted hashes and owned
cleanup bind the separate goal receipt. The terminal consumer rechecks original
failure/copies, actual worker wait0, source, output SHA and native/goal identity.
Every result retains `full_objective_accepted=False` and `goal_complete=False`.

`raw_terminal_reaggregation719225.py` adds a pure fresh conjunction only after
typed719225 failure admission and actual dead RAW795 with every original
normal-return child already retained. The original result must be a missing or
stale read failure; its full supervisor output, claim, pinned source, parts and
birth remain unchanged. A fresh original claim read pins the actual real Python
executable recorded by795, which differs from the producer's prepared env alias.
Accepted and attempt directories must be exact full sets, including guarded
zero-write normal returns. Each original child is reconsumed to real integer
wait0, terminal birth, empty stderr and unchanged source/claim/result/output.

RAW795 did not persist a traceback or internal exception phase. The new reader
explicitly says its original failing call site is unproven, rather than claiming
an exact aggregation or process-ledger failure. It freshly checks raw/private,
canonical data, tokenizer and original scientific source bytes with SharedReads,
rechecks all child anchors, matches fresh formal UUID order and finally rechecks
the original failure/task set. No original795 failure is replaced or relabeled
PASS. It creates no process, job, receipt or scope, decodes nothing again and
still requires the independent final holder's actual wait0 consumption.

The independent held final recovery branch and combined final positive
integration remain separate work. No recovery scope or job has been activated.
If the healthy original producer succeeds, its existing chain remains
authoritative and no recovery job is submitted. Constructed protocol tests do
not establish real training, real Slurm failure or real native recovery.

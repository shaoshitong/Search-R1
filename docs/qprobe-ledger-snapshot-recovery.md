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

There is currently no implemented shortcut that accepts a FAILED producer as
COMPLETED0. If job719225 actually fails only in its final CPU acceptance after
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

Implementing that typed recovery controller remains separate work. If the
healthy original producer succeeds, its existing chain remains authoritative and
no recovery job is submitted.

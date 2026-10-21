# Historical operations / control backup

These scripts preserve the exact code used from /tmp during preparation,
publication, resource inspection and the unique job745022 submission. Submission
has already happened: **never replay easy7b-submit.py**. Use job.id and current
squeue/sacct/logs instead. A pending job or observation timeout is not a failure.

The two bridge scripts read existing private credentials in RAM and pass them
only over stdin. They contain no credential values. The model stays on HPC3.
The initial broad preflight and bounded du attempts timed out and are retained
as failed observations. The successful storage scans use detached, explicitly
claimed PIDs; do not duplicate those scans. MOU cannot write the JXU-owned control
directory; the separate owned-/tmp MOU scan fixes only its audit destination.
No training code or source is changed by those audit tools.

The active training job is `745371`, using immutable runtime `e082a23b5a65b302e2016aa0e5ab43b10b543d4f`.
Its original HK CPU archive controller `3420413` exited after an SSH observation
authentication failure; its claim, failure and logs remain unchanged. The unique
replacement is under `job-745371/recovery-transport-r1-20261021`, using controller
code `0830581f16eee86d5a89100cda91b175de63635c` while all source checkpoint commands
still bind the original training runtime. Observe it with
`observe_hk_archive_transport_recovery.py`, not the historical root observer.
The recovery launcher has already been executed once. Never replay a launcher
because an observation timed out; inspect its existing claim and process birth.
No training or retrieval job was restarted for this CPU-only recovery.

The human's later 2026-10-21 cleanup instruction supersedes latest-two retention
only for the twelve explicit legacy roots in `apply_authorized_old_checkpoint_cleanup.py`.
It authorizes all old experiment checkpoint payloads there, while retaining logs,
all text, and evaluation trajectories. The 7B feature/probe root, QProbe roots,
current 7B source/recovery roots, shared model/data/index assets, and HK archives
are outside this cleanup. Use the frozen exact manifest and inspect existing
per-UID claims before any apply; never replay an uncertain deletion attempt.
The initial foreground read-only inventory timed out, so the completed inventories
were collected under separate pinned `/tmp/authorized-old-checkpoints-20261021-<uid>`
claims. No training was restarted for this inventory or cleanup.

The updated human goal also requires notification and a complete-state recovery
on clear EM/reward collapse. The read-only training observer exposes completed
outer metrics and actual fixed-validation summaries; it does not add a training
threshold, change the loss, or restart a healthy job for a single fluctuation.

The CP5 post-callback supplement uses `archive_post_callback5_local.py` and its
independent `archive_post_callback5_hk.py` CPU runner. The core archive must first
have actual fresh-source/fresh-destination SHA acceptance. Only then can the
one-shot runner preserve the seven post-QA5 boundary files plus the rolling
manifest in a separate child of that accepted archive. Module bytes are frozen
at `b87a0c3d7a7ca2d9f6751a99179897d8d8aca56d`; runner/observer bytes come from the
separately published operator commit. No model/Adam copy, model reload, training
edit, Slurm submission, or healthy-controller restart is performed. The optional
observer `--verify-files` reads all eight small destination files and their SHA.
Existing launch or operation claims must be inspected, never replayed. A successful
supplement preserves bytes; it does not certify an actual CUDA restore. The final
outer200 native-four-rank recovery remains a separate required acceptance.

Normal fixed512 evaluation precedes its checkpoint save; the extra QA5 callback
and the grounding refits at50/100/150 follow the save. A core snapshot taken
before a refit can therefore contain the previous grounding heads. The eight
post-callback files preserve the new manifest and RNG, but that alone does not
preserve the new head/report/dump bytes referenced by that manifest.

`post_callback_dependencies.py` independently closes those references in a new
`post-callback-dependencies` child: both sklearn heads, calibration report,
version manifest, all `dump_sha256` members and exact encoder mappings. Files
already covered by the accepted core reuse its SHA ACK and current physical
identity; default inspection does not read model tensor bodies again. New
transferred dependencies receive fresh source and destination SHA checks. A
restored round without local dumps can follow one authenticated source receipt
to its original manifest/dumps; no filesystem-wide search is performed.

`archive_post_callback_future_local.py` launches an independent CPU companion.
It qualifies this new dependency closure on CP5 without replaying the existing
eight-file writer, then handles50/100/150/200 after each actual core ACK and
post-callback boundary. Every writer has an exclusive claim. The existing core
controller and training remain unchanged. A specific dead core-controller birth
is a dependency failure; an observation timeout remains unknown. At200 the
reward round is150 and the RNG boundary precedes final51713. None of these byte
archives is an actual CUDA restore or a completion claim for the200-outer goal.

The first future companion, PID3844281, actually exited after transferring
50,580,970 dependency bytes: HK Python3.8 lacks `Path.is_relative_to`. Its
source snapshot, received stage, claims, failure and logs remain preserved.
`recover_post_callback_future_local.py --mode recover` is the explicit one-shot
compatibility recovery: authenticate that original terminal birth and exact
failure SHA, freshly verify and accept the existing stage, then run a clean CPU
companion under `future-50-200/recovery-python38-r1`. It does not replay the
source snapshot/emit/receive or the already accepted eight-file boundary.
The new companion inspects and reuses the accepted CP5 dependency child before
waiting for50. The dedicated recovery observer follows its own actual birth.
Never replay either launcher after an uncertain response; inspect its claims.
At the time of publishing this repair, actual recovery execution is still
pending. Training is healthy: outer7 is complete with20 writes per rank, CP7 is
sealed and outer8 has a real saved512-question chunk. No training, retrieval or
healthy core-controller restart is part of this CPU repair.

That Python3.8 recovery actually accepted CP5's10 dependency references, including
the6 transferred files/50,580,970 bytes. It then exited in a reader that compared
the dependency receipt's nested `binding.core` as though it were the boundary
receipt's flat `binding`. The data is already accepted; its original partial no
longer exists. `resume_post_callback_future_local.py --mode resume` continues
only from that accepted archive under `future-50-200/recovery-accepted-r2`.
It authenticates the exact r1 failure and terminal birth, and the normal
companion freshly inspects the accepted CP5 child using the corrected nested
binding. No accept, receive, emission or snapshot is replayed. Both failed CPU
roots remain immutable evidence. A fixture now exercises the real complete
dependency archive lifecycle and its actual receipt schema through the reader.

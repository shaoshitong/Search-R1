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

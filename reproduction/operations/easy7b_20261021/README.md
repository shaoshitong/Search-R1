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

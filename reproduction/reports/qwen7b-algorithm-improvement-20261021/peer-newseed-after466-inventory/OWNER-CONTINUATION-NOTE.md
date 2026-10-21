H100-10 actual owner/auxiliary continuation, 2026-10-21

The supplementary gpu4 slot stopped normally at a batch boundary, after112 own
sealed batches/14,336 questions. Its terminal receipt reports
`POOL_BATCH_BOUNDARY_TIME_LIMIT`, sampler return code0,
`OWN_RELAY_CONSUMER_FINISHED`, own children reaped, and
`global_goal_complete:false`. Systemd independently reports its deactivation as
successful at19:33:47 CST. The former transient unit has since been collected;
the default `systemctl show` exitcode on a not-found unit is not the proof of
its successful exit.

The original downstream owner remains PID3221084/birth1960484824. The fresh
owner-ready receipt is `RETRIEVAL_RELAY_OWNER_READY`, with deadline epoch
1792629292.3353126. Its published source maintains the CPU relay and declared
slot admission until its lifetime/drain condition, then drains dependants.
The queued wrapper adjusts slot specifications and calls that same owner.
This code does not itself launch a replacement supplementary systemd service.

The same-host, same-run `systemctl list-jobs` query has no queued start jobs.
There is therefore no observed replacement auxiliary service already queued
on H100-10. Prepared historical submit manifests do not establish a live queued
replacement. The `relay-live-slots/*.json` projection was empty; this is not a
claim that all other nodes or nested lease registries lack active producers.
This check did not inspect other nodes or contact their owning chat.

Overall source seals continued137,250→137,378 questions across the two fresh
observations, so the former auxiliary exit did not stop source-wide output.
The all169,028 source goal remains incomplete. The present124,194-question
training cohort remains fixed. No producer, unit, configuration, launch,
resource allocation, or payload was changed or invoked.

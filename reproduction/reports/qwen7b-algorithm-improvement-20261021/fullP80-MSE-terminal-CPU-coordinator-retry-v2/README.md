Seed2 terminal watcher metadata-read retry v2

Only known waiting-stage observation failures are changed: OSError/JSONDecodeError, sacct timeout/nonzero/unavailable row and temporarily missing successful complete receipt are recorded in observation-errors.jsonl and retried against the samejob. A stale last-successful snapshot is labeled as such. Actual training FAILED/TIMEOUT and other true terminal failures remain fatal before any CPU child. Checkpoint/CPU validation errors outside waiting remain fatal.

Only seed19137/job750190 is stageable here; the independentv2 root preserves oldv1/job-750190/failure.json (SHAb10be9829e3355a6f31af022b8ae6be2d764920d6195c275e77c31686945d088). PID3344791/start1410521 is dead and /proc scans showed no same-job CPUreader/oldPID lock/active-child. Healthyseed1 PID3336295/v1 and all training are unchanged.

The original693 CPU entry, accepted paths, data/model admission and commonCPU mutex remain unchanged. No start is performed in this task: Root is the sole starter after the immutablev2 source stage proof.

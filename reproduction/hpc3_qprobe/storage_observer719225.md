# One retained CPU storage observation for producer719225

This separately published observer measures allocated bytes under the existing
`/data/user/jxu957/experiments` namespace using GNU `du -s -B1`. It does not
follow symbolic links, import tensors/models, change training, submit Slurm
jobs, copy checkpoints, clean directories, or approve goal completion.

Run the pinned published script with the existing JXU CPU Python and `--start`.
Use the same pinned script with `--inspect` for subsequent checks. Both validate
the owner's native UID14090 and the source deployment's complete-file SHA
receipt. Only small claim/process/result/log files are created in the fixed
`ROOT/observations/storage719225-20261015` directory. An existing claim always
means inspect; the entry cannot generate a new nonce or replay that launch.

The detached supervisor records its actual Linux process birth and the direct
du child's PID/start_ticks/SID/UID. A short SSH observation timeout does not
terminate or restart either. The background measurement has an1800-second
runtime bound, following the existing archive guard. Cleanup addresses only
the held direct Popen child. An uninterruptible child is explicitly reported
as pending; it is never represented as cleaned or as successful measurement.

Acceptance requires the actual waited du exit0, empty stderr, exactly one
integer/tab/fixed-root output line, stable owner metadata, source/process
bindings, and output SHA. Missing process without a result remains unknown;
Large failed-output summaries are capped at16KiB each while preserving their
full byte counts and SHA; successful measurements require untruncated output
and actual empty stderr. Truncation never converts failure into success.
PID reuse fails the original-birth check. The result covers a non-atomic walk
while training may grow files. It does not prove a hard filesystem quota,
future2TB allowance, an atomic endpoint, or all200/final recovery completion.

The live producer1843 and followthroughEA remain pinned and unchanged. This
observer does not alter their existing per-archive2TB admission guard.

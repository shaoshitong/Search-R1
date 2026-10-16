# Job727251 thread-limit repair

Actual Ray2.40 startup failed before its driver registered any task: the account
had1995 threads against soft RLIMIT_NPROC2048. Ray automatically detected96 host
CPUs and prestarted workers with51 threads each. The hard limit is100000 and
the node's pids cgroup is unlimited. Actual training updates remain a separate
acceptance gate.

`raise_owned_nproc.py` is a narrow, once-only operator for job727251,
run`a79333e9574742a38f922974e30203e9`, nodeACD1-25, immutable runtime78074b6.
It raises only the soft limit to8192, keeping every actual hard limit. It first
promotes its own CPU diagnostic process so later read-only Slurm checks can
fork after the driver resumes. Other targets must be UID14090 with all four UID
fields equal and exact job727251 step_batch membership in every relevant
cgroup controller. Other steps and jobs are excluded.

Deploy only the committed/pushed operator bytes under
`BASE/preparation/nproc-rescue-code/FULL_PUBLISHED_COMMIT/raise_owned_nproc.py`.
The public plan binds its SHA, the compute-node bootID and four current receipt
SHA values. Run without`--apply` for read-only preflight. Execute once with
`--apply` through a CPU-only overlapping step of the existing allocation; no
additional GPU allocation, signal, worker restart or runtime source edit is
performed. The outer launcher must durably capture the child's actual wait and
exit status.

The operator exclusively claims`preparation/nproc-rescue727251`, retains each
kernel write intent/atomic old values/readback, checks process birth and proc
inode immediately around writes, performs two passes for newly inherited
children and finally checks every surviving target. Any partial result is
preserved and exits nonzero. An existing claim only returns its prior result;
it never retries writes. Existing higher or unlimited soft limits are retained.

CPU fixtures cover scope/birth changes, too-small hard limits, higher soft
limits, partial outcomes, new children and helper-first fork pressure. Kernel
limit verification does not establish Ray recovery or a complete actor/probe
outer. Both require subsequent actual observations. Prospective Ray scheduler
CPU-budget changes are separate and are not applied to this live runtime.

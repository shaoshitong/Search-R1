# Periodic DynRPO: actual Ray RPC thread failure and recovery

Job 724782 completed outer 11 with 220 Adam writes and scheduler outer 11 on all four ranks. On 2026-10-16 at 10:15 CST it failed before generating outer 12, with no optimizer writes in outer 12. The fatal stack was Ray 2.40 `GetServerCallExecutor`, creating a Boost Asio thread pool: `Resource temporarily unavailable [system:11]`.

Ray `num_cpus=16` bounds scheduling resources; it does not set the RPC executor pool. The fixed Ray 2.40 sources use `num_server_call_thread` for this lazy pool and derive its default from host hardware concurrency. The GCS client and server RPC pools also have separate settings:

- [Ray 2.40 server_call.cc](https://github.com/ray-project/ray/blob/ray-2.40.0/src/ray/rpc/server_call.cc)
- [Ray 2.40 ray_config_def.h](https://github.com/ray-project/ray/blob/ray-2.40.0/src/ray/common/ray_config_def.h)

This independent periodic HPC3 experiment sets all three pools to 2 before Python/Ray starts, passes the same cluster system config and worker environment, and verifies worker inheritance. The four GPU learner, scheduler CPU resources, rollout budgets, student objective, probe structure, and periodic update schedule are unchanged.

`ray_thread_smoke.py` is an isolated Slurm CPU workload using the installed Ray 2.40 environment, no models, no Torch and zero CUDA devices. It starts its own Ray instance, requires three real global-GC requests to reach both actor processes, verifies initialized cluster settings and worker thread counts, and shuts down only that instance. Run it in a compute allocation; never on the 8-GiB login cgroup.

The original failed job also reported cleanup failure because the node kernel does not implement pidfds. Slurm reports the parent FAILED and extern COMPLETED. Cleanup must be reconciled on ACD1-19 with the existing owned-process reconciler; a login-node `/proc` observation cannot prove compute-node cleanup. Restore accepts the retained cleanup-error stage only with the original-node reconciliation proof, all owned sessions absent and recorded CUDA contexts absent.

The latest complete recovery pair is outer 10. Outer 11 was completed but its full learner state was not saved. Resuming CP10 would recompute outer 11 and is not lossless continuation. The restore request now rejects any later source Adam writes unless an explicit user authorization is bound to that source job, checkpoint, actual counters and recomputed suffix. No automatic rollback or new training job is authorized by this fix.

Focused validation before publication: six restore-request regressions and two Ray environment/config propagation regressions pass. Real compute-node Ray/GC regression and original-node cleanup are separate acceptance steps; publishing this source does not claim they passed or that resumed training is running.

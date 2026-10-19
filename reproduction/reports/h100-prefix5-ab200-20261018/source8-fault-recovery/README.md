# Source8 NFS fault recovery, October20

The latest human explicitly authorized moving to other available H100 GPUs after source8 was released. This supersedes the earlier healthy-maintenance deferral for this actual fault. Original A/B leaders on h100-11 blocked opening source8 `fixed_test_identity_path.json`; source8 SSH/NFS timed out while NAS10 outputs remained reachable. Latest complete boundaries are A187 and B157, with no updates in A188/B158.

Exact original leader pidfd SIGTERM did not exit the dead-NFS read. Exact owned SIGKILL terminated both native leaders; owned cleanup left kernel-exit tasks/zombies and CUDA contexts on h100-11. These residues must never be described as released, native exit0 or a clean native-finally exit. A cross-node barrier accepts only an absent original leader and two actual observations showing remaining owned tasks are zombies or PF_EXITING D tasks. Ordinary sleeping/running writers are rejected. Linux PF_EXITING tasks are executing do_exit and cannot resume training user code. Their residual resource release remains pending separately.

The new recovery path pins this external barrier to the original source run/config/leader and latest complete joint. It imports the complete actor/probe, both optimizer states, scheduler, rank RNG and actual pool reservation. Placement alone changes to h100-5: A cards0–3, B cards4–7. Actual inventory showed approximately72GiB available per card; existing foreign tasks are preserved. The retained NAS10 wrapper also mounts the already-qualified private corpus cache view, eliminating source8 inputs without global mount changes.

No reset, warmup, older checkpoint fallback, discarded partial updates or scientific change is authorized. Actual CPU source qualification, target launch and restored progress must be recorded separately; this document is not a completion claim.

## First A continuation: shared-GPU vLLM profiling fault

A's first migrated continuation failed before new Adam writes. Installed vLLM0.6.3 computed `peak = initial_global_free - current_global_free`: initial42,912,055,296B/current43,587,010,560B, so another process releasing memory made its estimate negative. Source comments explicitly assume other GPU processes do not change usage. The original actual exit is1 and native owned cleanup completed; B is not stopped or hot-edited.

The A-only recovery enables a process-scoped initializer using the same mandatory profile forward and `torch.cuda.max_memory_allocated`. KV allocation is capped by both `(total*utilization - process_peak)` and `(current_free - transient_activation - existing_reserve)`. It restores the original method after engine construction and leaves sampling/model/attention/PPO/probe recipes unchanged. The three CPU arithmetic boundary cases cover requested-budget binding, free-memory binding and exhausted headroom. No separate GPU test is added. Dynamic foreign allocations can still cause genuine OOM; those keep the existing typed-OOM handling.

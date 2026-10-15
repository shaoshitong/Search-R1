# Full-state reload acceptance after training completion

This isolated c5-derived controller is **not executed or GPU accepted yet**. It
never changes training `bind_restore` (which still rejects CP200), calls `fit`,
selects questions, generates responses, computes loss, backpropagates, or updates
an actor/head optimizer. It does not run warmup, baseline, or quality gates.

After the same run f77c3245dcfe49a88a5a99b1dec161e6 has completed200 and its default
final work, capture the true native terminal **before Slurm purge** using the
published b88 execution controller. Then prove mandatory cleanup and actual
original birth/session/cgroup/GPU release. This controller reuses that pinned
terminal guard; queue absence or SSH failure never establishes termination.
Run only in a separately authorized native h100-8 shared96CPU/512G allocation,
with all8 original UUIDs visible, foreign bindings preserved, and the same
explicit global UUID OFD lock directory. No Slurm submit helper is invoked here.

## Binding (small JSON outside the future output directory)

Required fields:

- `role`: `post_completion_full_state_restore_only`
- `run_id`: `f77c3245dcfe49a88a5a99b1dec161e6`
- `source_runtime_commit`: `c5fc09603a10244655b98ec0ce8eada32668f83b`
- `source_base`: `/mnt/data/shaoshitong/search-g1-qprobe3-binary10-c5fc096-20261015`
- `base`: a new absolute remote output directory that does not yet exist.
- `execution_id`: a fresh32hex ID; its12hex Ray prefix must not already exist.
- `steps`: `[5,50,100,150,200]`
- `receipts`: map of each decimal step to `{"path": <actual source completed-pair source-receipt.json>, "sha256": <actual receipt SHA>}`. No fabricated future receipts.
- `resolved_config`: the original source `logs/resolved-training-config.json`; `resolved_config_sha256`: its actual SHA.
- `trainer_terminal_proof` and `trainer_terminal_proof_sha256`: the independently pinned b88 actual terminal capture.
- `execution_control_proof_sha256`: actual SHA of `/mnt/data/shaoshitong/search-g1-compatible-eval-b88c715-20261015/publication-proof.json` (published1038Git blobs).
- `physical_gpu_uuids`: the8 ordered original UUIDs.
- `allocation_mode`: `shared`; `cpu_request`: `96`; `memory_request`: `512G`.
- `forbidden_training_job_ids`: `["3402"]`.
- `preserved_foreign_allocations`, `preserved_foreign_gpu_processes`: actual launch-time allowed bindings, with the existing b88 native-admission schema. Changed cohosts require a new immutable binding, never an edit/replay of occupied output.
- `gpu_lock_directory`, `gpu_lock_filename`: existing global lock path/template identical to the run's original UUID leases.

The b88 proof is an execution dependency only. All actor/probe/math modules come
from this isolated c5-derived checkout. Both controller commits must be recorded
at publication; no old evaluation source/recipe is substituted for the BF16
actor + FP32 master/Adam restore.

## Future invocation (not run)

Inside the authorized native allocation, with visible0..7 and a published
controller source tree on PYTHONPATH:

```
PYTHONPATH=<published restore-only control> <existing remote env>/bin/python \
  -m reproduction.restore_only.controller --binding <immutable small binding.json>
```

`--child-step` is internal. It requires the live coordinator's actual parent
birth, pinned binding and actual same-job native admission; it is not an
independent restore launch shortcut. The outer output namespace is once-only:
any occupied namespace refuses replay, including failed/partially written runs.
This code provides no automatic restart or silent result reuse.

## Exact coverage

For each point, a separate8rank actor_rollout engine loads the actual HF model
and saved rank model/master Adam/scheduler/RNG. Rank0 compares every HF tensor;
all8ranks compare saved actual Adam writes, schedulerN and complete boundary RNG.
The original actual counter is authoritative (not opportunities or rank sums).
The frozen H1008 localmicro4/globalmicro32/accumulation4 budget and BF16 varlen runtime/FP32 master Adam and immutable official fixed-ref binding
are retained. Fixed reference is not trained or needlessly loaded as another model.

Driver restore materializes the actual169028-row candidate stream and compares
permutation/cursor/epoch/consumed/pending state; selection RNG and completed
attempts are compared on the restored live runtime. Rank7 restores the actual
probe model/optimizer/privateRNG/replay and sends a **fresh state_dict RPC** back
for full-tree equality. The input payload or load acknowledgement is not proof.
All8actor Adam/scheduler/boundaryRNG values are read back again after head restore,
without restoring them a second time. No next outer (including201) is executed.

Each child is owned by birth/session. Actual child process/GPU release is required
before the next point; foreign reused PIDs are distinguished by birth. Final
`restore-only-complete.json` is emitted only after all5 point results and owned
cleanup. It explicitly leaves native final exit0/cgroup release to a post-exit
observer; a live leader result does not prove native resource closure.

Focused CPU fixtures do not establish real CUDA/FSDP/engine reload acceptance.

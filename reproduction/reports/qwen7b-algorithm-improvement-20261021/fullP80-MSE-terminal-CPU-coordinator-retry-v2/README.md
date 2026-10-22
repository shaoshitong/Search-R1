Seed2 terminal watcher metadata-read retry v2

Only known waiting-stage observation failures are changed: OSError/JSONDecodeError, sacct timeout/nonzero/unavailable row and temporarily missing successful complete receipt are recorded in observation-errors.jsonl and retried against the samejob. A stale last-successful snapshot is labeled as such. Actual training FAILED/TIMEOUT and other true terminal failures remain fatal before any CPU child. Checkpoint/CPU validation errors outside waiting remain fatal.

Only seed19137/job750190 is stageable here; the independentv2 root preserves oldv1/job-750190/failure.json (SHAb10be9829e3355a6f31af022b8ae6be2d764920d6195c275e77c31686945d088). PID3344791/start1410521 is dead and /proc scans showed no same-job CPUreader/oldPID lock/active-child. Healthyseed1 PID3336295/v1 and all training are unchanged.

The original693 CPU entry, accepted paths, data/model admission and commonCPU mutex remain unchanged. No start is performed in this task: Root is the sole starter after the immutablev2 source stage proof.

Actualv2 source669427f37da01021756df18110cfb2b56fa1e632 (published/remoteSHA verified), six regression tests passed.

Root start from repository cwd:

```sh
python3 reproduction/operations/qwen7b_warmup/deploy_fullP80_MSE_terminal_CPU_coordinators.py --commit 669427f37da01021756df18110cfb2b56fa1e632 --action start --seed 19137
```

Status: same helper/commit with --action status --seed19137. Do not use thisv2 helper for healthyseed9137 (CLI rejects it).

worker: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fullP80-MSE-terminal-CPU-coordinators-20261023-v2/code-669427f37da0/fullP80_MSE_terminal_CPU_coordinator.py; SHA256 5304f9e9e290820fdde3123f0e34ab98b925f8ecd79a6cda011113ad56ca23ec

config: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fullP80-MSE-terminal-CPU-coordinators-20261023-v2/code-669427f37da0/job-750190.json; SHA256 c992b3aeaa4b3e6b8e969b6e1c83badba3cbc99132ed3b76299a09e5f7be4a48

coordinator_entry: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fullP80-MSE-terminal-CPU-coordinators-20261023-v2/code-669427f37da0/job-750190.sh; SHA256 4426038e97adbce47338759521412837ddf988a7a4e959d242f145f607deb16f

Independent readback confirmed allsource/config/entry/693 hashes and preservedoldfailure, while originalseed1 PID3336295/start1408487 remains alive. No worker was started by this preparation.

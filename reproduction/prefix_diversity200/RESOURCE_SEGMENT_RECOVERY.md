# Passive telemetry recovery for the unchanged 3433 training job

Actual unit readback proves `g1-resource5s-4a1cc2d500fe-3433` is failed with `Result=oom-kill`, `MainPID=0`, `ExecMainStatus=9`. Its 128 MiB cgroup is independent from training. The old summary ends at outer93 (Unix 1792256248.012871); the real training job has continued normally. This proves the sampler failure; process-ledger growth is a plausible memory driver, not a measured allocation profile.

Keep all `preparation/resource5s-v1` artifacts unchanged. The already authorized passive telemetry continues in `preparation/resource5s-v2`, unique unit `g1-resource5s-4a1cc2d500fe-3433-v2`. Deployment requires actual old sampler termination, current same-run native allocation, an absent new unit and one immutable intent/receipt. Repeating a deploy against an existing directory only reads its receipt/state; it never submits twice. An unknown outcome must be reconciled.

The v2 unit has CPUQuota10%, MemoryMax1G, Nice19, TasksMax32, CUDA_VISIBLE_DEVICES empty. It reads GPU telemetry with a requested5-second interval and scans the owned process ledger every60seconds to reduce observer work. Actual intervals can be longer and must be reported as observed; this is not continuous peak measurement. It never signals or modifies training/foreign processes, performs model forwards, or reads model weights.

Observer aggregation validates run/base/job identity, preserves per-segment source SHA/bytes, combines whole-card peaks across sampled windows, and explicitly records the unsampled gap. Same-job CPU lower bounds from different segments may include the same process lifetime: take their maximum, never sum them. Distinct native3432/3433 allocations retain their existing separate accounting. CPU remains incomplete; a gap is not zero consumption. The scalar streams and all large data stay remote.

After source commit/push and minimal offline mocked checks, use the existing control.py transport via resource_control (established_controller=True), once:

```
python3 -m reproduction.prefix_diversity200.resource_control deploy --segment resource5s-v2 --config <current CURRENT_CONFIG.json> --receipts <current output directory>
```

Then query the same segment/unit twice and verify actual sample timestamps/counts/stream bytes grow, no error/exit, and original3433 training continues. Reconcile any failed attempt before another action. Do not reset failedv1 or restart training.

Fault evidence: `control_evidence/RESOURCE_V1_OOM_READBACK.json`.

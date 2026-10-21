# One-shot official Mixed job observation

Published observer source: `2c632a1ddc01f08c68bf2ff221ce94adba59ecb9` on `codex/mixed-direct-official-worker-20261022`.

```bash
python3 -m reproduction.operations.qwen7b_warmup.observe_mixed_direct_progress \
  --control-root /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5
```

Each invocation reads the actual submission ID, verifies live Slurm UserId/Account/job name/Command/WorkDir and config SHA, and returns Pending reason/start estimate or a single bounded CPU1/GPU0 allocation step. The step reads the exact controller PID's command/cgroup/birth, RSS/CPU ticks and IO counters using stdlib. It does not send signals, submit, retry, create a remote watcher, load tensors or change training. One small local observation and `latest.json` are saved; the next invocation compares only the same verified host/job/PID/birth. I/O growth is reported separately from accepted update/complete-checkpoint progress. Changed PID is an observation, not an automated fatal condition.

The two actual reads of job746083 observed the same ACD1-25/PID3521320/birth841056146. Main rchar increased15.531GB→26.300GB in103.20s (+10.769GB/99.52MiB/s), with CPU growth and no failure; no accepted update or checkpoint existed yet. This confirms active source/feature initialization, not healthy training or a completed optimizer update. The script requires all first-update/CP1/next-update proofs and at least six accepted updates before its healthy qualification flag can become true.

See `FINAL.json` and the two timestamped observations for exact identities and read counters. No continuing observer was started.

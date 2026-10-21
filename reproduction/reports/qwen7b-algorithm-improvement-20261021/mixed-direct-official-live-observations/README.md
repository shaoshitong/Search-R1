# One-shot official Mixed job observation

Published observer source: `2c632a1ddc01f08c68bf2ff221ce94adba59ecb9` on `codex/mixed-direct-official-worker-20261022`.

```bash
python3 -m reproduction.operations.qwen7b_warmup.observe_mixed_direct_progress \
  --control-root /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5
```

Each invocation reads the actual submission ID, verifies live Slurm UserId/Account/job name/Command/WorkDir and config SHA, and returns Pending reason/start estimate or a single bounded CPU1/GPU0 allocation step. The step reads the exact controller PID's command/cgroup/birth, RSS/CPU ticks and IO counters using stdlib. It does not send signals, submit, retry, create a remote watcher, load tensors or change training. One small local observation and `latest.json` are saved; the next invocation compares only the same verified host/job/PID/birth. I/O growth is reported separately from accepted update/complete-checkpoint progress. Changed PID is an observation, not an automated fatal condition.

The two actual reads of job746083 observed the same ACD1-25/PID3521320/birth841056146. Main rchar increased15.531GB→26.300GB in103.20s (+10.769GB/99.52MiB/s), with CPU growth and no failure; no accepted update or checkpoint existed yet. This confirms active source/feature initialization, not healthy training or a completed optimizer update. The script requires all first-update/CP1/next-update proofs and at least six accepted updates before its healthy qualification flag can become true.

See `FINAL.json` and the two timestamped observations for exact identities and read counters. No continuing observer was started.

## Current manifest position diagnostic

Observer source `0c97d33f11ae519c1f4cb054e350a9d9026a765e` adds an optional readlink-only scan of the verified main PID's descriptors. Only `.bin`, `.jsonl`, and `.safetensors` absolute paths inside the declared experiment root are retained. Descriptor close races/errors are counted and never become training failures. It freshly verifies the1.54MB config-bound list of5356 manifest descriptors and matches an open file's parent to that list. An ambiguous/no-match snapshot reports no current ordinal. It does not open tensor payloads or infer how many prior bytes/files were completed.

The actual new snapshot found current ordinal2060/5356, reading a `policy.bin` file. Same PID/birth main rchar was119.311GB and had increased34.250GB since the preceding root observation; there were still no accepted optimizer updates or checkpoint. See `FD-OBSERVATION.json` for exact metadata/source bindings. The command is unchanged and remains one shot.

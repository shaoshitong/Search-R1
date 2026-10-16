# Prefix20 native4 checkpoint archive controller

This controller is independently versioned and deployed beside the immutable training runtime. CPU tests use constructed byte files and subprocesses; they do not establish actual checkpoint transport, GPU recovery, native51713 acceptance, or full experiment completion. Actual launch/progress/exit receipts are separate from this implementation document.

## Current binding supplied by the parent

The latest planned training is job **727148**, base **`/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727050-20261016`**, runtime commit **`f6e2f268128a888a6abbc7489a02871fe28c0178`**. At handoff it is pending resources and has no run ID. The controller has no hard-coded training job/base/commit defaults. It waits for the actual prepared native4 binding and records its run ID once. Missing run ID or queued resources cannot produce completion.

The training commit and archive implementation commit are separate. Publish the archive implementation in a **new** full Git commit; do not hot-edit the healthy training repo/source. The launcher deploys only an independent small sidecar under `base/preparation/archive-control/code/<archive-code-commit>`.

## Integration and call

Copy these four runtime modules into `reproduction/prefix20_joint4/` in the source repository, review, commit, push and verify the remote SHA:

- `checkpoint_archive.py` — copied unchanged from the reviewed v2 core.
- `archive_supervisor.py` — strict routing, held source operations, actual child exits and HKUST3 controller supervision.
- `archive_controller.py` — source polling, exact transport, dependency links, archive readback, second source acceptance.
- `archive_launcher.py` — local once-launch/status entry using the pinned modern existing bridge.

Also include `test_checkpoint_archive.py` and `test_archive_controller.py`; they are self-contained in this folder. Direct-file and package imports are supported. On the local handoff:

```bash
python3 -B -m unittest test_archive_controller test_checkpoint_archive -v
```

After the integration is committed and pushed, launch once with the following explicit arguments. Replace `FULL_PUSHED_ARCHIVE_CODE_SHA` with the exact new 40-character implementation commit. The runtime commit stays the original training SHA below. This example has **not** been executed:

```bash
python3 -B /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/reproduction/prefix20_joint4/archive_launcher.py launch \
  --repo /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016 \
  --bridge-source /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py \
  --code-commit FULL_PUSHED_ARCHIVE_CODE_SHA \
  --training-commit f6e2f268128a888a6abbc7489a02871fe28c0178 \
  --base /data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727050-20261016 \
  --source-job 727148 \
  --output /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/archive727148-launch.json
```

Query with the identical arguments and `status` in place of `launch`. A repeated `launch` with an existing local claim automatically sends **status only**, even if the current source checkout has since advanced. It does not reread the HPC3 password or dispatch another controller. Each local observation is preserved in an immutable timestamped JSON receipt, with a separate latest-status file.

The launcher checks the exact current Git HEAD, a clean tree, the remote branch SHA, committed bytes for all four runtime files, and modern bridge SHA `a75a6a302f4ffed002be36371dc193048cdfcc1a016d99ae98acdbfb101a915f`. The existing protected jxu957 credential memory is read only when a genuinely new controller is dispatched. Credentials go through stdin and process RAM; never into a plan, file, argv, receipt, or source archive. Existing SSH aliases, keys, auth.json and network settings are unchanged.

## Actual execution chain

1. Local entry installs only small committed code on actual HKUST3 `xielab3`, then dispatches a held controller supervisor. One persistent job-scoped claim and independent locks prevent duplicates. Linux host/boot/PID/start ticks/session/UID births identify the controller and supervisors. The supervisor calls actual `Popen.wait()` and records `controller-exit.json`; a self-written success marker without an actual zero exit and all five steps is insufficient.
2. HKUST3 installs the same small sidecar on pinned HPC3 `ACD-Login-1`, IP `10.120.48.31`, boot `a27512a3-24a0-4ca3-aa92-98d2a5256d9a`, PID namespace `pid:[4026531836]`. Every SSH call uses strict existing host keys and the same kernel guard as the modern account bridge. Host identity changes fail closed; no gateway fallback or silent pin update.
3. The controller observes only small prepared/run/submission metadata plus Slurm accounting/queue. It binds the actual same-job jxu957 native4/48CPU/512GiB/emergency_acd/formal-user proof and full training commit. Source absence or failed observations stay unknown. A proven terminal job without a required completed pair records incomplete, never complete.
4. At each fixed step 5/50/100/150/200, a source-side held supervisor runs the long CPU snapshot. It persists a public plan, once claim, supervisor/worker births, stdout/stderr, actual worker exit code and SHA-bound result. SSH observation timeout triggers **queries of the existing claim/process/progress**, never another snapshot. Existing incomplete source claims or unknown dead supervisors require inspection.
5. Only the snapshot's exact file graph is transferred directly HPC3 → HKUST3. Each source reader validates its snapshot SHA, selected file index, safe path and actual file identity, emits raw bytes and exits zero only after full source SHA. The receiver counts bytes actually read/written, hashes and fsyncs them, checks exact length and actual SSH exit zero, then publishes hardlinks. No sparse stat-size progress and no Mac asset staging.
6. Frozen model/data/initial-probe dependencies share a content-addressed regular-file blob store on HKUST3. Each reused blob is freshly fully hashed and fsynced before linking. All actor/probe states remain distinct copies. The controller checks actual free storage against the new allocated upper bound before each milestone. Verified staging/partial links and payload links share inodes, so retained transport evidence adds no duplicate tensor allocation.
7. Only after every required stream has actual exit zero or an explicitly recorded verified dependency reuse does the controller run HKUST3 full archive readback, exact file-set checks and fsync. Then a second held HPC3 operation runs core `accept_archive`, fully reading every original source/dependency file again. Per-file successful source SHA increments are journaled during this second pass. The five HPC3 original sources remain in place.
8. Completion requires all five ordered source/archive/source receipts including200 and the actual controller child exit zero. Status is scoped to checkpoint byte archives. `actual_cuda_restore_accepted` and `final_native51713_accepted` remain false. No training start/cancel/fit/generate/update/save or GPU request occurs anywhere in this controller.

## Durable locations and progress

- HKUST3 controller: `/backup/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/archive-controller/job-<actual-job>/`.
- HKUST3 payloads: `/backup/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/checkpoints/<actual-run-id>/global_step_<S>/payload/`.
- HKUST3 shared immutable dependencies: the run's `immutable-dependencies/<sha256>` directory.
- HPC3 held operations: `<base>/preparation/archive-control/operations/<archive-code-commit>/<actual-run-id>/step-<S>-snapshot|accept/`.
- HPC3 final byte receipts: `<base>/preparation/checkpoint-archives/global_step_<S>/complete.json`.

Read `progress.json`, `events.jsonl`, `controller-birth.json`, `controller-supervisor-birth.json`, `controller-exit.json`, per-step operation receipts and each archive's `transport/progress.json` / `archive-progress.jsonl`. Source operations expose actual `source-progress.jsonl` or `source-readback-progress.jsonl`. A live PID or RUNNING status is not acceptance. Verify at least two measured byte/file increments and terminal exit receipts. Unchanged waiting state produces no repeated progress event; SSH observation failures have a separate durable journal.

## Failure and restoration limits

Transport stall/oversize/short read/nonzero SSH exit preserves its partial file and exit receipt and cannot publish a payload. The code only terminates its own local SSH reader on a measured transport stall; source snapshots and accept workers are never cancelled due to observation timeouts. This version deliberately has no generic automatic retry of terminal/unknown controller claims. A later recovery must first inspect exact births, actual exits, source operation claims and retained partials; use an explicit reviewed recovery path, never delete claims or blindly replay a source snapshot. A source-side code-install timeout is also unknown and requires inspection of the existing sidecar directory.

The core creates no HPC3 archive or restore target. A separate actual native4 restore controller must enforce at most one HPC3 target, include newly materialized data/encoder bytes in the 2TB budget, and map original serialized absolute paths to verified target copies so retained original sources cannot mask a broken restore. Final200 CUDA acceptance requires actual actor model/Adam/scheduler/RNG, four probe model/Adam/private RNG states at2000 steps, selector/eligible mapping/candidate pending cursor/driver RNG/data equality, two readbacks, and no update201. None is inferred from this archive's SHA status.

# Actual checkpoint50 CPU acceptance

Result: `CPU_ACCEPTANCE_PASS` for run `4a1cc2d500fe40979ef8ca4792959438`, outer 50, audit `v2`.
This directory contains only small audit metadata. Model, optimizer and rollout payloads remain remote.

## Execution identity

- Frozen training source: `33ef232e605f7864086d1685d9a4d8ea0d67f4ab`.
- Runtime launch-config SHA256: `d2faa73dc70e741700323f0a3c3846abd09b560aa9af4c2a9a07b7b27aa189a1`.
- Immutable joint50 SHA256: `d1839fbc9dce647a6ee6256e57545b75f1e7d5e006be364f0b81de3d82dc0b8f`.
- Executed helper commit: `a85e351090cc89399b3169689788f81318549c1b`.
- Executed reader SHA256: `58264e45858f68603d93ba5badfdbdb4406b4facd0c0976014de2cf1f278570a`.
- Controller adapter publication: `24af78b189f27acda74c869aa0240c42d7c05fa8`.
- Actual proof SHA256: `4497bcfc73c028ac571e5d526ac538a32fcd4d479123917b06a8a8d9fcda562e`; the exact 50,805-byte JSON is `PROOF-v2.json`.
- Remote proof: `/mnt/data/shaoshitong/search-g1-prefix-diversity200-20261017-4a1cc2d500fe-calibration1/preparation/joint-milestone-0050-independent-full-sha-cpu-once-v2.json`.
- Remote control: `/mnt/data/shaoshitong/search-g1-prefix-diversity200-20261017-4a1cc2d500fe-calibration1/acceptance50-cpu-control-v2`.

The commit publishing this documentation is a separate record. It is not the executed helper commit; the executed identity remains `a85e351090cc89399b3169689788f81318549c1b`. Post-push documentation publication SHA/readback is also saved in the local acceptance50 `DOCUMENTATION_PUBLICATION.json`.

## Preserved v1 failure and explicit v2 correction

The first CPU attempt exited with code 1 after 1.6434 seconds because the old verifier incorrectly requested `scheduler_completed_outer` from `joint-resume-receipt.json` at the then-current `checkpoint_i1i6_cpu.py:95`. The 33ef import result stores actor/probe Adam counters there; scheduler is in `joint-source-terminal-proof.json` and the original completed joint actor receipt.

The correction checks terminal scheduler 6 against the original source joint 6 actor scheduler, actor/probe completed outer 6 and all 8 × 120 Adam counters. It also checks original joint/config/terminal SHA bindings. It does not remove scheduler validation.

The v1 proof SHA256 remains `5745c3c7752971f3a654648579a06697b2007d7358af0bdb4f49efadf9214c3f`. Its intent, failed proof, deployment records and 657-byte single-line failure log remain unchanged in `acceptance50-cpu-control-v1` and the original `once-v1` preparation paths. `V1_FAILURE_ANALYSIS.json` records this history.

The pinned v1 reader source fails at this metadata check before source/split/native/HF/Torch payload scans. Its known source/helper SHA, exact error, one-line log, terminal failure receipt, resource measurements and absent original reader birth were verified before v2. Thus v2 did not repeat an already completed large-file read.

V2 has a separate control directory, systemd unit, `once-v2` proof, intent and lock. The explicit correction configuration binds the old failure SHA; generic failed audits are ineligible. No failed proof was overwritten or relabelled PASS.

## Actual progress and terminal evidence

Two observations preceded terminal completion:

| Remote observed Unix time | Log bytes | CPU seconds | Actual stage |
| --- | ---: | ---: | --- |
|1792245975.6111073|4721|36.593165|435 HF FP32 tensors complete; actor rank0 CPU read begins|
|1792246027.1942277|8493|88.171056|All8 actor rank files independently SHA-checked and CPU-loaded|

Terminal child exit code 0 and systemd inactive/dead, PID 0 / ExecMainStatus 0 were observed. Wall time 91.957155 seconds, child CPU 91.938502 seconds, max RSS 4,543,792 KiB. Actual execution limits were 1 CPU, 12 GiB, Nice 19 and empty CUDA visibility, verified by the runner and active-unit observations. Unloaded terminal-unit limit fields do not describe the earlier execution limits.

## Accepted checks

- Actor: all 8 rank-local optimizer states, 37 Adam states per rank, 1000 actual Adam writes per rank, scheduler outer 50, saved Python/NumPy/Torch CPU and CUDA RNG bytes.
- Probe: 14 FP32 model/Adam states, all 8 rank RNG states, 1000 writes per rank; all 1024 questions / 5120 trajectories consumed in 20 synchronous writes, with original rank-weighted true loss means.
- HF/tokenizer: 435 finite FP32 tensors and pinned tokenizer semantics. Five selected small initial tensors contain 6604 nonzero differences among 6656 compared elements; max absolute difference 5.156343104317784e-5. No second learned-HF traversal and no repeated official full-weight SHA scan.
- Original full 1024 G5 native records and actor 512 subset are equal under the original prefix/outcome identity; deterministic strict-EM variance Top512 selection and all 8 logical-pair objectives / 20 finite gradient-loss writes / 40 monitors are verified.
- Split/source/config identity, original fresh-run continuation at joint 6, current execution-base driver metadata, and CP50 actual round0 reward attempt/manifest/adoption binding are verified.

This is CPU deserialization and scientific-state acceptance. It does not establish exact CUDA executor restore or an independent backup. Parameter change was compared for five small selected tensors. No training restart, additional rollout or GPU/QA evaluation was performed. Standalone QA50 is a separately accepted result.

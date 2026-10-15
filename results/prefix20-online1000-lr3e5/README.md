# Prefix20 resumed online training at LR 3e-5

Job 3409 on h100-8 restored complete epoch 7 (700 actual Adam updates on every rank) and completed fresh epochs 8–12. Every one of the eight ranks recorded exactly 100 updates per epoch, advancing from 701 through 1200, all at actual optimizer LR 3e-5. Epoch 13 had actual generation progress when the collector ran. The same execution continues toward 1000 epochs; this is initial resumed-run acceptance, not experiment completion.

The user authorized the LR change from 3e-4 to 3e-5. The isolated runtime is `4bc683883a569d7feb9d5da87307612b872e8f3a`, with the same logical run `8a6b24dc19664cb3a3fb97101f21c54e`. The source CP7 full SHA and CPU restore preparation proved model, Adam moments/step and eight-rank RNG state. E8 optimizer-load metadata records old LR 3e-4, followed by the actual 3e-5 override; all eight update ledgers independently record the target LR. E9–12 load the already changed LR without another override.

Each fresh epoch contains 5120 distinct questions × G5 = 25600 complete trajectories, natural sampling without replacement, global batch 256 / local batch 32, no accumulation and 100 Adam updates per rank. Epochs 8–12 contain 25600 distinct questions / 128000 trajectories, with no selected-question or normalized-content overlap across these epochs or with historical epochs 1–7. Selected question IDs also do not overlap the fixed diagnostic. The fixed diagnostic remains 512 questions × G5 = 2560 trajectories, 559 positive and 2001 negative; its manifest is unchanged. The collector documents the boundary of normalized diagnostic exclusion checks.

## Fixed diagnostic results

| Epoch | BCE | AUROC | Raw accuracy | Balanced accuracy | Positive recall |
| --- | --- | --- | --- | --- | --- |
| 7 | 0.608796 | 0.650932 | 77.9688% | 50.5841% | 1.9678% |
| 8 | 0.503421 | 0.678168 | 77.7734% | 54.5845% | 13.4168% |
| 9 | 0.502067 | 0.684116 | 77.8516% | 53.9899% | 11.6279% |
| 10 | 0.501740 | 0.684018 | 76.8750% | 57.0393% | 21.8247% |
| 11 | 0.501570 | 0.690305 | 78.2422% | 53.0795% | 8.4079% |
| 12 | 0.494035 | 0.696017 | 78.0469% | 53.2124% | 9.1234% |

These are fixed diagnostic results at threshold 0.5. Epoch 7 used the earlier LR; epochs 8–12 use 3e-5. Fresh training examples and additional updates also change, so this comparison does not isolate a causal effect of LR. Majority-class accuracy is 78.1640625%; accuracy alone does not establish useful positive detection. Historical epoch 1–7 metrics remain in the facts record and are not counted as the five resumed fresh epochs.

## Evidence and storage boundaries

`FIVE-ACTUAL-FACTS.json` contains independent small-file SHA readbacks of completed rollout receipts, selections/questions, per-rank update ledgers, metrics and pool boundaries, plus stage wall times. Pipeline stage wall time includes startup, polling, cleanup and publication; it differs from the head update and diagnostic loop timings. Complete rollout and compact-prefix payloads stay at the recorded remote paths; only their metadata/stat and producer hashes are archived here.

`CHECKPOINT10-METADATA.json` now includes subsequent independent verification of the immutable CP10 file (925542230 bytes): full-file SHA256 `9cadb713c580874e4c7bb2d19a05bb3d1f0148e0e6ece2a988cbbf1e2c4db43a`, stable file metadata and actual CPU deserialization. `CHECKPOINT10-INDEPENDENT-CPU-SUMMARY.json` records epoch 10, all eight rank counters at 1000, actual optimizer LR 3e-5, 14 FP32 model tensors, 14 Adam parameter states with FP32 moments and step 1000, eight rank RNG records, and matching pool/bundle hashes (cursor 51200 / next epoch 11). The remote proof is `preparation/checkpoint10-independent-full-sha-cpu-once-v1.json`, SHA256 `a7329c174236f8d9ec1a027a1ab7b623d00c6d584d045a3f658db8734aed1eb3` under the recorded run base.

This CPU inspection is not a GPU semantic forward or independent backup. `FIVE-ACTUAL-FACTS.json` remains unchanged: it accurately records the earlier producer-only observation. Later immutable checkpoints 20–1000 are still to be produced; neither this saved-point verification nor five-round acceptance completes the 1000-epoch task. No large payload, checkpoint, tensor or question text was copied into Git or to Mac. The consumed checkpoint CPU audit controller is archived with the other code backups.

The two scripts in `reproduction/prefix20_online/code_backup/` are archival read-only controllers. The five-epoch collector was consumed once; do not blindly replay it. The one-off current observer is also read-only. Neither changes training, state or automation. Old stop-v1/v2 and restore preparation evidence remain preserved separately; no old submission or stop is replayed by this publication.


## Later normal savepoint inspection

The archived `read_prefix20_immutable_checkpoint_cpu_once.py` supports a normal immutable savepoint such as `--epoch 20`, only after its actual checkpoint, paired bundle and completed immutable receipt exist. It derives the current run/base/runtime and LR from the current local state, reads immutable pool/selection bindings including later pool cycles, and checks complete-file SHA plus CPU model/Adam/RNG state. Missing savepoint files or an unfinished immutable receipt return PENDING without publishing a success proof. Existing successful once proofs, including the original CP10 proof, are inspected without replaying the large-file read/load. This controller has not been run for CP20 at this publication; it performs no GPU forward and is not a training gate. Checkpoints 20–1000 remain future work while the current execution continues.


## Actual epoch 20 savepoint

The same job 3409 completed 20/1000 epochs, with 2000 actual Adam updates on each of eight ranks and actual LR 3e-5, and continues into epoch 21. Runtime `4bc683883a569d7feb9d5da87307612b872e8f3a` is unchanged. On the same fixed 512-question × G5 diagnostic, epoch 20 raw accuracy is 0.781640625, BCE 0.488067568242712, AUROC 0.6960455371598637, balanced accuracy 0.5502771869879014 and positive recall 0.13953488372093023. Raw accuracy equals the majority-class baseline and does not establish quality success. Training online accuracy is 0.787109375 and BCE 0.47916799092665313: these are one-pass predictions before each sample's own batch update, not a post-epoch evaluation of the full training set. Exact small metrics are in `EPOCH20-QUALITY-SUMMARY.json`.

`CHECKPOINT20-INDEPENDENT-CPU-SUMMARY.json` records independent full-file SHA256 and actual CPU loading of immutable CP20: 925542230 bytes, SHA256 `d6c6d4d83361acd4b69b2e237969cda4aec4fe0257bed065c41f64bbbd6a40ab`. Model/Adam state, all eight rank RNG records and the pool bundle passed the recorded inspection; pool next epoch 21, cursor 102400, cycle 0. The remote proof `preparation/checkpoint0020-independent-full-sha-cpu-once-v1.json` has SHA256 `05a87a7d1b4a24867888f2f81a013687b4525c820a58a364a4439c01d3017ff1`. This is not a GPU semantic forward or independent backup. CP30–1000 and the overall goal remain incomplete. The earlier five-epoch and checkpoint observations remain unchanged historical records.

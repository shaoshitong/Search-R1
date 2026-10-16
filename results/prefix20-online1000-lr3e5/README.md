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

This CPU inspection is not a GPU semantic forward or independent backup. `FIVE-ACTUAL-FACTS.json` remains unchanged: it accurately records the earlier producer-only observation. At the CP10 audit, later immutable checkpoints 20–1000 were still to be produced; neither this saved-point verification nor five-round acceptance completes the 1000-epoch task. No large payload, checkpoint, tensor or question text was copied into Git or to Mac. The consumed checkpoint CPU audit controller is archived with the other code backups.

The two scripts in `reproduction/prefix20_online/code_backup/` are archival read-only controllers. The five-epoch collector was consumed once; do not blindly replay it. The one-off current observer is also read-only. Neither changes training, state or automation. Old stop-v1/v2 and restore preparation evidence remain preserved separately; no old submission or stop is replayed by this publication.


## Later normal savepoint inspection

The archived `read_prefix20_immutable_checkpoint_cpu_once.py` supports a normal immutable savepoint such as `--epoch 20`, only after its actual checkpoint, paired bundle and completed immutable receipt exist. It derives the current run/base/runtime and LR from the current local state, reads immutable pool/selection bindings including later pool cycles, and checks complete-file SHA plus CPU model/Adam/RNG state. Missing savepoint files or an unfinished immutable receipt return PENDING without publishing a success proof. Existing successful once proofs, including the original CP10 proof, are inspected without replaying the large-file read/load. At the controller-only publication, this controller had not yet been run for CP20; it performs no GPU forward and is not a training gate. At that controller-only publication, checkpoints 20–1000 remained future work; subsequent CP20 and CP30 results are recorded below.


## Actual epoch 20 savepoint

The same job 3409 completed 20/1000 epochs, with 2000 actual Adam updates on each of eight ranks and actual LR 3e-5, and continues into epoch 21. Runtime `4bc683883a569d7feb9d5da87307612b872e8f3a` is unchanged. On the same fixed 512-question × G5 diagnostic, epoch 20 raw accuracy is 0.781640625, BCE 0.488067568242712, AUROC 0.6960455371598637, balanced accuracy 0.5502771869879014 and positive recall 0.13953488372093023. Raw accuracy equals the majority-class baseline and does not establish quality success. Training online accuracy is 0.787109375 and BCE 0.47916799092665313: these are one-pass predictions before each sample's own batch update, not a post-epoch evaluation of the full training set. Exact small metrics are in `EPOCH20-QUALITY-SUMMARY.json`.

`CHECKPOINT20-INDEPENDENT-CPU-SUMMARY.json` records independent full-file SHA256 and actual CPU loading of immutable CP20: 925542230 bytes, SHA256 `d6c6d4d83361acd4b69b2e237969cda4aec4fe0257bed065c41f64bbbd6a40ab`. Model/Adam state, all eight rank RNG records and the pool bundle passed the recorded inspection; pool next epoch 21, cursor 102400, cycle 0. The remote proof `preparation/checkpoint0020-independent-full-sha-cpu-once-v1.json` has SHA256 `05a87a7d1b4a24867888f2f81a013687b4525c820a58a364a4439c01d3017ff1`. This is not a GPU semantic forward or independent backup. At the epoch 20 observation, CP30–1000 and the overall goal remained incomplete. The earlier five-epoch and checkpoint observations remain unchanged historical records.


## Actual epoch 30 savepoint

Job 3409 subsequently completed 30/1000 epochs, with 3000 actual Adam updates on every one of eight ranks at actual LR 3e-5. Epoch 31 had 6900 actual generated trajectories at the saved observation; the same runtime `4bc683883a569d7feb9d5da87307612b872e8f3a` continues unchanged. The fixed 512-question × G5 diagnostic at epoch 30 has raw accuracy 0.779296875, BCE 0.47851565673745255, AUROC 0.7129927880424725, balanced accuracy 0.5449104606909425 and positive recall 0.12880143112701253. Accuracy is below the majority baseline 0.781640625; these measures do not establish quality success or an LR-only causal effect. Training online accuracy is 0.79453125 and BCE 0.46518513534218076, measured before each sample's own batch update rather than by post-epoch training-set reevaluation. `EPOCH30-QUALITY-SUMMARY.json` retains the exact metrics and observation binding.

`CHECKPOINT30-INDEPENDENT-CPU-SUMMARY.json` records independent complete-file SHA and actual CPU state inspection: CP30 is 925542230 bytes, SHA256 `c172c5149f2320d72f9abcf9377a8cbb4205c70d45868f0e1d37de352f01fa5c`. The remote `preparation/checkpoint0030-independent-full-sha-cpu-once-v1.json` proof has SHA256 `cbce77b3bc336622fb60b27d4c100552bce8c518f5357238bdf1f2a6fa31010f`. This does not claim GPU semantic forward or an independent backup. At the E30 observation, later CP40–1000 and the overall goal remained incomplete. Earlier immutable facts JSON is preserved unchanged.


## First pool-cycle selection boundary

The same job 3409 completed epoch 32 with 3200 actual Adam updates on each rank and entered epoch 33 rollout. At this observation, epoch 33 had zero completed trajectories. Its actual selection contains 5120 distinct question IDs, pool indices and normalized-content hashes: 4670 remaining questions from cycle 0 followed by 450 from the newly shuffled cycle with seed 1730. The selection exactly matches the tail of epoch 32's immutable pool order and the prefix of the new order. `FIRST-POOL-CYCLE-SELECTION-SUMMARY.json` records this limited metadata check.

At the earlier selection-only observation, the epoch 33 completed-head pool snapshot was absent and pending. After successful epoch completion, its expected values are cycle 1, cursor 450, next epoch 34 and consumed questions 168960; these are expectations, not an already saved state. This check does not independently recheck complete historical question coverage or diagnostic exclusions. The archived controller reads metadata only, never stops or changes training, and copies no GPU tensors or weights. CP10, CP20 and CP30 CPU/full-SHA inspections are accepted; At that selection-only observation, CP40 was still future work and the 1000-epoch goal remained incomplete. Frozen runtime 4bc is unchanged, and selection structure is not a quality-success claim.


## Saved epoch 33 pool boundary

The same job 3409 subsequently completed epoch 33 with 3300 actual Adam updates on each rank at LR 3e-5. Epoch 34 rollout had started, with zero completed trajectories at the latest observation. `FIRST-POOL-CYCLE-SAVED-BOUNDARY-SUMMARY.json` now confirms that the actual E33 saved pool snapshot matches cycle 1, cursor 450, next epoch 34 and consumed questions 168960, with SHA256 `59f05ef043b69cbed93a3797b7cc38a995519ab18111a2af66067255578129aa`. Its 5120 unique selected question IDs/pool indices/normalized hashes remain the exact 4670-question old-cycle tail plus 450 from the newly shuffled cycle.

This closes only the E32→E33 saved-boundary check; it does not recheck all historical selections or fixed-diagnostic exclusions, read models, use GPU, copy large payloads or change the algorithm. The older selection-only JSON retains its historically correct pending status. E33 fixed diagnostic classification accuracy is 0.780859375 (majority 0.781640625), BCE 0.48350070798328926 and AUROC 0.7013000655307409. Training online accuracy is 0.788515625 and BCE 0.47998570317402484, measured before each own batch update. No quality-success or LR-only causal conclusion follows. Runtime 4bc remains frozen, the 1000-epoch target remained active and CP40 was future work at that E33 observation.


## Actual epoch 40 savepoint

The same job 3409 completed 40/1000 epochs with 4000 actual Adam updates on every one of eight ranks and LR 3e-5; epoch 41 had 4500 actual generated trajectories at the observation. Frozen runtime `4bc683883a569d7feb9d5da87307612b872e8f3a` remains unchanged. E40 fixed diagnostic accuracy is 0.776953125, BCE 0.48987869679932833, AUROC 0.6997677368829003, balanced accuracy 0.5517907414807801 and positive recall 0.1520572450805009. Accuracy is below the 0.781640625 majority baseline; this is not quality success or proof of an LR-only causal effect. Train online accuracy is 0.7875 and BCE 0.4706449410133064, with 5537 positive trajectories, measured before each own batch update rather than post-epoch training-set reevaluation. `EPOCH40-QUALITY-SUMMARY.json` preserves exact measures and observation identity.

`CHECKPOINT40-INDEPENDENT-CPU-SUMMARY.json` records the immutable 925542230-byte CP40, independent full-file SHA256 `234b014929082c7ceaed488c1014f62c22d48ac1357bece76e0c98aef90090c3` and actual CPU loading: epoch 40, all eight rank Adam counters at 4000, LR 3e-5, 14 FP32 head tensors/Adam parameter states and eight RNG records. Its saved pool is cycle 1, cursor 36290, next epoch 41, consumed questions 204800. This does not establish a GPU semantic forward or independent backup; no large payload was transferred to Mac. At the E40 observation, CP50–1000 and the overall goal remained incomplete; previous evidence JSON is unchanged.


## Actual epoch 50 savepoint

The same job 3409 completed 50/1000 epochs and continues toward 1000 with frozen runtime `4bc683883a569d7feb9d5da87307612b872e8f3a`. Every one of eight ranks has 5000 actual Adam updates at LR 3e-5. E50 fixed diagnostic accuracy is 0.779296875, BCE 0.48703575621821144 and AUROC 0.707252813664724, against majority accuracy 0.781640625. This is a saved milestone, not final completion or quality success. Exact threshold measures and train online metrics are archived in `EPOCH50-QUALITY-SUMMARY.json`; training online predictions are recorded before each sample's own batch update, not by post-epoch training-set reevaluation.

Across the existing E8–49 observations, E30 remains the lowest diagnostic BCE (0.47851565673745255) and highest AUROC (0.7129927880424725). E31–49 mean BCE/AUROC are 0.4924916553740628/0.6992431898164466, with BCE range 0.48350070798328926–0.50154410534905 and AUROC range 0.6899752270555242–0.7053132646556864. These observations indicate a fluctuating plateau rather than sustained improvement. E50 remains below the E30 best AUROC and above its best BCE; fresh examples and additional updates prevent attributing the results solely to LR.

`CHECKPOINT50-INDEPENDENT-CPU-SUMMARY.json` is the unchanged small audit summary for the immutable 925542230-byte CP50, full independent SHA256 `c1dc829d7a97075e1d934391a96c54a0225db874f8a4c872f3a2c51868f436c9`. Actual CPU inspection confirms head/Adam state, eight rank RNG records, 5000 updates per rank and LR 3e-5, with saved pool cycle 1, cursor 87490, next epoch 51 and consumed questions 256000. The remote `preparation/checkpoint0050-independent-full-sha-cpu-once-v1.json` proof has SHA256 `47c2d1dfcc705728af2a251ce1b9f228ac6cc4f8e7279449bf35660c86b55070`. This is not a GPU semantic forward or independent backup, and no model or large payload is stored locally or in Git. At the E50 milestone, CP60–1000 and the overall goal remained incomplete. Earlier evidence JSON is preserved unchanged.


## Actual epoch 60 savepoint

The unchanged job 3409/runtime 4bc reached immutable CP60 and continues toward 1000. Each of eight ranks has 6000 actual Adam updates at LR 3e-5. The exact E60 metrics, fetched separately from any later epoch, are in `EPOCH60-QUALITY-SUMMARY.json`: fixed diagnostic accuracy 0.776171875, BCE 0.4886032718963951, AUROC 0.7104855443476831 and positive recall 0.11270125223613596. Accuracy is below majority baseline 0.781640625. These results do not establish quality success or an LR-only causal effect. E60 train online accuracy is 0.7853515625 and BCE 0.47705243948847054, measured before each own batch update rather than by post-epoch training-set reevaluation. Later native progress is separate from these immutable E60 metrics.

`CHECKPOINT60-INDEPENDENT-CPU-SUMMARY.json` is the unchanged independent full-SHA/CPU inspection summary for the 925542230-byte CP60: SHA256 `5da3d9f4c1c9a59a49fbe0f240693215e63ea7e7a5711678291674a64c3dacf7`. Model, Adam state, eight rank RNG records and the pool bundle passed the recorded CPU inspection; saved pool cycle 1, cursor 138690, next epoch 61 and consumed questions 307200. Remote proof `preparation/checkpoint0060-independent-full-sha-cpu-once-v1.json` has SHA256 `4e431a2746110028451f7a2506735e7e2a9cd13ffbcc03856b17b6cdbc2e65df`. This is not GPU semantic forward or independent backup. No model/large payload is transferred to Mac or Git. CP70–1000 and the overall goal remain incomplete; earlier evidence JSON is unchanged.


## Stored data coverage through epoch 62

`DATA-COVERAGE-THROUGH62-SUMMARY.json` records metadata coverage through completed E62 only: 317440 stored prompt/gold rows, covering all 168510 unique eligible questions at least once (first complete coverage reached in E33). Fresh epochs E2–62 cover 168052 unique questions; 458 questions used in cached E1 have not yet appeared in a fresh rollout in this scope. E2–62 contain 1561600 fresh trajectories, plus 25600 cached E1 trajectories, for 1587200 total. All 488 compact-prefix and 488 full-raw rank payload files in the fresh scope are present with stat byte counts matching their receipts. This is metadata/stat evidence, not independent payload hashing or an independent backup. No later native progress is folded into the E62 coverage counts; frozen runtime 4bc is unchanged and the 1000-epoch goal continues.

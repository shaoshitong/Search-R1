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

`CHECKPOINT10-METADATA.json` records the immutable CP10 file (925542230 bytes), its producer SHA and paired bundle metadata. The model file was not independently hashed or CPU/GPU reloaded in this acceptance collector. Producer SHA is not independent full-file verification. No large payload, checkpoint, tensor or question text was copied into Git or to Mac.

The two scripts in `reproduction/prefix20_online/code_backup/` are archival read-only controllers. The five-epoch collector was consumed once; do not blindly replay it. The one-off current observer is also read-only. Neither changes training, state or automation. Old stop-v1/v2 and restore preparation evidence remain preserved separately; no old submission or stop is replayed by this publication.

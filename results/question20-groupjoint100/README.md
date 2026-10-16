# Fresh question+20 trajectory and G5 mixed joint probe

Authorized2026-10-16. Start a new probe from official copiedQwen block35 plus a fresh standard Linear, with fresh Adam/RNG/update0. No earlier learned probe checkpoint or optimizer is imported. The official actor/encoder remains frozen Qwen2.5-3B-Instruct revisionaa8e72537993ba99e69dfaafa59ed015b17504d1.

Train100epochs. Each epoch uses one original saved5120-question G5 batch: data sources1..70, then1..30. No new rollout or retrieval. Actual question/context token IDs plus the same trajectory first20 retained policy tokens enter the frozen encoder inside each training batch. Capture index27 post-block and detach; the head reads question pluspolicy representations, Qmax273/S293, copiedblock35 and maskedflatten to one logit. Do not pre-encode a whole epoch. Consumed features stay in rank-local CPU RAM until the post-update training-set evaluation, then are released; no disk feature cache is written.

For each original ownG5, train both mean per-trajectory BCEWithLogits(z,c) and mean group BCE with weight1. The mixed event is0<sum(ownstrictEM)<5, includingK1. The group score is1-prod(p)-prod(1-p). Stable32-pattern log-event algebra preserves the gradient without probability clipping. See [mathematical derivation](../../DERIVATION_PACKAGE.md).

Eight ranks, no gradient accumulation. One update contains51 or52 whole questions (255 or260 trajectories);80 and20 such updates give5120G5 and the average global trajectory batch256. Each rank receives6 or7 whole G5 per update,640questions/3200trajectories per epoch. Scale each local mean by8*local_questions/global_questions before DDP mean reduction. There are100 real Adam updates/rank/epoch and10000 in total. Learning rate applied at actualupdate t=0..9999 is3e-6+(3e-5-3e-6)/2*(1+cos(pi*t/9999)), with exact first3e-5/last3e-6 and no warmup.

After each epoch, the same post-update head evaluates the full own5120G5 batch and unchanged fixed512G5 diagnostic. Report separate trajectory/group/total losses, group ordinary and balanced accuracy, and both class recalls at unchangedthreshold0.5. The test feature manifest remainsa125fc5af90e34fcd728c82077d9d62070a78d5ef703ac11f3748687d088e64f. Train batches differ, so their metrics are not a fixed-training-set learning curve. Mathematical consistency and CPU tests do not establish improved quality.

All checkpoints, inputs, tensors, full predictions and original trajectories remain remote. Epochlatest includes head/Adam/eight-rank RNG/actual cosine clock/source schedule. Immutable complete checkpoints are retained every10epochs through100. Original source batches and completed replay70 artifacts remain read-only. NoGPUforward gate, new baseline, calibration or foreign-process termination is added.

The run completed all100 epochs onh100-8 at2026-10-16T10:37:53Z. Every rank reached10000 actualAdam updates, with the exact final learning rate3e-6. The original initialization proof records freshAdam0, a fresh standardLinear, official copiedblock35 and no old probe/optimizer import. No actor training or new rollout was performed.

Final post-update group diagnostics, at the unchanged0.5 threshold:

| Metric | Own epoch100 training batch | Fixed512-question diagnostic |
| --- | ---: | ---: |
| Ordinary mixed accuracy | 74.0625% | 61.5234375% |
| Balanced mixed accuracy | 73.91424974% | 59.87592805% |
| Mixed recall | 73.39149400% | 54.09836066% |
| Nonmixed recall | 74.43700548% | 65.65349544% |
| Trajectory BCE | 0.4237809287 | 0.5063566121 |
| Group BCE | 0.5371765241 | 0.6710414595 |

The highest fixed-diagnostic balanced accuracy over all100 logged epochs was62.96194795% at epoch45. Epoch45 was not an immutable savepoint, so this metric does not identify an available weight checkpoint. Among the ten actually retained and independently CPU-verified savepoints, epoch100 has the highest diagnostic balanced accuracy59.87592805%. Its remote checkpoint is `/mnt/data/shaoshitong/search-g1-question20-groupjoint100-20261016-cee20bcd/train/head-epoch-0100.pt`,932252950bytes, independentSHA256 `8df6bd445945ffbbf6a94171f24934495f0ce3ff17a3d2577bbd53d4b8fefd56`.

Both losses are lower than in the first ten epochs, but heldout balanced accuracy fluctuates and the last-ten mean58.24987128% remains below the41–50 mean58.78859601%. Final ordinary accuracy is below the fixed diagnostic majority baseline64.2578125%; balanced accuracy is above the constant-class baseline50%. This is modest discrimination, not demonstrated policy or QA improvement. The fixed diagnostic is used to select a checkpoint, so its selected score is not an independent final-test estimate. Training batches differ by epoch and sources1..30 are replayed at epochs71..100.

Median interval between complete epoch receipts was51.820646seconds. The final inline encoder and training loop took14.780988seconds; the latter excludes source preparation, worker/model startup and saving.

[ALL-EPOCHS.md](ALL-EPOCHS.md) and [FINAL-METRICS-100.json](FINAL-METRICS-100.json) contain the small aggregate diagnostics. [Curves](curves/README.md) provide the complete100-epoch static figures and interactive view. [FIRST-ACTUAL-SUMMARY.json](FIRST-ACTUAL-SUMMARY.json) remains the historical startup proof. All ten savepoints have passed independent full-fileSHA and actualCPU model/Adam/RNG-state reload; [FINAL-ACCEPTANCE.json](FINAL-ACCEPTANCE.json) records the native terminal and retained-data acceptance. Slurm controller original records show JobId3412 `WEXITSTATUS 0` followed by `done`, bound to the original running allocation; accounting is disabled and the in-memory scontrol record had expired. All owned process births/cgroups/GPU processes are gone. The1,006 terminal original-inode logs,49,969,292bytes, were fully SHA-read, and all70 source batches remain retained. No epoch101 was run.

The proposed next prefix-selection actor experiment is a draft awaiting human approval. Completion of this warmup does not authorize starting it.

# Question plus20-token replay70: completed

All70 replay iterations completed on h100-8. Native job3411 ended `COMPLETED0:0`; owned process births, native cgroups and owned GPU processes were gone. Each rank made7000 new Adam updates, continuing the source head from7100 to14100 at LR3e-6. No iteration71 or new rollout was run in this replay.

## Mixed-decision accuracy

The score is `q=1-prod(p_j)-prod(1-p_j)`. Prediction is mixed when `q>=0.5`; the true G5 label is mixed when `0<sum(own strict EM)<5`. These are question-group classification accuracies, not QA EM/F1.

| Post-update head | Own-batch train5120G5 | Fixed-test512G5 |
|---|---:|---:|
| Final replay70 | 3871/5120 =75.60546875% | 325/512 =63.4765625% |
| Best fixed-test, replay63 | 3651/5120 =71.30859375% | 327/512 =63.8671875% |

The unchanged fixed test has183 mixed and329 nonmixed questions. Always predicting nonmixed gives329/512=64.2578125%; the best observed test accuracy remained two questions below this baseline, and the final head remained four questions below it. The completed training and savepoint checks do not establish quality success or an algorithm benefit.

Train accuracy is a full post-update evaluation on the particular5120-question batch just trained, using the same head as that iteration's test evaluation. Those training batches differ between iterations; their scores are not a fixed-training-set learning curve. Test uses the same original512G5 cached trajectories every time. Iterations1..6 originally recorded only test accuracy; their train values remain missing. Iterations7..70 have paired train and test results.

[All70 iteration accuracies](ALL-ITERATIONS.md) and [the exact small remote accuracy report](FINAL-ACCURACY-SUMMARY.json) include every completed iteration. The report crosschecks saved aggregate confusion counts against completion receipts; it does not independently reload full prediction tensors.

## Recipe and retained state

The encoder is frozen official Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Its layer27 post-block representation contains the actual initial request plus the same trajectory's first20 retained policy tokens. Original token IDs, causal contexts, masks and per-turn positions are preserved. Qmax is273, head sequence width is293, and the flattened dimension is600064. The probe trains the copied complete block35 and Linear output with per-trajectory BCE.

The replay processes the previously saved E1..70 batches in their original order. Each has5120 G5 questions and25600 own-EM trajectories. Training uses globalBS256/local32/world8/no accumulation and100 real Adam updates per rank per iteration. The actor and encoder receive no updates. This is a continuation of the complete sourceE71 probe, not a fresh probe trained from official random head weights.

Replay1..6 ran in job3410. After the requested paired training-set evaluation was added, job3411 restored the complete replay6 head, Adam and eight-rank RNG and continued7..70 without replaying the earlier updates. Frozen final training runtime is `5fa565b576811a5ddec7bdc3edcb384bd05f608c`. Later report/audit code `3f24079a` runs separately on CPU and does not change training.

The seven immutable checkpoints at10/20/30/40/50/60/70 passed independent complete-file SHA and actual CPU state loading. Each has14 FP32 model tensors,14 Adam states and eight saved rank RNG records, with the expected Adam counters. The audit also hashed198 original log files and confirmed retention of all70 source batches.

[Final acceptance summary](FINAL-ACCEPTANCE-SUMMARY.json) records exact remote proof paths, SHA values, checkpoint results and native resource evidence. CPU state loading is not an independent GPU semantic reload, CUDA RNG application, or an independent durable backup; those operations were not performed by this final audit. Retained source trajectory payloads were inspected by metadata and were not independently rehashed in this audit.

Final artifacts remain remote under `/mnt/data/shaoshitong/search-g1-question20-replay70-train-test-20261016-5fa565b`. The final checkpoint is `train/head-replay-0070.pt`, SHA `26b71d1018d8246bdb90dbd49570d8ce441e3b453b5f24896cb7cff9c9354125`,932252310 bytes. Original cached data, full predictions, model/Adam states and tensors stay remote; this directory contains only code-linked documentation and small audit summaries.

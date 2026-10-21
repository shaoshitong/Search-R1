# Training-only true G10 diversity by domain

New HotpotQA data has substantially higher observed correctness diversity than NQ. New NQ's uniform-G5 expectation is32.403%, close to the current NQ cohort's32.533%. The new15616-question aggregate expectation is35.564%, primarily reflecting addition of3316 HotpotQA questions with a47.292% expectation. These are properties of accepted training rollouts, not probe accuracy, predicted Mixed scores, or evidence that a future model will improve.

| Accepted cohort/domain | Questions | G10 all wrong | G10 all correct | G10 heterogeneous | Expected Mixed under uniform distinct5 of10 |
|---|---:|---:|---:|---:|---:|
| Initial imported, NQ | 31232 | 46.036% | 10.553% | 43.411% | 32.911% |
| Current frozen training, NQ | 30016 | 46.359% | 10.658% | 42.984% | 32.533% |
| New increment, NQ | 12300 | 46.366% | 10.919% | 42.715% | 32.403% |
| New increment, HotpotQA | 3316 | 34.409% | 6.514% | 59.077% | 47.292% |
| New increment, aggregate | 15616 | 43.827% | 9.983% | 46.190% | 35.564% |

G10 heterogeneous means `0<K<10`. For a question with K correct trajectories among10 authentic trajectories, the exact expected true Mixed label when sampling five **without replacement** is

`1 - [C(K,5) + C(10-K,5)] / C(10,5)`, where `C(n,5)=0` for n<5.

Each original question has total weight1. The reported rate averages this exact finite-sample expectation across questions. It is not a conditional-independence Bernoulli estimate and is not the G10 heterogeneous rate. For example, K=1 makes G10 heterogeneous but only50% of its252 distinct G5 subsets are Mixed; K=5 gives125/126 rather than100%.

## Actual K distributions

Counts below are ordered K=0,1,...,10:

- Initial31232 NQ: `[14378,2144,1403,1196,1027,1038,1176,1340,1777,2457,3296]`.
- Current30016 NQ: `[13915,2062,1333,1137,966,987,1117,1257,1688,2355,3199]`.
- New12300 NQ: `[5703,844,545,403,436,388,457,552,695,934,1343]`.
- New3316 HotpotQA: `[1141,246,222,181,197,200,230,213,220,250,216]`.
- New15616 aggregate: `[6844,1090,767,584,633,588,687,765,915,1184,1559]`.

Actual old5/newseed5 true Mixed rates in the new NQ increment are both32.195%. In new HotpotQA they are47.075% and47.859%, respectively. The corresponding new aggregate rates are35.355% and35.521%. All K counts use own strict EM attached to the exact ordered authentic trajectory IDs; old5/new5 joins and independent seeds were already accepted by the actual ingester. This analysis rechecks their group files and accepted histogram bindings without regenerating rollouts or rerunning token-level EM.

## Identity-only overlap

The new15616 questions were compared using both stable qid and canonical question hashes:

| Reference scope | Reference questions | Shared qids | Shared distinct hashes | New questions with any shared hash |
|---|---:|---:|---:|---:|
| Initial imported training | 31232 | 0 | 0 | 0 |
| Current frozen training | 30016 | 0 | 0 | 0 |
| Previous controlled cohort | 6528 | 584 | 584 | 584 |
| Diagnostic identities | 512 | 0 | 0 | 0 |
| Selection identities | 2048 | 0 | 0 | 0 |
| Frozen final identities | 4096 | 0 | 0 | 0 |

The584 overlaps with the previous controlled cohort matter for comparisons. The complete new increment cannot be described as independent of that cohort. A stricter expansion excluding those prior controlled qids and hashes would have15032 questions before any new split. All questions still belong to the original169028-question warmup pool; none of these counts establishes unseen-original-warmup performance.

Only embedded identity-only metadata for diagnostic/selection/final scopes was opened. Its final source references were not followed. No fixed2048 or final4096 labels, rollouts, predictions, scores or model features were read. The controlled6528 qids came from its accepted frozen plan; their hashes were independently rederived from an identity-only canonical parquet projection (prompt/data_source/extra_info.index, without reward/target columns). The original canonical ordered-qid digest matched the accepted warmup manifest.

## Suggested next-stage grouping, not applied

Keep the active30016-question training plan unchanged. For a separately registered next-stage experiment, first freeze eligibility using identities alone, retaining the verified diagnostic/selection/final exclusions. If independence from all prior controlled fine-tuning is required, also exclude the584 controlled-cohort overlaps before splitting.

Partition whole canonical question-hash clusters, keeping all aliases, ten trajectories, two real G5 events and any virtual G5 views together. Within each actual domain, use one predeclared seeded hash ordering to reserve approximately20% of clusters for a development group and80% for training; freeze membership before consulting correctness labels or model scores. Use the same frozen groups for all recipes. Do not select questions or reserve groups according to K, Mixed labels, current probe confidence or external validation errors. Any cross-domain alias must stay in a single group rather than following two independent domain assignments.

Report NQ and HotpotQA separately and include the population-weighted aggregate. This new heldout group would measure separation from the next fine-tuning stage, **not** unseen original warmup questions. No split, training plan or new candidate was created by this analysis.

## Actual execution and source binding

Source commit90d05c6f was pushed and remote verified before execution. Two focused stdlib fixtures passed: all252 G5 combinations for each K=0..10 independently match the exact formula, and a G10-heterogeneous example is explicitly different from its G5 Mixed expectation; identity-alias count denominators are also checked.

The actual worker completed in5.265 seconds, peak RSS529747968 bytes, two CPUs and zero GPUs. A second terminal read reproduced the same accepted small receipt without recomputing the dataset analysis. Source metadata/raw-group files have SHA bindings in the result; large models, optimizer states, feature tensors and rollout payloads were not read.

Remote receipt: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/training-G10-delta-domain-statistics-v1/complete.json`,16257 bytes, SHA256`48fd7877d15c1fa6b49b7d046ca30b1e3f318ad5f5ff205aecff1f45342f23a0`.

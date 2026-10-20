# Peer data integration and completed controls — 2026-10-21

The authorized peer frozen subset is fully accepted at
`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/peer-newseed-h100-20261021/newseed-20261021-185747-445fc691/raw`.
All16 archives (1,366,896,640 bytes) and3466 members (1,360,998,806 bytes) passed the receiver and a separate full fresh SHA readback. The actual target complete SHA is `8a0f5128f7f613d4f6ee9cc5d6135f461f66feb66e73b65bbfa5e78ada875d5d`. This is31,232 NQ questions/156,160 genuine new-seed G5 trajectories, not all169,028 source questions. All source and intermediate copies remain; no peer job or Mac network configuration was changed. See the transport incident report and exact independent readback audit.

CPU ingest has now accepted all31,232 questions and312,320 original5+new5 trajectories after recomputing strict EM and checking seed independence, causal contexts and trajectory identities (zero EM mismatches or seed collisions). An identity-only hash sample selects8192 questions outside the controlled6528 cohort and evaluation pools, before labels are used. All81,920 Prefix20 trajectories have subsequently been captured in four normal-priority one-H100 feature jobs, each completed0:0. Source and feature identities and producer fresh payload SHA checks passed; all tensor data remain remote.

The new HF batch32/16384 backend passed causal-mask/position/length checks, with small BF16 numerical differences from scalar replay. It is not described as bit-exact. Training and the fixed2048-question validation use the same new backend, and E1/E4 were re-scored on these exact validation features. Untouched final pools remain sealed. The original warmup already used all169,028 training questions: this adds independently sampled trajectories, rather than claiming previously unseen questions. The selected cohort is NQ-only, so seven-domain generalization is not inferred.

## Finished controls

All rows below use the SAME newly frozen2048 validation questions/G5 IDs (663 Mixed=32.37%), threshold0.5. All four jobs completed0:0 with final full-state SHA/Adam/RNG and prediction-ownEM/q audits passed. There is no claim of matched start/budget across the two different experiment pairs.

| Experiment | Updates | Mixed balanced accuracy | Mixed recall | nonMixed recall | allwrong FPR | allcorrect FPR | Mixed@512 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E4 before continuation | — | 55.53% | 66.37% | 44.69% | 49.86% | 72.73% | 200/512 (39.06%) | 0.57610 |
| hard-vs-random-E4-477-v1/targeted | 477 | 53.72% | 76.47% | 30.97% | 64.74% | 82.73% | 201/512 (39.26%) | 0.57706 |
| hard-vs-random-E4-477-v1/random_control | 477 | 55.57% | 73.15% | 37.98% | 57.54% | 76.36% | 182/512 (35.55%) | 0.57092 |
| g10-matched6528-v1/random | 136 | 55.15% | 89.14% | 21.16% | 73.46% | 96.06% | 194/512 (37.89%) | 0.57322 |
| g10-matched6528-v1/global-weighted-balanced | 136 | 54.64% | 90.95% | 18.34% | 77.16% | 96.06% | 198/512 (38.67%) | 0.58033 |

Targeted enrichment improved selection by19 Mixed questions versus its same-start/same-budget random control, but only1 versus the source E4 and with worse balanced accuracy. Strict global balance also did not resolve the large nonMixed false-positive rate. None meets the joint improvement gate. Full audits are durably published on backup branch `codex/qwen7b-event-followthrough-audits-20261020T200618Z`, remote commit26388aca7d3179abe1b5aabb014f87689f9e9cfe; audit SHA a27a2644841ef5bcf8b9ce1c6a4ac6d37a6de538f3c5238b4a2a8565147c6123.

## Actual use in probe fitting

Independent NQ-only8192 experiment743367 branches from sealed E1 raw/EMA/full Adam10464/RNG, unchanged5e-6 LR, B32×256 updates, four unique random G5 subsets per real G10, each view weight1/4. The native block/readout is inherited; there is no reset or student actor update. Immutable runtime is code-d8f74420aa09 and config SHA16b2fd34e3e8907307b3866298b33216478fd77c2dc1b9ca80638e0929800130. The first actual update10464→10465 passed deterministic raw/EMA/Adam/buffer/RNG replay. Independent fullCP32 SHA acceptance and later periodic checkpoints confirm actual training, not merely Slurm RUNNING. At epoch1792529438 the fit had151/256 new updates, Adam10615, completeCP144. Final validation was still pending at that observation.

The actual G5 training-view Mixed fraction is10,643/32,768=32.48%. The42.59% heterogeneous fraction of the underlying G10 groups is not the G5 training-label rate. Exact full-causal Prefix20 opposite-label conflicts affect505/57,765 correct/wrong pairs (0.874%); their finite-input empirical BCE bound does not explain the entire generalization problem or establish a test Bayes bound.

Same-HF fixed-G5 validation contains663/2048 real Mixed questions (32.37%). Before continuation, E1 has BA50.68% and Mixed@512185/512=36.13%; E4 has BA55.77% and196/512=38.28%. Use these numbers for the new candidate's fair comparison, rather than the historical native-backend E4 result200/512. This cohort/backend/update budget differs from earlier6528 mixed-domain experiments, so cross-suite differences are not an equal-budget causal estimate of new-data benefit. Final joint BA and selection improvement remains unproved.

## Final peer-data result — 2026-10-21 05:03 CST

Job743367 completed0:0 in27m19s, with256 consecutive real updates and Adam10720. Final full-stateCP256 contains3,745,832,474 bytes, SHA256 `36d9bcc3be71ba94eaca3a1e1b292c0ed9a65297fb96a746ea5295432f09c000`. The GPU saver accepted raw/EMA/Adam/buffers/RNG; the independent final audit re-read the entire checkpoint SHA and recomputed all own-EM, identity, probability, formula and metric checks. Student actor updates remain zero. Final4096 identities remain untouched by generation or scoring.

| Same HF validation | Mixed BA | Mixed recall | nonMixed recall | allwrong FPR | allcorrect FPR | AUC | Mixed@512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 source | 50.68% | 99.55% | 1.81% | 97.73% | 99.70% | 0.55444 | 185/512 (36.13%) |
| E4 comparison | 55.77% | 67.57% | 43.97% | 51.18% | 71.52% | 0.57648 | 196/512 (38.28%) |
| Peer8192 G10, 256 updates | 54.66% | 67.72% | 41.59% | 50.24% | 84.55% | 0.55990 | 182/512 (35.55%) |

The peer-data fit does not pass either improvement gate against E4. Allcorrect false-Mixed predictions increased by43 net questions (66 newly wrong,23 corrected), whereas Mixed false negatives changed215→214. Allcorrect questions have mean trajectory probability0.38773 and mean Mixed score0.74355; true Mixed questions have mean score0.63149. Mean correct-minus-wrong trajectory probability within actual Mixed groups is only0.00971. None of the663 Mixed groups has five exactly equal probabilities or five probabilities all in0.45–0.55. Thus exact repeated-prefix identity or a literal all-0.5 output is insufficient to describe this failure: separation of correct and incorrect trajectories remains weak, while many allcorrect questions receive high Mixed scores.

These are distinct new predictions: compared with E4, mean absolute trajectory-probability change is0.13475,442 questions cross the0.5 Mixed threshold, and the selected512 overlap on243 identities. Repeating the integer182 is not evidence of reusing the earlier output. The new pool itself also differs from the historical641-Mixed pool. Raw predictions stay remote; only small audits are published.

The separate bounded CPU diagnostic using frozen L26 Prefix20 mean/last embeddings also failed: its internally selected mean-pooling linear head reaches fixed0.5 BA49.95%, AUC0.52629 and158/512 (30.86%). An internal-CV-frozen threshold reaches BA51.82% on the same single external scoring pass. It is not promoted, and this finite linear experiment does not establish that L26 has no useful information. See `../peer8192-linear-group-head/README.md`.

The independent real-G20 controlled experiments remain in progress. They use the exact earlier G10 question batches and update order, E1 state and136-update budget, with random and globally balanced four-view plans; their results must be compared with the native-backend E4 baseline above. A new, not-yet-submitted implementation will test exact uniform averaging over all genuine five-subsets per question, avoiding the current four-view Monte Carlo estimate while retaining per-question total weight1. No claim of improved selection or completed overall goal is made.

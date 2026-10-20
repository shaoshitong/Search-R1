# Paired old5 versus new-seed5: seen NQ8192 training-cohort diagnosis

This is CPU-only diagnosis of sealed frozen scores, not a new training or validation experiment. It uses the same8192 NQ training questions: the first5 trajectories are from the original warmup, the last5 from an independently seeded rollout. These question identities were seen in original warmup. E4 saw the original trajectories; this is not an unseen-question test. The peer256 branch used this old5+new5 training pool. Full probability/paired rows stay on HPC3. No model or feature payload, external validation or final pool was opened; no Platt grid or threshold was changed.

Important comparison limit: E4 is the accepted Adam14418 EMA checkpoint; peer256 is a different branch inheriting E1 Adam10464 plus256 updates, ending10720. Cross-model changes below are not a causal E4-before/after-finetune comparison. Within each fixed model, old5/new5 use exactly the same question cohort.

Source identities/own strict EM, probability/logit sigmoid, file SHA, and label-blind stratification were checked. The seed9137 plan was fixed using only qid hashes before label analysis, yielding four disjoint2048-question strata; the same strata are reused for both models and trajectory events. Each event is scored by `1-prod(p)-prod(1-p)`, with fixed0.5 classification threshold.

| Frozen model/event | Mixed prevalence | Mixed BA | TPR | TNR | AUC | Allwrong FP | Allcorrect FP | Global Top2048 Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E4/old5 | 32.031% | 69.300% | 84.756% | 53.843% | 0.750703 | 41.974% | 59.459% | 1118/2048 (54.590%) |
| E4/new5 | 32.385% | 63.363% | 76.555% | 50.172% | 0.672378 | 43.857% | 68.859% | 963/2048 (47.021%) |
| peer256/old5 | 32.031% | 63.178% | 76.105% | 50.251% | 0.671832 | 38.928% | 84.159% | 924/2048 (45.117%) |
| peer256/new5 | 32.385% | 62.872% | 75.952% | 49.792% | 0.662843 | 39.018% | 85.865% | 929/2048 (45.361%) |

Top512 true Mixed counts in the four fixed2048-question strata:

- `E4/old5`: `[289, 271, 267, 290]`.
- `E4/new5`: `[236, 234, 238, 255]`.
- `peer256/old5`: `[238, 222, 229, 236]`.
- `peer256/new5`: `[223, 221, 254, 236]`.

The same frozen E4 loses5.94 BA percentage points on fresh-seed trajectories. Its old-trajectory selection advantage is therefore not an unseen-question advantage. The within-Mixed statistics below show that E4 separates correct/wrong trajectories much better on original rollouts than fresh ones. The independently trained peer branch does not show a material improvement of fresh-trajectory within-group separation. This is consistent with trajectory-specific overfitting/weak transfer, but this observational comparison does not identify a single causal mechanism.

| Model/event | All trajectory BCE | Correct BCE | Wrong BCE | Overall trajectory AUC | Within-Mixed macro p(correct)-p(wrong) | Within-Mixed macro trajectory AUC |
|---|---:|---:|---:|---:|---:|---:|
| E4/old5 | 0.514651 | 0.841174 | 0.355169 | 0.803603 | 0.073047 | 0.692295 |
| E4/new5 | 0.558893 | 0.914540 | 0.384705 | 0.775630 | 0.012593 | 0.538227 |
| peer256/old5 | 0.481602 | 1.046393 | 0.205744 | 0.828295 | 0.027264 | 0.596052 |
| peer256/new5 | 0.497547 | 1.076831 | 0.213825 | 0.815461 | 0.013167 | 0.537034 |

Macro within-Mixed statistics give each actual Mixed question equal weight; pooled values are also retained in the JSON. The lower overall trajectory BCE in peer256 accompanies improved wrong-trajectory BCE and worse correct-trajectory BCE. Its higher overall trajectory AUC therefore does not establish improved Mixed classification or within-question discrimination.

Actual Mixed labels change between the two independent G5 events in1693/8192 questions (20.667%):861 nonMixed→Mixed,832 Mixed→nonMixed,1792 remainMixed and4707 remainnonMixed. In E4 the two q scores remain highly correlated (Pearson0.9686, Spearman0.9589). For832 old-Mixed→new-nonMixed questions, mean q changes only0.7332→0.7189 and676 fresh groups remain false positives. The score largely follows stable question-level difficulty and often fails to track this realized group-label change; old/new trajectory positions are not treated as paired independent outcomes.

Two actual compute stages were verified: E4 and peer256 each wrote8 monotone progress points from1024 to8192 questions. The entire bounded CPU operation finished in about3seconds, faster than the first poll; the independent final audit checked these real progress receipts, own rows/SHA, metrics and stratum rankings. CPU affinity was restricted to2 cores and address-space cap3GiB. The process exited, its lock is free and no failure receipt exists.

Evidence: [small aggregate and independent audit](actual-launches/peer-oldnew-complete-and-independent-audit.json). Remote complete: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-paired-old-new-20261021/complete.json`, SHA `f755ad9d187438ac6b156ee78d5ee81e882b26e071943dca35bc5124fd8f71f6`. Runtime source was published and checked as `262f549ea03c7f0fbec71118f4615d76afae7512` (`56cc5b3b...` standalone file).

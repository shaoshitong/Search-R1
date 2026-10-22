Full Z two-training-seed E10/raw development comparison

E10/raw was selected only by the already declared seed1 rule and applied unchanged to seed2. No seed2 epoch/state/threshold was selected. This is the viewed2048 development ownG5 pool:663 Mixed,1055 all-wrong,330 all-correct. Every qid, ordered ownG5 trajectory list and all5 strictEM labels match exactly.

| E10/raw model | Top512 Mixed | Mixed BA (vhat>=.08) | Top all-wrong/Mixed/all-correct |
|---|---:|---:|---|
| seed1 | 245/512=47.8516% | 59.2958% | 194/245/73 |
| seed2 | 224/512=43.7500% | 57.4321% | 195/224/93 |

Per-seed mean selection fraction45.8008%; meanBA58.3640%. This averages model-level results, not prediction ensembling.

| Actual true group | seed1 errors | seed2 errors | Error meaning |
|---|---:|---:|---|
| all_wrong (N=1055) | 210 | 224 | nonMixed_false_positive |
| Mixed (N=663) | 399 | 406 | Mixed_false_negative |
| all_correct (N=330) | 84 | 107 | nonMixed_false_positive |

Within Top512, seed2 loses21 Mixed while all-correct distractors rise73→93 and all-wrong194→195. Thus the TopK difference is mainly additional all-correct distractors. Across the whole pool at the fixed.08 diagnostic threshold, all-wrong false positives increase210→224, all-correct false positives84→107, and missed Mixed399→406. These are binary Mixed errors; the model does not separately output all-wrong/all-correct class labels.

The selected sets overlap on289 questions, including137 trueMixed.

2000 paired-question resamples rerank each2048→Top512. Seed2−seed1 Top delta−4.1016percentage points, conditional95% interval[−7.8125,+.3906]pp. BA delta−1.8636pp, interval[−4.2647,+.4420]pp. Both intervals include0; these are viewed-question conditional intervals, not training-seed uncertainty or unseen/final confirmation. Seed1 development was used for epoch selection, so this is not an unbiased new-question estimate.

Input identity SHA2568576fb8f80441e2ef37f36ea57009a218384abab6faddfbaa362243d4e1f642c. Remote result/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/viewed2048-fullZ-two-seed-E10raw-comparison-20261023-v1/result.json; SHA256c623c3359178c69ad191b13c2aede70ee01111b9c4a2a52e5c91f5e3000792c1. Result independently rehashed unchanged. Raw predictions remain remote; local files contain only small aggregate reports. No checkpoint/model/hidden tensor, final4096, new inference or Slurm operation.


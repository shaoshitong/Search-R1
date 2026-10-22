# Complete five-layer P20 matched pilot results

Actual measurement UTC 2026-10-22T15:32:28.677232+00:00. L15/L18/L21/L24/L27 each completed3epochs and192updates with the same8192 questions, ownG5 plans, seed, LR5e-6 and EMA0.99. Every independent2048 question/ownG5 trajectory identity and strictEM is exactly identical. Fixed identity SHA: `c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7`. No final4096, new inference, model or hidden-feature tensor was read.

This is an already-viewed exploratory pool, not untouched final confirmation. Paired2000 bootstrap samples use the same questions across methods. Intervals omit training-seed variability and are not corrected for multiple layer comparisons.

| E3 model | Mixed BA | BCE | AUC | Top512 trueMixed | Mixed recall | nonMixed FPR | allWrong FPR | allCorrect FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1_raw | 52.28% | 0.643807 | 59.28% | 201/512 (39.26%) | 13.88% | 9.31% | 8.53% | 11.82% |
| E1_EMA | 52.88% | 0.653016 | 59.50% | 205/512 (40.04%) | 21.87% | 16.10% | 13.93% | 23.03% |
| L15P20_E3_raw | 58.58% | 0.687674 | 60.02% | 200/512 (39.06%) | 63.95% | 46.79% | 42.84% | 59.39% |
| L15P20_E3_EMA | 57.10% | 0.684058 | 58.97% | 205/512 (40.04%) | 60.48% | 46.28% | 42.84% | 57.27% |
| L18P20_E3_raw | 58.65% | 0.727862 | 60.93% | 206/512 (40.23%) | 77.53% | 60.22% | 55.64% | 74.85% |
| L18P20_E3_EMA | 57.22% | 0.676125 | 59.69% | 205/512 (40.04%) | 54.30% | 39.86% | 36.49% | 50.61% |
| L21P20_E3_raw | 57.23% | 0.711915 | 58.88% | 198/512 (38.67%) | 69.83% | 55.38% | 51.09% | 69.09% |
| L21P20_E3_EMA | 56.34% | 0.685009 | 58.80% | 207/512 (40.43%) | 58.67% | 45.99% | 42.84% | 56.06% |
| L24P20_E3_raw | 54.21% | 0.729479 | 55.82% | 187/512 (36.52%) | 70.44% | 62.02% | 57.73% | 75.76% |
| L24P20_E3_EMA | 54.19% | 0.693372 | 55.73% | 188/512 (36.72%) | 59.43% | 51.05% | 48.72% | 58.48% |
| L27P20_E3_raw | 53.88% | 0.776072 | 56.95% | 200/512 (39.06%) | 89.14% | 81.37% | 78.48% | 90.61% |
| L27P20_E3_EMA | 53.90% | 0.674934 | 56.73% | 198/512 (38.67%) | 45.70% | 37.91% | 35.64% | 45.15% |

L15/L18 improve fixed-threshold BA, but do not stably improve selection. Both EMAs select205Mixed, exactly matching E1EMA. L18raw AUC60.93% is1.43pp above E1EMA59.50%, but paired95% interval[-0.81,+3.77]pp crosses0; Top512 improves by only1question. L24EMA AUC is3.77pp below E1EMA (95%[-6.00,-1.63]pp), selecting188Mixed.

Within the same8192 budget: L18EMA minus L27EMA AUC is+2.96pp (95%[+1.11,+4.95]pp). Versus L21EMA it is only+0.89pp (95%[-0.12,+1.91]pp), so18 versus21 is not established. L15EMA also improves AUC against L27 but not against E1.

AllCorrect remains confounded with Mixed: L18EMA mean scores are0.503708 and0.505949 respectively; AllCorrect FPR50.61%. L15EMA means0.507148 and0.511632, with AllCorrect FPR57.27%. Shallower layers improve parts of classification without resolving own-group composition or high-score selection.

Each epoch actually contains3489Mixed/4703nonMixed. All3489G10Mixed questions are assigned a genuine G5Mixed subset; no G10Mixed-to-homogeneousG5 negatives occur. Global BCE weights are1.1739753511/0.8709334467. This isolates layer choice and is not the natural-randomG5 repair.

Per-epoch fixed train and independent results (BA/BCE/Top512Mixed):

| Layer | Epoch | Raw train | Raw independent | EMA train | EMA independent |
|---|---:|---|---|---|---|
| 15 | 0 | 50.00%/0.7265/235 | 50.00%/0.7630/174 | 50.00%/0.7265/235 | 50.00%/0.7630/174 |
| 15 | 1 | 51.14%/0.6736/280 | 50.23%/0.6444/204 | 50.17%/0.7069/248 | 49.85%/0.7292/176 |
| 15 | 2 | 58.30%/0.6928/304 | 54.75%/0.7463/204 | 57.57%/0.6872/288 | 54.56%/0.7006/194 |
| 15 | 3 | 63.36%/0.6446/325 | 58.58%/0.6877/200 | 59.70%/0.6701/305 | 57.10%/0.6841/205 |
| 18 | 0 | 50.00%/0.7248/243 | 50.00%/0.7601/170 | 50.00%/0.7248/243 | 50.00%/0.7601/170 |
| 18 | 1 | 51.13%/0.6724/281 | 50.45%/0.6352/203 | 50.01%/0.7046/263 | 50.35%/0.7261/183 |
| 18 | 2 | 57.80%/0.6911/309 | 55.74%/0.7574/205 | 57.82%/0.6812/290 | 56.59%/0.6930/198 |
| 18 | 3 | 61.88%/0.6587/329 | 58.65%/0.7279/206 | 61.40%/0.6622/312 | 57.22%/0.6761/205 |
| 21 | 0 | 50.00%/0.7555/220 | 50.00%/0.8093/161 | 50.00%/0.7555/220 | 50.00%/0.8093/161 |
| 21 | 1 | 52.63%/0.6679/294 | 51.32%/0.6459/198 | 50.25%/0.7163/246 | 50.40%/0.7473/177 |
| 21 | 2 | 59.21%/0.6770/318 | 55.52%/0.7353/210 | 57.05%/0.6852/298 | 55.12%/0.7055/201 |
| 21 | 3 | 63.92%/0.6446/332 | 57.23%/0.7119/198 | 61.90%/0.6619/316 | 56.34%/0.6850/207 |
| 24 | 0 | 50.00%/0.8026/216 | 50.00%/0.8797/162 | 50.00%/0.8026/216 | 50.00%/0.8797/162 |
| 24 | 1 | 52.66%/0.6711/295 | 51.21%/0.6531/182 | 50.00%/0.7376/242 | 50.00%/0.7828/177 |
| 24 | 2 | 58.95%/0.6822/317 | 54.25%/0.7363/189 | 54.46%/0.6951/284 | 53.02%/0.7218/182 |
| 24 | 3 | 63.65%/0.6489/340 | 54.21%/0.7295/187 | 60.88%/0.6677/319 | 54.19%/0.6934/188 |
| 27 | 0 | 50.42%/0.7379/211 | 49.48%/0.7720/164 | 50.42%/0.7379/211 | 49.48%/0.7720/164 |
| 27 | 1 | 50.75%/0.6753/282 | 50.72%/0.6443/181 | 51.08%/0.7023/237 | 52.06%/0.7128/175 |
| 27 | 2 | 59.34%/0.6799/308 | 54.62%/0.7262/187 | 56.16%/0.6804/272 | 53.04%/0.6834/181 |
| 27 | 3 | 59.01%/0.6817/337 | 53.88%/0.7761/200 | 59.22%/0.6626/303 | 53.90%/0.6749/198 |

Fixed train2048 was drawn from the training pool; independent2048 receives no updates. Mixed prevalence differs (train888/2048 vs independent663/2048), so raw Top counts between pools are not overfitting evidence. This analysis records actual completed epochs/checkpoint hashes; it does not claim fresh full-model/Adam/RNG CPU-load acceptance for the three new cells.

Remote analysis receipt: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/viewed2048-expanded-depth-comparison-20261022-v1/complete.json`, 358019B, SHA `a16da461b89e4a8b83f5bf8aacd42da62d9aa0649904415b45f805c9b2fc0d03`.

ACTUAL-SUMMARY.json contains full classwise metrics and paired intervals. actual-result.json retains prediction/evaluation/checkpoint source hashes.

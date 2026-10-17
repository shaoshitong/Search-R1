# G5 DIST + group mean KL + diversity KL: all 10 epochs

Frozen actor0050; same source head40/Adam200; 5120G5 train, fixed512G5 heldout. Question+20 readout. LR5e-5; EMA.99; train-only continuous dominant hidden mixup; no new rollout. Raw/EMA full unaugmented train/test evaluation after each epoch. Training loss = within-G5 Pearson DIST + mean-correctness Bernoulli KL + 5×finite-G diversity Bernoulli KL. No individual-trajectory or binary-mixed BCE in training.

`q=1-prod(p)-prod(1-p)` is the historical binary mixed-event score. `D=4/25*sum_{j!=k}p_j(1-p_k)` is expected finite-G diversity intensity; real K=0/5→0, K=1/4→.64, K=2/3→.96. Both threshold.5 balanced accuracies are reported separately. The synthetic target is one five-vector r; both scalar targets come from r. Constant-target groups omit Pearson only; the two KL terms remain.

## Interpretation of the actual loss changes

EMA E0→10: train DIST 0.199199→0.157500; mean KL 0.701579→0.559015; diversity KL 0.408667→0.238431. Heldout DIST 0.213158→0.223236; mean KL 0.823284→0.869019; diversity KL 0.391372→0.528142. All three heldout terms finish worse, despite improved training fit. Train total rises through E2–3 then falls; heldout total peaks E3 then partially recovers but remains above its initial value.

With weight5, diversity KL accounts arithmetically for 82.20% of the train total-loss reduction and 92.45% of the heldout total-loss increase. This is a decomposition of loss values, not proof of causal impact on accuracy or gradient dominance.

The intended vector discrimination is not transferring: within-real-mixed-group Pearson rises 0.023084→0.227587 on train, but changes -0.010527→-0.058306 on heldout. The reported DIST loss averages over all questions with constant-label groups contributing zero; it is not one minus this mixed-group-only Pearson.

Diversity KL is the direct supervision of the newly requested diversity strength. DIST constrains which of the ordered five trajectories should be higher/lower; mean KL anchors correctness scale and separates all-correct from all-wrong. None alone is sufficient. The monitored historical binary mixed-event q is a different functional from D, so falling diversity KL is not a guarantee that q-threshold accuracy increases. All three terms were trained together here; identifying the largest causal contributor would require matched ablations, which this run does not provide. No additional training has been launched.

10 complete epochs, 1000 new Adam/rank, source200→final1200, EMA1000. Actual native3474 COMPLETED0:0; final raw/EMA/Adam/eight RNG independently full-file SHA and CPU load passed; two owned birth/cgroup/GPU release checks passed. Final checkpoint SHA4185b47862e526bfc3faf97f6e486add3692e04f9ee3f693f52df2def6a72ea7. Frozen runtime3f75afdd; final code/results publication is separate.

## EMA post-update mixed classification

| Epoch | Train q BA % | Test q BA % | Train D BA % | Test D BA % | Train total | Test total |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54.49 | 51.65 | 52.56 | 51.69 | 2.9441 | 2.9933 |
| 1 | 56.52 | 54.09 | 55.83 | 55.50 | 3.7000 | 3.4917 |
| 2 | 57.23 | 53.58 | 56.55 | 53.44 | 4.0515 | 4.0821 |
| 3 | 58.57 | 49.36 | 58.25 | 49.07 | 4.0065 | 4.3846 |
| 4 | 64.19 | 52.71 | 63.13 | 52.64 | 3.3535 | 4.0090 |
| 5 | 69.81 | 51.83 | 67.01 | 53.76 | 2.8667 | 3.7404 |
| 6 | 75.74 | 53.20 | 71.90 | 53.37 | 2.4005 | 3.6806 |
| 7 | 78.42 | 51.62 | 75.27 | 55.37 | 2.3778 | 3.6887 |
| 8 | 81.20 | 51.21 | 78.61 | 53.42 | 2.2669 | 3.8303 |
| 9 | 83.07 | 55.05 | 80.46 | 55.03 | 2.1010 | 3.7325 |
| 10 | 84.89 | 54.00 | 82.55 | 54.85 | 1.9087 | 3.7330 |

## EMA component losses on real unaugmented data

| Epoch | Train DIST | Test DIST | Train mean KL | Test mean KL | Train diversity KL | Test diversity KL | Train vector MAE | Test vector MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1992 | 0.2132 | 0.7016 | 0.8233 | 0.4087 | 0.3914 | 0.3498 | 0.3813 |
| 1 | 0.1935 | 0.2080 | 0.6663 | 0.7894 | 0.5680 | 0.4989 | 0.3427 | 0.3925 |
| 2 | 0.1934 | 0.2040 | 0.6265 | 0.7436 | 0.6463 | 0.6269 | 0.3503 | 0.4073 |
| 3 | 0.1913 | 0.2089 | 0.5768 | 0.6858 | 0.6477 | 0.6980 | 0.3490 | 0.4077 |
| 4 | 0.1843 | 0.2184 | 0.5704 | 0.6876 | 0.5198 | 0.6206 | 0.3315 | 0.3907 |
| 5 | 0.1749 | 0.2244 | 0.5995 | 0.7698 | 0.4185 | 0.5492 | 0.3212 | 0.3831 |
| 6 | 0.1709 | 0.2207 | 0.6151 | 0.8343 | 0.3229 | 0.5251 | 0.3112 | 0.3811 |
| 7 | 0.1673 | 0.2236 | 0.5762 | 0.8053 | 0.3268 | 0.5320 | 0.3032 | 0.3805 |
| 8 | 0.1635 | 0.2264 | 0.5318 | 0.7717 | 0.3143 | 0.5664 | 0.2914 | 0.3820 |
| 9 | 0.1611 | 0.2266 | 0.5593 | 0.8412 | 0.2761 | 0.5329 | 0.2870 | 0.3804 |
| 10 | 0.1575 | 0.2232 | 0.5590 | 0.8690 | 0.2384 | 0.5281 | 0.2815 | 0.3793 |

## RAW post-update mixed classification

| Epoch | Train q BA % | Test q BA % | Train D BA % | Test D BA % | Train total | Test total |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54.49 | 51.65 | 52.56 | 51.69 | 2.9441 | 2.9933 |
| 1 | 49.96 | 49.78 | 49.99 | 50.77 | 8.0922 | 7.2230 |
| 2 | 49.92 | 48.18 | 50.14 | 47.94 | 5.2627 | 5.4693 |
| 3 | 56.80 | 45.38 | 56.48 | 46.17 | 4.2442 | 4.6611 |
| 4 | 55.35 | 51.41 | 54.02 | 49.90 | 3.3929 | 3.9882 |
| 5 | 71.95 | 52.22 | 70.72 | 49.33 | 3.0460 | 4.2353 |
| 6 | 58.65 | 54.03 | 55.89 | 52.71 | 3.7143 | 4.9981 |
| 7 | 69.51 | 53.95 | 66.02 | 52.02 | 2.7379 | 3.7449 |
| 8 | 59.51 | 51.10 | 58.86 | 52.22 | 3.6826 | 4.5767 |
| 9 | 79.71 | 54.52 | 76.59 | 51.60 | 2.3093 | 3.8420 |
| 10 | 78.68 | 50.66 | 73.60 | 51.28 | 2.0906 | 3.8669 |

## RAW component losses on real unaugmented data

| Epoch | Train DIST | Test DIST | Train mean KL | Test mean KL | Train diversity KL | Test diversity KL | Train vector MAE | Test vector MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1992 | 0.2132 | 0.7016 | 0.8233 | 0.4087 | 0.3914 | 0.3498 | 0.3813 |
| 1 | 0.1986 | 0.2116 | 3.1609 | 3.1372 | 0.9465 | 0.7749 | 0.6042 | 0.6086 |
| 2 | 0.1934 | 0.2042 | 0.6846 | 0.7404 | 0.8769 | 0.9049 | 0.4088 | 0.4531 |
| 3 | 0.1879 | 0.2140 | 0.6482 | 0.7027 | 0.6816 | 0.7489 | 0.3610 | 0.4173 |
| 4 | 0.1777 | 0.2291 | 1.0219 | 1.2669 | 0.4387 | 0.4984 | 0.3337 | 0.3757 |
| 5 | 0.1723 | 0.2229 | 0.5488 | 0.7360 | 0.4650 | 0.6553 | 0.3165 | 0.3985 |
| 6 | 0.1744 | 0.2181 | 1.3564 | 1.7023 | 0.4367 | 0.6156 | 0.3285 | 0.3750 |
| 7 | 0.1678 | 0.2252 | 0.7746 | 1.0231 | 0.3591 | 0.4993 | 0.3096 | 0.3762 |
| 8 | 0.1673 | 0.2255 | 0.5543 | 0.7423 | 0.5922 | 0.7218 | 0.3074 | 0.4198 |
| 9 | 0.1611 | 0.2231 | 0.5614 | 0.8005 | 0.3174 | 0.5637 | 0.2807 | 0.3803 |
| 10 | 0.1564 | 0.2202 | 0.7544 | 1.1261 | 0.2360 | 0.5041 | 0.2876 | 0.3770 |

## Online loss on augmented training batches

These batch losses are recorded before each batch update and use synthetic inputs; they are not the post-update fulltrain/heldout losses above.

| Epoch | Updates/rank | Mean DIST | Mean mean-KL | Mean diversity-KL | Mean total | First total | Last total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0.3453 | 2.0007 | 0.9979 | 7.3353 | 3.0253 | 7.3064 |
| 2 | 100 | 0.3362 | 1.3791 | 0.8626 | 6.0282 | 8.2222 | 4.4526 |
| 3 | 100 | 0.3264 | 1.0031 | 0.6498 | 4.5787 | 4.4445 | 4.5739 |
| 4 | 100 | 0.3297 | 0.7764 | 0.5957 | 4.0847 | 3.8164 | 3.3372 |
| 5 | 100 | 0.3252 | 0.7055 | 0.4869 | 3.4650 | 2.8096 | 3.0934 |
| 6 | 100 | 0.3106 | 0.7529 | 0.4782 | 3.4543 | 2.7884 | 2.8239 |
| 7 | 100 | 0.3075 | 0.7655 | 0.5218 | 3.6822 | 6.3982 | 3.5321 |
| 8 | 100 | 0.3088 | 0.7674 | 0.5180 | 3.6662 | 1.8289 | 4.8889 |
| 9 | 100 | 0.3058 | 0.6262 | 0.4132 | 2.9978 | 4.5144 | 3.3453 |
| 10 | 100 | 0.3050 | 0.6706 | 0.4097 | 3.0240 | 2.1916 | 2.3211 |

Best EMA heldout q BA: 55.05% at epoch9; best EMA heldout D BA: 55.50% at epoch1. These are observed maxima on the same repeatedly monitored diagnostic set, not an independent final quality estimate.

All model/optimizer/native-hidden/trajectories/fullpredictions remain remote. Native terminal, final checkpoint fullSHA/CPU state and owned-resource-release evidence are separate small summaries; they do not establish model quality or an independent persistent backup. Formal actor training remains stopped.

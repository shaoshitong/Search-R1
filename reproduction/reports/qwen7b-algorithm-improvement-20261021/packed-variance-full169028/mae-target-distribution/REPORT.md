# Full-training random-M target distribution

Read only the actual accepted 169,028-question own-G10 train-group metadata. No checkpoint, features, final pool, GPU process or training configuration was accessed or changed.

The probability mass is calculated exactly with rational arithmetic: uniform question, independent uniform M in 3..10, and an EM-blind uniform M-subset without replacement from each question’s own 10 trajectories. For K correct trajectories, P(j|K,M)=C(K,j)C(10−K,M−j)/C(10,M). The variance is v=(j/M)(1−j/M), while the actual training target is 4v.

| Own-G10 correct K | Questions |
|---:|---:|
| 0 | 68285 |
| 1 | 12888 |
| 2 | 9119 |
| 3 | 7981 |
| 4 | 7625 |
| 5 | 7727 |
| 6 | 8267 |
| 7 | 9137 |
| 8 | 10698 |
| 9 | 12858 |
| 10 | 14443 |

| M | P(v=0) | E[v] | Median(v) |
|---:|---:|---:|---:|
| 10 | 48.943370% | 0.087727891 | 9/100 |
| 3 | 70.757370% | 0.064983623 | 0 |
| 4 | 64.612394% | 0.073106576 | 0 |
| 5 | 60.268703% | 0.077980348 | 0 |
| 6 | 56.981650% | 0.081229529 | 0 |
| 7 | 54.378910% | 0.083550373 | 0 |
| 8 | 52.250265% | 0.085291005 | 0 |
| 9 | 50.466550% | 0.086644831 | 0 |

Mixture P(v=0)=57.332401%, E[v]=0.08006427199, and median(v)=0. The actual 4v target has mean 0.32025708796 and median 0.

| Constant predictor | Expected MAE on v | Expected MSE on v | Expected MAE on actual 4v | Expected MSE on actual 4v |
|---|---:|---:|---:|---:|
| mean | 0.09180553975 | 0.00973230999 | 0.36722215900 | 0.15571695982 |
| median | 0.08006427199 | 0.01614259764 | 0.32025708796 | 0.25828156221 |
| zero | 0.08006427199 | 0.01614259764 | 0.32025708796 | 0.25828156221 |

Because more than half of this actual random-M training distribution has target zero, the L1-optimal marginal constant predictor is zero. This supports a concrete objective-level explanation for the observed near-zero MAE predictions; it does not prove a sufficiently informative conditional model must predict zero or rule out other optimization/representation issues. No classification threshold was fitted.

Source: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1/fixture/inputs/train-groups.json` (133368410 bytes), SHA256 `4528712a7dd5f34e676e2bf2c2e8746b07ebda8112ed59ba100551c4891ae6f2`.

The JSON preserves the complete target mass function, conditional distributions and exact rational values.

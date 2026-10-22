# Viewed2048 E1 and matched pilot depth comparison

Actual CPU-only analysis of existing prediction files; no new inference, model/feature tensor read, final4096 read, threshold fit, candidate freeze or Slurm mutation. All2048 questions, original own5 trajectory IDs including order, and strict EM match exactly. Identity SHA: `c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7`. Pool has663 Mixed,1055 all-wrong and330 all-correct questions.

| Model | Mixed BA | Mixed BCE | Top512 true Mixed | Mixed recall | nonMixed FPR | all-wrong→Mixed | all-correct→Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1_raw | 52.2811% | 0.643807 | 201/512 (39.2578%) | 13.8763% | 9.3141% | 90/1055 (8.5308%) | 39/330 (11.8182%) |
| E1_EMA | 52.8846% | 0.653016 | 205/512 (40.0391%) | 21.8703% | 16.1011% | 147/1055 (13.9336%) | 76/330 (23.0303%) |
| L27P20_E3_raw | 53.8842% | 0.776072 | 200/512 (39.0625%) | 89.1403% | 81.3718% | 828/1055 (78.4834%) | 299/330 (90.6061%) |
| L27P20_E3_EMA | 53.8976% | 0.674934 | 198/512 (38.6719%) | 45.7014% | 37.9061% | 376/1055 (35.6398%) | 149/330 (45.1515%) |
| L21P20_E3_raw | 57.2275% | 0.711915 | 198/512 (38.6719%) | 69.8341% | 55.3791% | 539/1055 (51.0900%) | 228/330 (69.0909%) |
| L21P20_E3_EMA | 56.3400% | 0.685009 | 207/512 (40.4297%) | 58.6727% | 45.9928% | 452/1055 (42.8436%) | 185/330 (56.0606%) |

## Paired differences

2000 paired question-bootstrap resamples, seed2026102213. Same resampled question indices for all six methods; Top512 is reselected inside every resample. Intervals are exploratory percentile95 and unadjusted for multiple comparisons. They quantify this viewed pool, not new training seeds or untouched confirmation.

| Difference | BA difference / 95% interval (percentage points) | Top512 difference / 95% interval (percentage points) |
|---|---:|---:|
| L21P20_E3_raw_minus_E1_raw | +4.9464 [+2.5343, +7.3375] | -0.5859 [-4.6875, +3.3252] |
| L21P20_E3_EMA_minus_E1_EMA | +3.4554 [+1.0452, +5.7750] | +0.3906 [-2.9297, +4.6875] |
| L21P20_E3_raw_minus_L27P20_E3_raw | +3.3433 [+1.3373, +5.3669] | -0.3906 [-3.7109, +3.3252] |
| L21P20_E3_EMA_minus_L27P20_E3_EMA | +2.4423 [+0.2883, +4.5646] | +1.7578 [-1.3672, +5.6641] |

## Concrete interpretation

L21/P20 E3 EMA raises fixed-threshold Mixed balanced accuracy versus E1 EMA by3.4554 percentage points with a positive paired interval. This comes from substantially higher Mixed recall (21.87%→58.67%), accompanied by higher nonMixed false positives (16.10%→45.99%). Its BCE is worse,0.653016→0.685009. Thus this is useful evidence for class discrimination at the current fixed threshold, and does not establish improved calibration.

Top512 rises only205→207, a difference of two questions. Its paired interval crosses zero; ranking improvement is not demonstrated. L21 raw reaches BA57.2275%, but Top512 is198, again showing that classification and top-budget ranking are different outcomes.

The same8192-question/three-epoch pilot comparison isolates the layer/capture choice more closely: L21/P20 EMA exceeds L27/P20 EMA in BA by2.4423 points with positive paired interval, but Top512 and BCE do not establish a consistent overall win. Both used identical questions, own-G5 epoch plans, seeds, head, optimizer, LR5e-6 and192 updates. The changed factor is native block21 with capture20 versus block27 with capture26; this is not proof about LR. E1 comparisons additionally differ in training budget and sampler, so cannot identify an LR or isolated-depth cause.

Next evidence-driven action: keep E1 as the ranking baseline, preserve the healthy full-data E2/E3 and natural-G5 controls, and prioritize the L20-feature/block21 branch for a matched larger-data or repeat-seed experiment if the full-data controls do not improve. Do not freeze a final candidate solely on207/512; first inspect the ongoing full-data controls and the P80 result.

Remote complete receipt: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/viewed2048-pilot-depth-comparison-20261022-v1/complete.json`, 17211 bytes, SHA `b647610831e9179dc7ff678c348ffc2be2b9c3d5773e257d029147f144ebeaad`. Analysis runtime 1.992s.

# Viewed2048 E1 and matched pilot depth comparison

CPU-only analysis of existing predictions. No new inference, model/feature tensor read, final4096 read, fitted calibration, threshold tuning, candidate freeze or Slurm mutation. All2048 questions, original own5 trajectory IDs including order and strict EM match exactly. Identity SHA: `c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7`. Pool has663 Mixed,1055 all-wrong and330 all-correct questions; true Mixed prevalence32.3730%.

| Model | Mixed BA | ROC-AUC | Mixed BCE | Top512 Mixed | Mixed recall | nonMixed FPR | all-wrong→Mixed | all-correct→Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1_raw | 52.2811% | 59.2772% | 0.643807 | 201/512 (39.2578%) | 13.8763% | 9.3141% | 8.5308% | 11.8182% |
| E1_EMA | 52.8846% | 59.5012% | 0.653016 | 205/512 (40.0391%) | 21.8703% | 16.1011% | 13.9336% | 23.0303% |
| L27P20_E3_raw | 53.8842% | 56.9518% | 0.776072 | 200/512 (39.0625%) | 89.1403% | 81.3718% | 78.4834% | 90.6061% |
| L27P20_E3_EMA | 53.8976% | 56.7325% | 0.674934 | 198/512 (38.6719%) | 45.7014% | 37.9061% | 35.6398% | 45.1515% |
| L21P20_E3_raw | 57.2275% | 58.8819% | 0.711915 | 198/512 (38.6719%) | 69.8341% | 55.3791% | 51.0900% | 69.0909% |
| L21P20_E3_EMA | 56.3400% | 58.8019% | 0.685009 | 207/512 (40.4297%) | 58.6727% | 45.9928% | 42.8436% | 56.0606% |

## Score location and calibration

Brier is mean squared probability error. ECE uses the predeclared ten equal-width probability bins; neither statistic fits any parameters.

| Model | Mean q | Median q | Brier | ECE10 | mean q, all-wrong | mean q, Mixed | mean q, all-correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1_raw | 0.432111 | 0.436382 | 0.225824 | 10.8381% | 0.420960 | 0.444437 | 0.442995 |
| E1_EMA | 0.451475 | 0.456410 | 0.230278 | 12.7745% | 0.439227 | 0.463452 | 0.466572 |
| L27P20_E3_raw | 0.592260 | 0.599251 | 0.289132 | 26.8530% | 0.575753 | 0.608592 | 0.612222 |
| L27P20_E3_EMA | 0.480866 | 0.482673 | 0.241080 | 15.7135% | 0.470943 | 0.493322 | 0.487564 |
| L21P20_E3_raw | 0.527462 | 0.529385 | 0.258399 | 20.3731% | 0.503221 | 0.552230 | 0.555198 |
| L21P20_E3_EMA | 0.501142 | 0.500168 | 0.245964 | 17.7411% | 0.487125 | 0.516016 | 0.516072 |

## Paired uncertainty

2000 paired question-bootstrap resamples, seed2026102213. All six methods use the same resampled question indices. AUC uses weighted Mann–Whitney counts with half-credit ties, equivalent to explicitly duplicated bootstrap questions. Two pure toy tests verify tie and count arithmetic. Top512 is reselected within each resample. Intervals are exploratory percentile95 without multiple-comparison correction: they quantify this viewed pool, not new training seeds or untouched confirmation.

| Candidate minus baseline | BA difference / 95% interval (pp) | AUC difference / 95% interval (pp) | Top512 difference / 95% interval (pp) |
|---|---:|---:|---:|
| L21P20_E3_raw_minus_E1_raw | +4.9464 [+2.5343, +7.3375] | -0.3953 [-2.8595, +2.1158] | -0.5859 [-4.6875, +3.3252] |
| L21P20_E3_EMA_minus_E1_EMA | +3.4554 [+1.0452, +5.7750] | -0.6994 [-2.7462, +1.2099] | +0.3906 [-2.9297, +4.6875] |
| L21P20_E3_raw_minus_L27P20_E3_raw | +3.3433 [+1.3373, +5.3669] | +1.9301 [-0.0191, +3.9853] | -0.3906 [-3.7109, +3.3252] |
| L21P20_E3_EMA_minus_L27P20_E3_EMA | +2.4423 [+0.2883, +4.5646] | +2.0694 [+0.3802, +3.8452] | +1.7578 [-1.3672, +5.6641] |

## Interpretation and next action

Against E1 EMA, L21/P20 E3 EMA improves fixed-0.5 BA by3.4554 percentage points, but its AUC is0.6994 points lower with an interval spanning zero, and Top512 rises only205→207 with an interval spanning zero. Mean q moves0.4515→0.5011 and median q0.4564→0.5002, while the actual positive prior is0.3237. BCE, Brier and ECE all worsen. These observations are consistent with a substantial score-location/fixed-threshold effect; they do not demonstrate a broader ranking improvement over E1. They also do not prove that every BA change is caused solely by a uniform offset.

L21 EMA does outperform the same8192-budget L27 EMA in AUC: +2.0694 points, paired interval[+0.3802,+3.8452]. Thus the earlier capture20/block21 choice has a useful relative discrimination signal against the weak fresh L27 pilot, rather than only a threshold shift. It still has not surpassed the existing E1 ranking baseline. Both pilots share8192 questions, own-G5 epoch plans, seeds, head, Adam, LR5e-6 and192 updates; the changed factor is capture20/block21 versus capture26/block27. E1 differs in training budget and sampler, so those comparisons do not identify isolated depth or LR causality.

The unresolved all-correct versus Mixed distinction is concrete: L21 EMA mean q is0.516016 for true Mixed and0.516072 for all-correct. Its all-correct→Mixed FPR remains56.06%; all-wrong→Mixed is42.84%. Recall has risen, but a useful calibrated selector is not established.

Keep E1 as ranking baseline and preserve the healthy full-data E2/E3 and natural-G5 controls. Inspect the already-running P80 result and these full-data controls before freezing any final candidate. L21 warrants matched larger-data/repeat-seed follow-up if needed, but neither207/512 nor BA alone justifies an untouched-final success claim. No threshold or calibrator was fitted by this analysis.

Remote complete receipt: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/viewed2048-pilot-depth-comparison-20261022-v1/complete.json`, 32806 bytes, SHA `dced81de85c30dbefd7f64065f5721baeb892ed8a5ec314fd1873b76da13a3f8`. Extended analysis runtime 2.256s.

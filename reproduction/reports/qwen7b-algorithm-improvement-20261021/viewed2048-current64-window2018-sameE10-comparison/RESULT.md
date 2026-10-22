# CURRENT64 vs WINDOW2018: identical E10/raw comparison

All four models use the same viewed2048 own-G5 questions/EM(663Mixed), the same E10/raw state,1280 global64 batch attempts, and within each seed identical10 sampling-plan SHAs and run()AST. Window capacity is0 versus2018; different skipped optimizer updates are part of this algorithmic difference. No E5/E7 model substitutes for the E10 controls.

| model | devMixed@512 | devBA at0.5 | devAUC | actualAdam updates | trainB Mixed@512 | trainBBA |
|---|---:|---:|---:|---:|---:|---:|
| CURRENT64_seed1 | 257(50.20%) | 57.67% | 0.6709 | 1280 | 449 | 67.65% |
| WINDOW2018_seed1 | 269(52.54%) | 54.84% | 0.6861 | 1278 | 415 | 55.25% |
| CURRENT64_seed2 | 262(51.17%) | 62.48% | 0.6748 | 1280 | 449 | 75.94% |
| WINDOW2018_seed2 | 273(53.32%) | 50.51% | 0.6923 | 1273 | 419 | 50.73% |

| comparison(window−current) | Top512 delta percentagepoints | paired-question95% | BA delta percentagepoints | paired-question95% |
|---|---:|---|---:|---|
| seed1 | +2.3438 | [-0.3906,+5.4688] | -2.8230 | [-4.8518,-0.7919] |
| seed2 | +2.1484 | [-1.1719,+5.2783] | -11.9765 | [-14.1745,-9.7451] |
| two_seed_average | +2.2461 | [+0.0000,+4.6875] | -7.3997 | [-9.1413,-5.6665] |

The two-seed meanTop512 yield rises50.6836%→52.9297%(+2.2461points), but the paired-question95% interval is[0.0000,+4.6875]points and each single-seed interval crosses zero. This is not evidence of a robust confirmed improvement. The0.5 BA diagnostic decreases60.0736%→52.6738%; these sigmoid ranking scores are not calibrated Mixed probabilities, and TopK-ramp does not constrain their common score offset.

Bootstrap uses2000 paired question draws and reranks each model before selecting512; the two-seed average averages per-model yields, not scores. These are conditional exploratory question-resampling intervals, not training-seed confidence intervals or untouched-question generalization. FixedtrainB is reported separately; oldtrainA is not used. No model/checkpoint/hidden/final4096 was opened; no new inference or Slurm action.

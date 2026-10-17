# Same h in training and selection: softmixedKL50, LR1e-5, all10epochs

Same original20480 cached actor0050 G5 questions, partition seed20261018:5120train/15360test, fullG5 question-disjoint. Old512 diagnostic not used. Same matched sourcehead40/Adam200/eightRNG, not earlier fitted weights. No new rollout/encoder forward. Question+20,8GPU,globalmeanBS256/fullG5/noaccum,EMA.99,train-only dominant continuous hidden mixup.

PRIMARY prediction in BOTH training and evaluation: h=1-prod(p)-prod(1-p). Soft training targetbyK=[0,.9,1,1,.9,0]. Loss=DIST(ordered5probabilities,orderedownEMvector)+50*KL(Bern(t(K))||Bern(h)). IndividualBCE,groupmeanKL,D-intensityKL are excluded. Training mixup uses the same lambda>.5 for hidden,orderedownEMvector,and convex combination of the original group targets t(K_A),t(K_B). No hardK inferred from fractional vectors. Test/realtrain evaluation uses unaugmented inputs and the original soft target, while binary mixed accuracy uses0<K<5 and h>=.5. Legacy D metrics are diagnostics only and are not primary results.

All raw/EMA epoch0..10 results below are full real post-update evaluations. Online synthetic-batch losses are separate. LR,groupobjective/softtargets,and groupweight differ from the earlier run; this is not a single-factor LR ablation. The test set is an internal heldout partition of the previously used20480 corpus, not a newly collected external final test.

## EMA mixed decision using h

| Epoch | Train h balanced % | Test h balanced % | Train h accuracy % | Test h accuracy % | Train mean h | Test mean h |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54.90 | 54.60 | 69.47 | 70.25 | 0.35489 | 0.34724 |
| 1 | 55.33 | 52.10 | 79.61 | 77.79 | 0.20468 | 0.20206 |
| 2 | 60.66 | 51.64 | 83.07 | 77.96 | 0.18994 | 0.18865 |
| 3 | 69.01 | 51.82 | 86.95 | 77.80 | 0.19385 | 0.18759 |
| 4 | 78.02 | 51.74 | 90.88 | 77.73 | 0.19509 | 0.18001 |
| 5 | 84.41 | 51.77 | 93.55 | 77.64 | 0.19563 | 0.16930 |
| 6 | 90.97 | 51.83 | 96.27 | 77.72 | 0.19893 | 0.15995 |
| 7 | 94.78 | 51.62 | 97.83 | 77.70 | 0.20031 | 0.14945 |
| 8 | 97.60 | 51.38 | 99.00 | 77.83 | 0.20219 | 0.13740 |
| 9 | 98.89 | 51.34 | 99.53 | 77.85 | 0.20237 | 0.12943 |
| 10 | 99.32 | 51.22 | 99.71 | 77.90 | 0.20174 | 0.12142 |

## EMA current-objective losses

| Epoch | Train DIST | Test DIST | Train soft h KL | Test soft h KL | Train total | Test total | Train mixed-group Pearson | Test mixed-group Pearson |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.20240 | 0.20300 | 0.57473 | 0.56168 | 28.93872 | 28.28700 | 0.01211 | -0.00614 |
| 1 | 0.20192 | 0.20435 | 0.42792 | 0.47747 | 21.59812 | 24.07784 | 0.01444 | -0.01283 |
| 2 | 0.20056 | 0.20523 | 0.34095 | 0.47185 | 17.24783 | 23.79751 | 0.02110 | -0.01723 |
| 3 | 0.19847 | 0.20579 | 0.26461 | 0.47417 | 13.42921 | 23.91430 | 0.03132 | -0.01998 |
| 4 | 0.19696 | 0.20515 | 0.20181 | 0.48192 | 10.28754 | 24.30131 | 0.03865 | -0.01681 |
| 5 | 0.19478 | 0.20520 | 0.15235 | 0.49162 | 7.81220 | 24.78621 | 0.04931 | -0.01704 |
| 6 | 0.19254 | 0.20505 | 0.11190 | 0.50404 | 5.78760 | 25.40699 | 0.06022 | -0.01632 |
| 7 | 0.18974 | 0.20452 | 0.08360 | 0.51707 | 4.36989 | 26.05807 | 0.07392 | -0.01371 |
| 8 | 0.18803 | 0.20453 | 0.06065 | 0.53431 | 3.22034 | 26.92009 | 0.08228 | -0.01373 |
| 9 | 0.18700 | 0.20441 | 0.04762 | 0.54601 | 2.56801 | 27.50491 | 0.08729 | -0.01313 |
| 10 | 0.18544 | 0.20458 | 0.03886 | 0.56296 | 2.12824 | 28.35257 | 0.09492 | -0.01398 |

## RAW mixed decision using h

| Epoch | Train h balanced % | Test h balanced % | Train h accuracy % | Test h accuracy % | Train mean h | Test mean h |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54.90 | 54.60 | 69.47 | 70.25 | 0.35489 | 0.34724 |
| 1 | 59.68 | 51.81 | 81.68 | 77.42 | 0.16689 | 0.16448 |
| 2 | 63.97 | 51.13 | 84.96 | 78.11 | 0.16222 | 0.15625 |
| 3 | 73.77 | 51.50 | 89.12 | 78.08 | 0.17719 | 0.16272 |
| 4 | 85.95 | 52.48 | 94.14 | 76.87 | 0.21182 | 0.19503 |
| 5 | 89.29 | 51.58 | 95.57 | 77.66 | 0.19194 | 0.15521 |
| 6 | 93.79 | 51.82 | 97.44 | 77.61 | 0.19950 | 0.15496 |
| 7 | 94.32 | 51.11 | 97.66 | 78.02 | 0.18257 | 0.11925 |
| 8 | 97.66 | 51.50 | 99.04 | 77.86 | 0.20004 | 0.13109 |
| 9 | 97.80 | 50.94 | 99.08 | 78.27 | 0.18308 | 0.09943 |
| 10 | 98.24 | 50.97 | 99.28 | 78.33 | 0.18634 | 0.09782 |

## RAW current-objective losses

| Epoch | Train DIST | Test DIST | Train soft h KL | Test soft h KL | Train total | Test total | Train mixed-group Pearson | Test mixed-group Pearson |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.20240 | 0.20300 | 0.57473 | 0.56168 | 28.93872 | 28.28700 | 0.01211 | -0.00614 |
| 1 | 0.20229 | 0.20609 | 0.39102 | 0.49596 | 19.75328 | 25.00411 | 0.01268 | -0.02149 |
| 2 | 0.19755 | 0.20561 | 0.28266 | 0.48526 | 14.33049 | 24.46882 | 0.03580 | -0.01910 |
| 3 | 0.19763 | 0.20494 | 0.21228 | 0.48461 | 10.81184 | 24.43529 | 0.03539 | -0.01578 |
| 4 | 0.19547 | 0.20438 | 0.16409 | 0.49322 | 8.40008 | 24.86520 | 0.04596 | -0.01299 |
| 5 | 0.19288 | 0.20492 | 0.12202 | 0.50909 | 6.29406 | 25.65940 | 0.05860 | -0.01565 |
| 6 | 0.19047 | 0.20468 | 0.09213 | 0.51486 | 4.79678 | 25.94752 | 0.07033 | -0.01448 |
| 7 | 0.18867 | 0.20405 | 0.07526 | 0.55349 | 3.95174 | 27.87839 | 0.07913 | -0.01134 |
| 8 | 0.18747 | 0.20438 | 0.06114 | 0.55142 | 3.24460 | 27.77539 | 0.08499 | -0.01298 |
| 9 | 0.18545 | 0.20465 | 0.05096 | 0.59218 | 2.73330 | 29.81383 | 0.09485 | -0.01434 |
| 10 | 0.18464 | 0.20416 | 0.04594 | 0.59994 | 2.48161 | 30.20121 | 0.09881 | -0.01193 |

## EMA test recalls and calibration

| Epoch | Mixed recall % | Nonmixed recall % | Mean h | Mean soft target |
|---:|---:|---:|---:|---:|
| 0 | 28.36 | 80.83 | 0.34724 | 0.19006 |
| 1 | 9.04 | 95.16 | 0.20206 | 0.19006 |
| 2 | 7.52 | 95.76 | 0.18865 | 0.19006 |
| 3 | 8.26 | 95.38 | 0.18759 | 0.19006 |
| 4 | 8.16 | 95.31 | 0.18001 | 0.19006 |
| 5 | 8.39 | 95.15 | 0.16930 | 0.19006 |
| 6 | 8.42 | 95.24 | 0.15995 | 0.19006 |
| 7 | 7.91 | 95.33 | 0.14945 | 0.19006 |
| 8 | 7.03 | 95.73 | 0.13740 | 0.19006 |
| 9 | 6.91 | 95.78 | 0.12943 | 0.19006 |
| 10 | 6.49 | 95.95 | 0.12142 | 0.19006 |

## RAW test recalls and calibration

| Epoch | Mixed recall % | Nonmixed recall % | Mean h | Mean soft target |
|---:|---:|---:|---:|---:|
| 0 | 28.36 | 80.83 | 0.34724 | 0.19006 |
| 1 | 8.87 | 94.75 | 0.16448 | 0.19006 |
| 2 | 5.91 | 96.36 | 0.15625 | 0.19006 |
| 3 | 6.94 | 96.06 | 0.16272 | 0.19006 |
| 4 | 11.58 | 93.37 | 0.19503 | 0.19006 |
| 5 | 7.84 | 95.31 | 0.15521 | 0.19006 |
| 6 | 8.58 | 95.06 | 0.15496 | 0.19006 |
| 7 | 6.00 | 96.22 | 0.11925 | 0.19006 |
| 8 | 7.29 | 95.70 | 0.13109 | 0.19006 |
| 9 | 5.13 | 96.75 | 0.09943 | 0.19006 |
| 10 | 5.10 | 96.84 | 0.09782 | 0.19006 |

## Online loss on augmented batches

This is before-update loss on mixed hidden tensors; not fulltrain/test evaluation.

| Epoch | New updates/rank | Mean DIST | Mean soft h KL | Mean total | First total | Last total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0.34080 | 0.51986 | 26.33392 | 22.34453 | 21.19373 |
| 2 | 100 | 0.34041 | 0.36155 | 18.41781 | 19.17917 | 21.63613 |
| 3 | 100 | 0.34176 | 0.26539 | 13.61103 | 12.58603 | 15.58004 |
| 4 | 100 | 0.34125 | 0.21975 | 11.32854 | 10.17831 | 12.85311 |
| 5 | 100 | 0.33329 | 0.17789 | 9.22763 | 6.08141 | 6.77701 |
| 6 | 100 | 0.32833 | 0.13698 | 7.17732 | 5.34081 | 6.54408 |
| 7 | 100 | 0.33357 | 0.10371 | 5.51923 | 5.32884 | 5.58317 |
| 8 | 100 | 0.33751 | 0.08347 | 4.51092 | 5.02313 | 4.16204 |
| 9 | 100 | 0.33421 | 0.07700 | 4.18398 | 4.22431 | 2.49997 |
| 10 | 100 | 0.32877 | 0.07224 | 3.94098 | 3.33004 | 4.41654 |

Best trained EMA test h BA: 52.10% at epoch1. This is a maximum on a repeatedly monitored internal diagnostic set, not independent final QA quality.

All models/Adam/nativehidden/trajectory/fullpredictions/split IDs remain remote. Local files contain only code/config/docs/small aggregate and figures. Formal actor training remains stopped.

## Actual completion and conclusion

Native3487 COMPLETED0:0; all8rank1000 newAdam/source200→1200,EMA1000. Final checkpoint independent fullSHA92ac358938830697ea8da0116bdbceec438fb5769629fcf3db49bf4b6c933db5, CPUload raw/EMA/FP32Adam/eightRNG passed; original logs inodeSHA and two ownbirth/cgroup/GPUrelease checks retained in softmix50-lr1e5-final-state-release-small-proof.json. No actor resumed.

EMAtrain hBA54.90→99.32%,test54.60→51.22%. Train soft hKL.57473→.03886; test.56168→minimum.47185 atE2→.56296. Strong training fit does not generalize: final testmixed recall6.49%,nonmixed95.95%, ordinary77.90% below79.82% majority baseline. This is evidence of a train/test generalization gap under the current protocol, not proof of a single causal mechanism. No subsequent experiment launched.

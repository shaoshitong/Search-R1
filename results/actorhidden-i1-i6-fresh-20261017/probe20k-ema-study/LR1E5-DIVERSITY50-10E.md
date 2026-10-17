# LR1e-5 and diversity KL50: ten-epoch matched rerun

Human requested LR5e-5→1e-5 and diversityKLweight5→50; everything else unchanged. Same20480 cached actor0050 data, question-level seed20261018 split5120G5train/15360G5test; no old512 diagnostic. Same original sourcehead40/Adam200/eightRNG, not previous fitted weights; newEMA starts from this source. Same head architecture/question+20,8GPU,globalBSmean256/fullG5/noaccum,10epochs/100newAdam/rank/epoch/1000new/1200final,EMA.99,training-only dominant continuous hiddenmixup. No new rollout/encoder forward or GPU tests. Existing proof reconciled before submission, unique isolated namespace.

Actual loss DIST + 50*KL(Bern(D(r))||Bern(D(p))). GroupmeanKL0 and individualBCE0 remain absent. D(v)=4/25 sum_{j!=k}v_j(1-v_k), and hard d_target=4K(5-K)/25. The only changes from3486 are learning rate and diversity weight. Legacy group_weight5 is only an evaluator diagnostic for old binary mixedBCE, not a training coefficient; diversity_weight50 is the active coefficient.

Primary mixed decision remains q=1-prod(p)-prod(1-p) threshold.5 against0<K<5. Secondary D-threshold.5 accuracy is recorded separately. True d_target requires actualEM labels and is never used as a model prediction. Both losses/metrics recorded everyepoch0..10 on complete unaugmented train/test,raw andEMA. No loss/threshold/label redefinition during the run. Because LR and weight both change, effects cannot be isolated to learning rate alone. Tests are an internal random repartition, not a new external QA benchmark. Large models/features/predictions remain remote; Mac onlysmallaggregate/docs/code/figures. Formalactor remains stopped.

Status: prepared, launch pending.

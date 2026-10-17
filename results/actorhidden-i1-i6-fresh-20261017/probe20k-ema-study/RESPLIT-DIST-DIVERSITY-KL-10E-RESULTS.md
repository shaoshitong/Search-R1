# DIST + diversity KL: 5120 train / 15360 test, all 10 epochs

Original20480 actor0050 cached questions are partitioned by question with fixed seed20261018, label-blind:5120G5 training,15360G5 test. All five sibling trajectories stay together. The original512 diagnostic is not used. Same matched initial head40/Adam200/eightRNG; prior fitted weights are not imported. These are internal heldout data from the earlier20480 corpus, not a new external final test.

LR5e-5,EMA.99,question+20,8GPU,100newAdam/rank/epoch. Training-only dominant continuous hidden mixup. No rollout or encoder forward. Loss = within-G5 Pearson DIST + 5×Bernoulli diversity KL. Groupmean KL and individual BCE are absent from optimization; evaluator retains them only as inactive diagnostics. D(v)=4/25 sum_{j!=k}v_j(1-v_k); the mixed soft5vector supplies the entire target. Constant-label groups have no Pearson term and D=0 for both all-correct/all-wrong, so these terms do not identify their absolute correctness scale.

q=1-prod(p)-prod(1-p) is the historical mixed-event probability; D is diversity intensity. Threshold.5 ordinary/balanced accuracy is computed separately for both. All tables use real, unaugmented data, fulltrain/test post-update evaluation. Loss and data partition both changed compared with the previous experiment; differences cannot be attributed solely to removing mean KL.

## Actual outcome and boundaries

Native3486 actually COMPLETED0:0. All10 epochs complete,1000new Adam per each of8ranks/source200→final1200,EMA1000. Final raw/EMA/Adam/eightRNG full-file independent SHA and actual CPUload passed, SHA686eee5ee7ac4ae26484aa57cd0b530d864128cae8796652430a15439975b131. Two ownedbirth/cgroup/GPU release checks passed. Source4f24d333 is the frozen training runtime; later result publication does not alter it. Models/fullpredictions remain remote.

EMA q BA: train 54.90→66.52%, test 54.60→51.73%. Best trained EMA test q BA is 53.99% at epoch1, below the initial score. Raw final q BA: train42.10%/test48.21%. Raw weights show large oscillation; improvement of EMA train accuracy alone does not establish stable fitting.

Train total objective 2.23601→2.95502, test 2.18677→4.00870. Training DIST improves, but diversity KL ends above its initial value on both splits. This is not only a high-train/low-test accuracy gap: fitting/calibration is unstable as well. No quality improvement is established by this setting.

True mixed prevalence: train1049/5120=20.49%,test3099/15360=20.18%. Final EMA test mean q=0.62190, mixed recall68.80%, nonmixed recall34.66%. The q decision overpredicts mixed groups. These are probe classification results, not QA EM/F1.

The first attempt3485 failed before any Adam updates because the shared evaluator's old count whitelist omitted15360; original terminal logs/fullSHA and two owned-release records were saved. The evaluator was extended to the requested fullG5 size, its actual grouping path was CPU-tested with76800 synthetic prediction rows, then3486 reran from the same source with identical split seed. No incomplete or unsaved updates were stitched. The final actual split identity hash matches the initial attempt and confirms5120/15360 disjoint questions. No GPU test or new teacher rollout was added.

## EMA classification

| Epoch | Train q BA % | Test q BA % | Train D BA % | Test D BA % | Train q accuracy % | Test q accuracy % |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54.90 | 54.60 | 52.45 | 52.39 | 69.47 | 70.25 |
| 1 | 51.67 | 53.99 | 52.54 | 54.10 | 46.45 | 47.15 |
| 2 | 48.09 | 50.48 | 48.70 | 50.33 | 32.64 | 31.13 |
| 3 | 50.94 | 50.59 | 50.91 | 50.60 | 33.18 | 29.08 |
| 4 | 53.78 | 50.56 | 53.21 | 50.93 | 38.03 | 32.78 |
| 5 | 56.31 | 51.11 | 56.90 | 51.53 | 44.71 | 37.55 |
| 6 | 58.83 | 51.55 | 59.77 | 52.18 | 46.35 | 37.29 |
| 7 | 54.88 | 50.06 | 55.97 | 50.37 | 39.90 | 32.36 |
| 8 | 61.31 | 51.78 | 61.45 | 52.33 | 50.41 | 39.10 |
| 9 | 65.50 | 51.70 | 65.33 | 51.35 | 54.20 | 39.99 |
| 10 | 66.52 | 51.73 | 66.27 | 51.26 | 56.60 | 41.55 |

## EMA loss and vector relation

| Epoch | Train DIST | Test DIST | Train diversity KL | Test diversity KL | Train total | Test total | Train mixed-group Pearson | Test mixed-group Pearson |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.20240 | 0.20300 | 0.40672 | 0.39675 | 2.23601 | 2.18677 | 0.01211 | -0.00614 |
| 1 | 0.19287 | 0.20164 | 0.73011 | 0.67002 | 3.84341 | 3.55172 | 0.05864 | 0.00060 |
| 2 | 0.19269 | 0.20104 | 0.89227 | 0.89954 | 4.65405 | 4.69876 | 0.05952 | 0.00358 |
| 3 | 0.18888 | 0.19904 | 0.85839 | 0.90503 | 4.48081 | 4.72419 | 0.07810 | 0.01347 |
| 4 | 0.18281 | 0.20093 | 0.78910 | 0.84055 | 4.12829 | 4.40368 | 0.10774 | 0.00410 |
| 5 | 0.17834 | 0.20113 | 0.68964 | 0.75501 | 3.62656 | 3.97620 | 0.12955 | 0.00312 |
| 6 | 0.17378 | 0.20060 | 0.67161 | 0.78270 | 3.53181 | 4.11411 | 0.15183 | 0.00572 |
| 7 | 0.17235 | 0.20127 | 0.77625 | 0.87491 | 4.05359 | 4.57583 | 0.15880 | 0.00244 |
| 8 | 0.17038 | 0.20129 | 0.63256 | 0.77117 | 3.33320 | 4.05714 | 0.16842 | 0.00234 |
| 9 | 0.16713 | 0.20193 | 0.58926 | 0.76615 | 3.11344 | 4.03268 | 0.18424 | -0.00083 |
| 10 | 0.16562 | 0.20174 | 0.55788 | 0.76139 | 2.95502 | 4.00870 | 0.19163 | 0.00010 |

## RAW classification

| Epoch | Train q BA % | Test q BA % | Train D BA % | Test D BA % | Train q accuracy % | Test q accuracy % |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54.90 | 54.60 | 52.45 | 52.39 | 69.47 | 70.25 |
| 1 | 51.61 | 52.15 | 51.53 | 51.64 | 57.15 | 56.24 |
| 2 | 50.58 | 49.81 | 50.36 | 49.75 | 76.09 | 76.97 |
| 3 | 49.87 | 49.80 | 49.74 | 49.99 | 55.06 | 55.54 |
| 4 | 48.34 | 49.38 | 48.45 | 49.58 | 68.95 | 71.48 |
| 5 | 51.26 | 49.63 | 50.52 | 49.88 | 75.61 | 76.08 |
| 6 | 44.72 | 49.05 | 45.21 | 48.91 | 60.43 | 64.84 |
| 7 | 57.67 | 53.36 | 56.42 | 52.95 | 61.39 | 55.55 |
| 8 | 62.99 | 51.93 | 60.77 | 51.56 | 61.29 | 50.05 |
| 9 | 49.86 | 50.20 | 49.84 | 50.12 | 76.88 | 78.88 |
| 10 | 42.10 | 48.21 | 43.56 | 48.67 | 60.76 | 67.52 |

## RAW loss and vector relation

| Epoch | Train DIST | Test DIST | Train diversity KL | Test diversity KL | Train total | Test total | Train mixed-group Pearson | Test mixed-group Pearson |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.20240 | 0.20300 | 0.40672 | 0.39675 | 2.23601 | 2.18677 | 0.01211 | -0.00614 |
| 1 | 0.19501 | 0.20308 | 0.82473 | 0.69989 | 4.31868 | 3.70252 | 0.04819 | -0.00657 |
| 2 | 0.19696 | 0.20227 | 0.70858 | 0.69629 | 3.73987 | 3.68372 | 0.03865 | -0.00253 |
| 3 | 0.18751 | 0.19915 | 0.68802 | 0.65035 | 3.62762 | 3.45092 | 0.08481 | 0.01291 |
| 4 | 0.18355 | 0.20074 | 0.61345 | 0.53895 | 3.25079 | 2.89549 | 0.10414 | 0.00507 |
| 5 | 0.18114 | 0.20051 | 0.63393 | 0.64356 | 3.35081 | 3.41832 | 0.11590 | 0.00621 |
| 6 | 0.17377 | 0.20092 | 0.64740 | 0.56273 | 3.41077 | 3.01456 | 0.15186 | 0.00417 |
| 7 | 0.17805 | 0.20107 | 0.55980 | 0.63277 | 2.97708 | 3.36492 | 0.13095 | 0.00342 |
| 8 | 0.17198 | 0.20152 | 0.54701 | 0.68734 | 2.90701 | 3.63823 | 0.16060 | 0.00116 |
| 9 | 0.18649 | 0.20180 | 0.86132 | 0.90953 | 4.49309 | 4.74946 | 0.08979 | -0.00019 |
| 10 | 0.17436 | 0.20097 | 0.73467 | 0.61716 | 3.84771 | 3.28676 | 0.14896 | 0.00392 |

## Online synthetic-batch objective

Before-update batch losses on mixed hidden inputs; do not compare directly to full real-data evaluation above.

| Epoch | Actual new updates/rank | Mean DIST | Mean diversity KL | Mean total | First total | Last total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0.34028 | 0.98596 | 5.27010 | 1.89599 | 5.49530 |
| 2 | 100 | 0.33863 | 0.82925 | 4.48487 | 5.15032 | 7.98841 |
| 3 | 100 | 0.33284 | 0.68961 | 3.78091 | 4.45923 | 2.67222 |
| 4 | 100 | 0.33129 | 0.67434 | 3.70299 | 3.00481 | 3.51700 |
| 5 | 100 | 0.31883 | 0.67643 | 3.70097 | 3.75136 | 2.97185 |
| 6 | 100 | 0.31437 | 0.64937 | 3.56121 | 4.21489 | 2.87841 |
| 7 | 100 | 0.32430 | 0.91617 | 4.90515 | 2.93148 | 2.84039 |
| 8 | 100 | 0.31648 | 0.63122 | 3.47259 | 3.20144 | 3.71557 |
| 9 | 100 | 0.31347 | 0.60883 | 3.35764 | 2.94307 | 1.96044 |
| 10 | 100 | 0.31577 | 0.68173 | 3.72440 | 5.73736 | 8.26038 |

Best observed EMA test q BA: 54.60% at epoch0; D BA: 54.10% at epoch1. These maxima use the repeatedly monitored internal test set.

All tensors/weights/optimizer/nativehidden/fullpredictions/split identities remain remote. Mac retains only code/config/docs/small aggregates and figures. Formalactor remains stopped; this is the authorized finite probe ablation.

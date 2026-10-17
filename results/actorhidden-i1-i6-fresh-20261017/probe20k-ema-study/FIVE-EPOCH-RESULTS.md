# Frozen actor0050 probe: five-epoch results
Date:2026-10-18 Asia/Shanghai. All figures are probe classification, not actor QA EM/F1.

Latest human setting:5120 distinct train questionsG5, fixed512 heldout questionsG5, full question+20 readout; copied block35+Linear; source head40/Adam200; LR5e-5; EMA.99; loss trajectory BCE +5*mixed BCE; compatible hidden/true soft-label mixup Beta(.2,.2), training only. All evaluations use original unaugmented features/labels. No new rollout or actor update. Five epochs100 actual Adam/rank each; final700/rank and EMA500. Native3473 COMPLETED0:0, final fullSHA/CPUstate/eightRNG/ownedrelease verified.

## Latest5120 group5+mixup, LR5e-5

|Epoch|EMA train mixed BA|EMA test mixed BA|Raw train mixed BA|Raw test mixed BA|EMA train trajectory acc|EMA test trajectory acc|
|---|---:|---:|---:|---:|---:|---:|
|0|54.49%|51.65%|54.49%|51.65%|65.00%|62.54%|
|1|53.65%|50.86%|46.13%|47.85%|66.63%|61.33%|
|2|54.57%|50.46%|52.19%|46.62%|66.59%|62.89%|
|3|55.95%|50.61%|53.98%|54.23%|66.55%|61.56%|
|4|59.49%|53.62%|58.38%|47.50%|67.43%|61.33%|
|5|60.99%|52.37%|50.01%|48.45%|68.00%|60.74%|

|Epoch|EMA train traj BCE|EMA test traj BCE|EMA train mixed BCE|EMA test mixed BCE|EMA train mixed ordinary acc|EMA test mixed ordinary acc|
|---|---:|---:|---:|---:|---:|---:|
|0|0.8217|0.9458|0.6209|0.6041|69.45%|69.73%|
|1|0.8151|0.8746|1.0072|0.9716|53.54%|49.22%|
|2|0.8180|0.8695|0.9889|1.0191|52.73%|48.05%|
|3|0.7871|0.8591|0.9519|1.0343|53.07%|48.83%|
|4|0.8293|0.9548|0.8116|0.9018|60.62%|56.25%|
|5|0.7840|0.9054|0.9001|0.9865|57.23%|50.00%|

## Earlier matched20480-question comparison

Identical frozen actor50, source head40/Adam200/RNG, dataset/order/test; LR1e-5, EMA.99,400 updates per epoch. A baseline is intentionally stopped; use E0..5 only. B and C each completed5 epochs,2000 new/final2200 actual Adam/rank, native3472 COMPLETED0:0; final states/fullSHA/CPU/eightRNG and ownrelease verified.

|Epoch|A EMA test mixed BA|B group2 EMA test mixed BA|C mixup EMA test mixed BA|
|---|---:|---:|---:|
|0|51.65%|51.65%|51.65%|
|1|52.07%|54.03%|52.19%|
|2|53.05%|53.01%|53.49%|
|3|51.66%|52.92%|53.98%|
|4|51.69%|53.65%|53.89%|
|5|52.93%|51.80%|50.89%|

### data_ema

|Epoch|EMA train BA|EMA test BA|Raw test BA|EMA train traj BCE|EMA test traj BCE|EMA train mixed BCE|EMA test mixed BCE|
|---|---:|---:|---:|---:|---:|---:|---:|
|0|54.68%|51.65%|51.65%|0.8216|0.9458|0.6083|0.6041|
|1|64.40%|52.07%|50.43%|0.7333|0.8947|0.5654|0.6339|
|2|74.45%|53.05%|51.57%|0.6776|0.8818|0.4849|0.6585|
|3|81.23%|51.66%|54.25%|0.6403|0.9103|0.4136|0.6723|
|4|85.34%|51.69%|51.60%|0.6060|0.9113|0.3639|0.7006|
|5|88.05%|52.93%|51.13%|0.5893|0.9546|0.3178|0.7046|

### data_ema_group2

|Epoch|EMA train BA|EMA test BA|Raw test BA|EMA train traj BCE|EMA test traj BCE|EMA train mixed BCE|EMA test mixed BCE|
|---|---:|---:|---:|---:|---:|---:|---:|
|0|54.68%|51.65%|51.65%|0.8216|0.9458|0.6083|0.6041|
|1|61.37%|54.03%|50.32%|0.9050|1.0554|0.4607|0.5482|
|2|74.20%|53.01%|51.26%|0.8573|1.0316|0.3812|0.5632|
|3|84.69%|52.92%|50.83%|0.8356|1.0524|0.3131|0.5822|
|4|91.07%|53.65%|51.94%|0.8164|1.0556|0.2631|0.5934|
|5|94.93%|51.80%|51.53%|0.8139|1.0987|0.2187|0.5973|

### data_ema_mixup

|Epoch|EMA train BA|EMA test BA|Raw test BA|EMA train traj BCE|EMA test traj BCE|EMA train mixed BCE|EMA test mixed BCE|
|---|---:|---:|---:|---:|---:|---:|---:|
|0|54.68%|51.65%|51.65%|0.8216|0.9458|0.6083|0.6041|
|1|62.53%|52.19%|50.09%|0.7465|0.9031|0.5729|0.6287|
|2|70.82%|53.49%|52.06%|0.7081|0.8980|0.5049|0.6371|
|3|77.64%|53.98%|53.71%|0.6775|0.9214|0.4473|0.6463|
|4|82.41%|53.89%|50.05%|0.6430|0.9144|0.4003|0.6740|
|5|86.02%|50.89%|52.51%|0.6290|0.9385|0.3525|0.6747|

## Mixup target incompatibility found in actual plans

For an all-correct G5 and an all-wrong G5, convex trajectory targets are all lambda, while both real mixed labels are0 so their convex mixed target is0. Fitting trajectory soft targets gives q=1-lambda^5-(1-lambda)^5>0. At lambda=.5 this is .9375 despite mixed target0. With q tied to the five probabilities, both targets cannot simultaneously attain their individual BCE minima. The group5 weight amplifies the tradeoff. This is a proved target inconsistency for these synthetic samples; it does not establish its sole causal role in all observed quality.

Analysis reads actual saved epoch plans, actual strict EM, and reconstructs the exact seeded lambdas; no model forward or new training.

|Epoch|Pairs|Allcorrect/allwrong pairs|Fraction|Those with lambda .2.. .8|
|---|---:|---:|---:|---:|
|1|2400|640|26.67%|136|
|2|2400|650|27.08%|154|
|3|2400|654|27.25%|128|
|4|2400|661|27.54%|117|
|5|2400|624|26.00%|121|

Do not silently change targets in these completed runs. A future consistent variant could retain real-group BCE on unaugmented G5 and apply mixup to trajectory supervision separately; that requires a separately authorized comparison.

## Interpretation and retention

The5120 run changes data amount, group weight, learning rate and number of updates per epoch together; do not attribute effects to a single factor. EMA train BA54.49→60.99%, heldout51.65→52.37%; raw final train50.01/test48.45%. This setting has optimization/target-fit problems as well as weak generalization, not merely the earlier high-train/low-test pattern. Full raw/EMA losses, both accuracy definitions, AUC/Brier and probability-by-K summaries are in the small JSON aggregates. No claim of quality success. Formal actor training remains human-stopped.

Weights/Adam/native hidden/raw trajectories/full predictions remain remote. Mac retains only code/config/docs and small aggregate/proof JSON. Final normal checkpoint CPU load and fullSHA are not an independent persistent checkpoint backup or a post-load GPU forward.

Frozen runtime for3472:dbf426f739227c7650848b69c9ad0b50778770cc. Frozen runtime for3473:c93f55a0382ff3191b91755fa8309b1b0af4bb70. Later report/controller commits do not hotedit either runtime.

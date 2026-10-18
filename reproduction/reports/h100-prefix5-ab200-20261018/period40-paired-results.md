# Period40: paired probe and selection results

Both independent native h100-11 runs have committed40/200. Each physical actor rank has800 actual Adam writes. A remains frozen with0 probe writes; B has560 actual raw/EMA writes, including50 writes in this period at LR5e-6. Frozen runtime2e397623 is unchanged. Original rank-journal verification and actual41 B EMA adoption are reconciled separately with the existing normal-boundary observer.

## Fresh data and selection

| Period40 | A frozen | B balanced + EMA |
|---|---:|---:|
|Candidate questions / trajectories|2560 / 12800|2560 / 12800|
|Original pool mixed / nonmixed|591 / 1969|649 / 1911|
|Selected mixed / 512|512|512|
|Diagnostic mixed / nonmixed|131 / 381|133 / 379|
|Whole wall including persistence|1036.976s|1127.952s|

The same fixed512 diagnostic questions receive fresh G5 from each arm's own actor0039. Before/after metrics within an arm use exactly the same trajectories and labels, without diagnostic training. A/B actors and outcomes differ. Both periodic selected sets are100% mixed because the reference rule prioritizes actual mixed groups after full-pool generation; this is not perfect probe classification.

B draws1280 mixed and1280 nonmixed groups from its own original2560-question pool, with1504 unique groups and41.25% repeated draws. It fits once, with no additional selected512 fit. Probe update loop6.821356s is not the full iteration time.

## B paired original-distribution metrics

| Data / head | Balanced accuracy before→after | Ordinary accuracy before→after | Trajectory BCE before→after | Mixed BCE before→after | Weighted total before→after |
|---|---:|---:|---:|---:|---:|
|train / raw|56.2655%→69.8439%|62.7344%→65.1562%|0.699040→0.419693|0.831990→0.710257|4.858989→3.970980|
|train / ema|64.0018%→65.7491%|61.5234%→62.4609%|0.461644→0.443208|0.796902→0.766664|4.446154→4.276528|
|test / raw|49.9454%→54.3942%|60.9375%→40.4297%|1.180520→0.761949|0.816236→1.218301|5.261699→6.853454|
|test / ema|50.7410%→50.7539%|42.9688%→40.8203%|0.812650→0.790593|1.079363→1.120269|6.209467→6.391940|

Training-side metrics evaluate all original unresampled2560 groups. They are not accuracy on repeated balanced draws. These are group classification metrics, not QA EM/F1.

## Interpretation

EMA training balanced accuracy improves64.0018%→65.7491%, while fresh diagnostic balanced accuracy is essentially unchanged50.7410%→50.7539%. Diagnostic ordinary accuracy falls42.9688%→40.8203%, below the74.0234% nonmixed-majority baseline. Mixed recall rises66.9173%→71.4286%, while nonmixed recall falls34.5646%→30.0792%. Mean predicted mixing rises60.8407%→63.6858% against actual133/512=25.9766%; mixed BCE1.079363→1.120269 worsens. The current results do not establish generalization or calibration gains.

Frozen A original-pool BA60.6181% and diagnostic BA53.2277% are unchanged before/after, as expected. B EMA post-update diagnostic BA across20/30/40 is53.6887%/56.7986%/50.7539%; the earlier increase is not a sustained improvement. These periods use changing actor trajectories/labels and are not a fixed-label learning curve.

## Ordinary selected mixed fraction trend

| Ordinary outer range | A frozen mean | B EMA mean |
|---|---:|---:|
|6–9|75.3418%|64.4531%|
|11–19|61.5668%|57.4219%|
|21–29|46.2240%|40.6250%|
|31–39|35.6554%|40.5382%|

Uncontinued candidates have no outcome labels on ordinary rounds, so selected ratios do not establish whole-pool accuracy or enrichment. B is higher in the recent31–39 window, but both are below early selected mixing. No isolated causal claim is made.

QA5 remains A EM27.5391%/F1 36.1807%, B EM28.5156%/F1 37.3150%. There is no QA40. QA50/100/150/200, final51713, scheduled grounding refits, later checkpoint/retention and actual terminal cleanup remain pending. Both budgets remain200/no201.

All560 globally question-weighted B update-loss records have sequential Adam/EMA clocks and total=trajectory BCE+5×mixed BCE within FP32 error. Full per-outer paired metrics and loss rows are in `through40-comparison/comparison-progress.json`. Sources and their independent small-file hashes are in `period40-paired-small-metrics.json`. Model/optimizer/full-rollout outputs remain remote.

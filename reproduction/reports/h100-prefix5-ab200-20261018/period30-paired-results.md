# Period30: fresh paired probe results

Both native h100-11 runs completed30/200, with600 actor Adam writes per physical rank. Frozen A has0 probe writes; B has420 actual probe Adam/EMA writes. Actual31 selection uses the unchanged frozen A and B EMA420, both matching the previous completed head hashes. No outcome labels were used for ordinary31 selection. Frozen training runtime2e397623 is unchanged.

## Current data and execution

| Period30 | A frozen | B balanced + EMA |
|---|---:|---:|
| Sampling actor in each arm |actor0029|actor0029|
| Candidate questions / trajectories |2560 /12800|2560 /12800|
| Original mixed / nonmixed |620 /1940|649 /1911|
| Selected mixed /512 |512|512|
| Diagnostic mixed / nonmixed |130 /382|142 /370|
| New probe writes per rank |0|50|
| Whole wall including persistence |1053.015s|1080.750s|

Four full-pool workers each completed3200 trajectories; four diagnostic workers each completed640 prefixes and640 full trajectories. Diagnostics use the same fixed512 question identity, new G5 from each arm's own actor0029, native hidden and no diagnostic training. A/B actors and outcomes differ, so paired causal comparisons here are within an arm on unchanged data.

Periodic actual-EM mixed-first selection yielded512 mixed groups/no fallback in both arms. This100% ratio is the oracle-rule outcome, not perfect probe accuracy. Ordinary selection remains before labels.

B resampled its own current pool to1280 mixed+1280 nonmixed,1513 unique groups,40.8984% repeated draws. It made50 writes once at LR5e-6, EMA.99 per write, without an extra selected512 fit. The update loop took6.751s, not the whole18-minute iteration.

## Paired original-distribution metrics

| B distribution / head | BA before→after | Ordinary accuracy before→after | Traj BCE before→after | Mixed BCE before→after | Weighted total before→after |
|---|---:|---:|---:|---:|---:|
|train / raw|37.4342%→70.3483%|43.2031%→64.9219%|1.090439→0.406445|1.202232→0.723369|7.101597→4.023292|
|train / ema|48.0166%→56.6844%|47.3047%→53.7891%|0.588031→0.472792|1.018983→0.863656|5.682944→4.791071|
|test / raw|49.6650%→54.0027%|53.9062%→43.5547%|1.411874→0.703522|0.868772→1.102523|5.755734→6.216137|
|test / ema|54.4652%→56.7986%|42.9688%→42.5781%|0.857849→0.732588|1.122567→1.166710|6.470682→6.566138|

Training metrics use all original unresampled2560 groups, not repeated balanced draws. Test means fresh512-question G5, evaluated with identical trajectories/labels before and after. These classification metrics are not QA EM/F1.

| B diagnostic head | Mixed recall before→after | Nonmixed recall before→after | Mean predicted mixing before→after |
|---|---:|---:|---:|
|raw|40.1408%→77.4648%|59.1892%→30.5405%|43.0890%→64.7162%|
|ema|80.2817%→88.7324%|28.6486%→24.8649%|66.9710%→70.5580%|

B diagnostic actual mixing is142/512=27.7344%, majority accuracy72.2656%. EMA ordinary accuracy42.5781% is below that baseline. BA improves, but nonmixed recall falls and mean predicted mixing rises to70.5580%; diagnostic mixed BCE and total worsen. Raw results similarly trade higher mixed recall for more false positives. There is no generalization-success claim or unrequested algorithm change.

Frozen A has identical pre/post metrics: original-pool BA64.2235%, diagnostic BA51.3894%, diagnostic ordinary accuracy55.8594%. A total loss uses group weight1 versus B5; do not compare raw cross-arm total loss.

## B actual update loss

| Period30 steps | Trajectory BCE | Mixed BCE | Weighted total |
|---|---:|---:|---:|
|1–10|0.817416|1.071067|6.172749|
|11–20|0.664950|0.741390|4.371898|
|21–30|0.596236|0.733034|4.261407|
|31–40|0.588080|0.710701|4.141583|
|41–50|0.605888|0.657690|3.894337|

Rows are globally question-weighted across four ranks and averaged over different minibatches. All420 global loss rows through30 have sequential Adam/EMA clocks and satisfy total=trajectory BCE+5×mixed BCE within FP32 error. This is not repeated evaluation of fixed data.

## Continuing comparison

- A ordinary6–29 excluding10/20 (22 rounds): mean selected mixed57.7947%.
- B ordinary6–29 excluding10/20 (22 rounds): mean selected mixed51.8288%.

Uncontinued candidates have no true labels, so these selected ratios do not show full-candidate accuracy or enrichment. Across periods actor and diagnostic labels change; this is not a fixed-label learning curve. QA5 remains A EM27.5391%/F1 36.1807%, B EM28.5156%/F1 37.3150%. There is no QA30. QA50/100/150/200, final51713, normal savepoint checks and final state/resource closure remain pending. Both runs continue200/no201.

Source hashes and actual31 model adoption are in `period30-paired-small-metrics.json`; committed small records and420 global loss rows are in `through30-small-metrics-and-loss-curves.json`. Models/full outputs remain remote.

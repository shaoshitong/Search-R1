# Mixed discrimination failure: sealed epoch1 versus epoch2

The requested primary endpoint is Mixed balanced accuracy = (Mixed recall + nonMixed recall)/2, followed by independent predicted-q Top512 true Mixed yield. Trajectory accuracy cannot substitute for either endpoint.

| Metric | E1 Mixed-only | E2 balanced, rank weight1 |
|---|---:|---:|
| Fixed512 Mixed balanced accuracy (q>=0.5) |51.3864%|53.5235%|
| Fixed512 Mixed recall |97.5610%|73.1707%|
| Fixed512 nonMixed recall |5.2117%|33.8762%|
| Independent2048 Mixed balanced accuracy |50.3138%|56.3608%|
| Independent2048 Mixed ROC AUC |0.533673|0.578071|
| Independent2048 average precision |0.340901|0.356370|
| Independent2048 predicted-q Top512 trueMixed |182/512 (35.546875%)|182/512 (35.546875%)|

Independent2048 has641 Mixed,1407 nonMixed. At q>=0.5 E2: TP482,FN159,FP879,TN528. Thus24.805% of Mixed are missed and62.473% of nonMixed are false positives. Fixed512: TP150,FN55,FP203,TN104;26.829% of Mixed missed and66.124% of nonMixed false positives. Both error directions exist; false positives dominate.

## The repeated182 is genuine, not cached evaluation

Full prediction SHAs differ and all2048 five-probability vectors differ. Independent recomputation verified q formula, trajectory IDs, strict EM alignment and score sorting. Only155/512 selected question IDs overlap;357 are replaced. TrueMixed intersection58;124 leave and124 enter. E1 selected histogram K0..5=[249,44,33,49,56,81]; E2=[217,42,33,36,71,113]. Thirty-two fewer allwrong are exactly offset by32 more allcorrect, not more Mixed.

## Ranking failure, not only threshold calibration

E2 selected meanq=0.921303 but actualMixed=0.355469. In fixed512, allcorrect meanq0.705521 exceeds realMixed0.682427. Allcorrect65/79 and allwrong138/228 are false positives. Best threshold fitted on fixed512 raises its balanced accuracy only to58.0972%; applying that threshold to independent2048 gives55.6472%. This is diagnostic analysis, not a training/selection change, and no threshold was fitted to independent2048 for deployment. Threshold adjustment cannot alter Top512 order.

Training balanced accuracy73.8221% versus fixed diagnostic53.5235% additionally indicates a substantial generalization gap. Ranking accuracy within realMixed is69.4009% train versus56.2874% fixed diagnostic. Fitting/trajectory scores alone are insufficient.

## Confirmed objective conflict; contribution to end-to-end failure remains a hypothesis

For negative groups, let A=prod(p), B=prod(1-p), y=0 or1, and G=5. The current trajectory mean BCE + binary Mixed BCE has per-logit gradient (p-y)/5 + p-A/(A+B). Ranking is zero on these groups. Binary Mixed loss knows only that the group is homogeneous and does not distinguish the correct extreme. Its gradient can overwhelm own-EM BCE in the wrong direction.

At actual sealed E2 independent2048 outputs, allcorrect489/1640 trajectory gradients (29.8171%) point down, contrary to own EM1; allwrong683/5395 (12.6608%) point up, contrary to own EM0.132/328 allcorrect groups and185/1079 allwrong groups contain at least one such trajectory. These are exact output-logit gradients at EMA predictions, not a replayed shared-parameter optimizer update; do not claim they alone prove model trapping or explain all generalization error.

The existing synthetic full-Hessian/Adam test also establishes a wrong-extreme local minimum of the current independent-logit objective. Increasing ranking weight cannot directly repair negative groups because their ranking gradient is zero.

## Concrete candidate, not yet applied to live training

The prepared three-event target distinguishes allwrong, Mixed and allcorrect using the same five p_j. It retains the Mixed loss exactly; for negatives it replaces -log(A+B) by -log(B) or -log(A), respectively. The negative output gradient becomes1.2*(p-y), with no wrong-direction components. This addresses the verified ambiguity without a new head or changed inference q. Existing synthetic CPU tests passed; there is no claim yet of improved heldout Mixed accuracy or selection yield, and no live code/config hot edit. Actual runtime qualification and a preserved-state isolated continuation are required before use. Thresholds and further recipe choices should be selected on diagnostic data; repeated use of the2048 set makes it validation rather than an untouched final test.

Active job742276 remains on immutable ea845a85 runtime. This report is analysis, not an assertion that training goal succeeded.

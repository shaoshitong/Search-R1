# Full169028 Z arm: E1–E4 measured progression

Active job749577; actual E4 evaluation snapshot time1792689165.9549766. Training has continued into E5. These are four completed epochs of a ten-epoch run, not final convergence or checkpoint restoration acceptance. No final4096 labels or checkpoint contents were opened for this report.

Target is the population strict-EM variance of the actually sampled M trajectories: v=meanEM×(1−meanEM). The head sigmoid predicts4v; the model score used to rank questions is sigmoid/4, not a Mixed probability. One question and M random Prefix20 snippets are repacked into one prefill; M is sampled3…10 independently of labels.

For each real global optimizer batch128 (tail68), both ranks autograd-all-gather the float32 predictions s and targets t=4v. Let z(x)=(x−mean(x))/max(std_population(x),10^−6). The actual objective is mean[(z(s)−z(t))²]+1.0×mean[(s−t)²]. Prediction mean/std remain differentiable; statistics are neither per-microbatch nor evaluation-wide. With nondegenerate std the first term equals2(1−Pearson) on this training batch; reported full-pool Pearson/Spearman are independently computed diagnostics, not the exact batch loss. The anchor fixes absolute variance scale. No trajectory BCE, Mixed BCE, ranking loss or class reweighting is added. Raw Adam updates first; EMA←.9995EMA+.0005raw.

All reported MSE/MAE below are in original variance v units, so the anchor MSE on4v is16×MSE_v. Fixed-train2048 questions are a previously trained subset; independent2048 questions are excluded from this fit and have663 real Mixed in the fixedG5. The fixed BA diagnostic threshold is predicted v≥.08; it was not tuned by epoch.

| Epoch | State | Train MSE_v | Train BA | Independent MSE_v | Independent Pearson | Independent Spearman | Independent BA | Top512 real Mixed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|1|raw|0.00875587|56.9870%|0.00828889|0.189752|0.191058|53.7839%|218/512 (42.5781%)|
|1|EMA|0.00879494|57.3757%|0.00883213|0.165896|0.156836|57.2725%|215/512 (41.9922%)|
|2|raw|0.00840647|60.5285%|0.00823818|0.208091|0.203084|55.5755%|224/512 (43.7500%)|
|2|EMA|0.00863435|58.5683%|0.00834652|0.197295|0.191491|54.9401%|223/512 (43.5547%)|
|3|raw|0.00822759|61.7005%|0.00820418|0.217620|0.216648|54.9089%|217/512 (42.3828%)|
|3|EMA|0.00848980|60.0365%|0.00822664|0.213054|0.210059|55.1767%|225/512 (43.9453%)|
|4|raw|0.00755645|67.0659%|0.00827856|0.228225|0.233456|59.6658%|223/512 (43.5547%)|
|4|EMA|0.00832696|61.1502%|0.00817209|0.222683|0.222955|55.0130%|220/512 (42.9688%)|

Training fit keeps improving more than independent selection. Independent EMA MSE and both correlation coefficients improve gradually, but Top512 is215→223→225→220 and BA57.27→54.94→55.18→55.01%; neither ranking yield nor threshold BA improves monotonically. Raw BA rises to59.67% at E4 while Top512 stays223. Thus there is a learning signal and no strict constant collapse, but there is no evidence yet of stable convergence or a strong independent ranking breakthrough.

E4 independent predicted-v std is.02860raw/.02001EMA versus true-v std.09273. Scores remain compressed. EMA MSE.00817209 is below the validation-oracle constant-mean baseline.00859799 by about4.96%; that oracle constant baseline uses the validation mean and is diagnostic only. Raw E4MSE worsens slightly from E3 while correlation improves, so calibrated MSE and ranking quality must be read separately.

Source commit 7ac274ac8ff5dea0a12dfd44cfb03fe57491458f; actual source SHA d786c2ca99a564adec82180321c86dd94aa2dab04e984ad4363b1f09a130d1d1. Exact metrics and source provenance are in E1-E4-raw-EMA-convergence.json; immutable evaluation metadata are in observed-through-evaluation-epoch-04.json.

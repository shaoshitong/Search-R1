# Earlier 7B feature/loss/readout pilot inventory

All **46 recorded result rows** in this assigned scope; this is not a best-checkpoint list. Archived local aggregate evidence only. No new scoring, GPU allocation, checkpoint loading or final4096 access.

Old screening runners scored **EMA only**. Their raw checkpoints existed but raw fixed-pool metrics and fixed-training BA are missing. A 512-question diagnostic metric is not training accuracy. Scalar models have no EMA.

Pool identities are explicit: old selection2048 has641 true Mixed; fixed-HF selection2048 has663; prefix-confidence INTERNAL NEWSEED5 has655; linear internal holdout has1589 distinct questions/6356 views. Never rank these pools together.

| Experiment | Checkpoint/state | Train BA | Heldout BA | AUC | Selected Mixed | Pool |
|---|---|---:|---:|---:|---:|---|
| matched-loss128/trajectory | cursor128 / EMA | missing | 51.872% | 56.009% | 187/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| matched-loss128/joint | cursor128 / EMA | missing | 53.723% | 56.463% | 175/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| matched-loss128/event | cursor128 / EMA | missing | 53.816% | 56.974% | 195/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| followup-loss128/within1 | cursor128 / EMA | missing | 54.162% | 56.890% | 185/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| followup-loss128/inter1 | cursor128 / EMA | missing | 54.522% | 57.969% | 194/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| followup-loss128/bothrank1 | cursor128 / EMA | missing | 54.524% | 57.490% | 183/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| followup-loss128/mean1 | cursor128 / EMA | missing | 53.403% | 57.159% | 189/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| followup-loss128/var5 | cursor128 / EMA | missing | 53.837% | 56.895% | 190/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| followup-loss128/mean1-var5 | cursor128 / EMA | missing | 53.346% | 57.163% | 189/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| followup-loss128/bothrank1-mean1-var5 | cursor128 / EMA | missing | 55.844% | 57.335% | 187/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| followup-loss128/inter1-hard16 | cursor128 / EMA | missing | 54.355% | 57.316% | 192/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| l27-readout128/last-prefix | cursor128 / EMA | missing | 47.926% | 48.177% | 153/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| l27-readout128/mean-prefix | cursor128 / EMA | missing | 50.616% | 49.741% | 152/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| l27-readout128/mean-question-prefix | cursor128 / EMA | missing | 50.000% | 49.066% | 154/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| l27-readout128/attention-prefix | cursor128 / EMA | missing | 50.509% | 49.743% | 153/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| l27-readout128/attention-question-prefix | cursor128 / EMA | missing | 50.000% | 49.076% | 155/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| l27-readout128/flatten | cursor128 / EMA | missing | 50.492% | 50.941% | 159/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer21 | epoch1 / EMA | missing | 50.645% | 52.553% | 174/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer21 | epoch2 / EMA | missing | 50.881% | 54.874% | 182/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer21 | epoch3 / EMA | missing | 51.940% | 55.773% | 189/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer21 | epoch4 / EMA | missing | 53.058% | 56.359% | 189/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer24 | epoch1 / EMA | missing | 49.980% | 51.976% | 182/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer24 | epoch2 / EMA | missing | 50.575% | 54.248% | 187/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer24 | epoch3 / EMA | missing | 51.296% | 54.955% | 184/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer24 | epoch4 / EMA | missing | 52.799% | 55.403% | 188/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer27 | epoch1 / EMA | missing | 51.643% | 52.283% | 169/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer27 | epoch2 / EMA | missing | 53.610% | 53.211% | 166/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer27 | epoch3 / EMA | missing | 53.654% | 53.846% | 164/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| fresh-official-layer27 | epoch4 / EMA | missing | 54.281% | 54.402% | 177/512 | historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed |
| peer8192-rank/inter1 | cursor256 / EMA | missing | 55.475% | 56.725% | 189/512 | fixed-HF selection2048 genuine fixedG5 / 663 Mixed |
| peer8192/no-rank-control | cursor256 / EMA | missing | 54.655% | 55.990% | 182/512 | fixed-HF selection2048 genuine fixedG5 / 663 Mixed |
| peer8192-rank/bothrank1 | cursor256 / EMA | missing | 55.085% | 56.318% | 186/512 | fixed-HF selection2048 genuine fixedG5 / 663 Mixed |
| frozen-L26-linear/prefix_mean/L2=0.001 | LBFGS80 / plain linear head (no EMA) | missing | 53.608% | 53.988% | missing | internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048 |
| frozen-L26-linear/prefix_mean/L2=0.01 | LBFGS80 / plain linear head (no EMA) | missing | 54.654% | 54.960% | missing | internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048 |
| frozen-L26-linear/prefix_mean/L2=0.1 | LBFGS80 / plain linear head (no EMA) | missing | 56.125% | 57.238% | missing | internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048 |
| frozen-L26-linear/prefix_last/L2=0.001 | LBFGS80 / plain linear head (no EMA) | missing | 52.321% | 52.813% | missing | internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048 |
| frozen-L26-linear/prefix_last/L2=0.01 | LBFGS80 / plain linear head (no EMA) | missing | 52.910% | 53.388% | missing | internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048 |
| frozen-L26-linear/prefix_last/L2=0.1 | LBFGS76 / plain linear head (no EMA) | missing | 54.170% | 55.051% | missing | internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048 |
| frozen-L26-linear/winner-prefix_mean-L2=.1/fixed0.5 | same winner LBFGS80 / plain linear head (no EMA) | missing | 49.946% | 52.629% | 158/512 | fixed-HF external viewed2048 /663 Mixed; exact same logits both thresholds |
| frozen-L26-linear/winner-prefix_mean-L2=.1/internal-CV-threshold | same winner LBFGS80 / plain linear head (no EMA) | missing | 51.820% | 52.629% | 158/512 | fixed-HF external viewed2048 /663 Mixed; exact same logits both thresholds |
| prefix-confidence/matched_platt2 | completed auxiliary fit / auxiliary scalar model (no EMA) | missing | 63.437% | 67.054% | 229/512 | INTERNAL NEWSEED5 whole-qid holdout2048 /655 Mixed; already seen by original probe warmup |
| prefix-confidence/prefix_full8 | completed auxiliary fit / auxiliary scalar model (no EMA) | missing | 63.361% | 67.056% | 232/512 | INTERNAL NEWSEED5 whole-qid holdout2048 /655 Mixed; already seen by original probe warmup |
| prefix-confidence/raw_E4 | frozen E4 prediction reference / uncalibrated frozen E4 prediction; raw label is not raw-weight state | missing | 64.052% | 68.066% | 238/512 | INTERNAL NEWSEED5 whole-qid holdout2048 /655 Mixed; already seen by original probe warmup |
| G10-exact-uniform-all252 | cursor136 / EMA | missing | 55.391% | 57.485% | 197/512 | fixed native-backend selection2048 ownG5 /663 Mixed; not same numerical HF feature backend as peer8192 |
| G10-exact-reference/native_E4 | source reference or cursor136 / EMA | missing | 55.529% | 57.610% | 200/512 | fixed native-backend selection2048 ownG5 /663 Mixed |
| G10-exact-reference/same_G10_random4_B48x136 | source reference or cursor136 / EMA | missing | 55.148% | 57.322% | 194/512 | fixed native-backend selection2048 ownG5 /663 Mixed |

The six linear candidates were evaluated only on their internal holdout to choose one winner. Its fixed0.5 and frozen internal-CV-threshold rows share one model/logit file, not two fits. Multilayer rows include every actual epoch1–4 for all3 layers. Their same-budget training size was6528, not8192.

The58432-question descriptor and8192 G10 learnability/collision study are data analyses, not trained-model result rows. The descriptor did not itself run20epochs. No actual layer19 record was found. Root separately inventories full count/head-only20epoch and direct-Mixed families.

Each JSON row retains exact source path+SHA, budget, pool identity, missing values, checkpoint refs where present, and comparison limitations.

# Frozen final confirmation: primary Top512 rule passed

The frozen **full169028 L15/P80/MSE, E9/raw** candidate passed the predeclared primary numerical confirmation rule on both independent2048-question pools. Actual final job **750481 completed0:0 in4m54 onACD1-4**. This report verifies the primary result; complete goal closure remains with root for source/resource/evidence reconciliation.

| Final pool | Model | Mixed@512 | Fraction | BA (secondary) | Mixed recall | All-wrong→Mixed | All-correct→Mixed |
|---|---|---:|---:|---:|---:|---:|---:|
| confirmation2048 | E1_EMA | 209/512 | 40.8203% | 53.8030% | 24.0525% | 135/1024 (13.1836%) | 89/338 (26.3314%) |
| confirmation2048 | seed1::raw | 341/512 | 66.6016% | 70.5945% | 60.0583% | 245/1024 (23.9258%) | 12/338 (3.5503%) |
| confirmation2048 | seed2::raw | 331/512 | 64.6484% | 71.6771% | 65.1603% | 283/1024 (27.6367%) | 14/338 (4.1420%) |
| stability2048 | E1_EMA | 221/512 | 43.1641% | 54.3223% | 24.7407% | 136/1006 (13.5189%) | 85/367 (23.1608%) |
| stability2048 | seed1::raw | 339/512 | 66.2109% | 72.8971% | 64.1481% | 236/1006 (23.4592%) | 16/367 (4.3597%) |
| stability2048 | seed2::raw | 345/512 | 67.3828% | 72.6571% | 65.9259% | 266/1006 (26.4414%) | 17/367 (4.6322%) |

| Final pool | Two-seed mean Mixed@512 | Mean improvement vs E1 | Paired95% CI of improvement | Individual seed improvements |
|---|---:|---:|---:|---:|
| confirmation2048 | 65.6250% | +24.8047pp | [20.1172, 29.1992]pp | +25.7812pp, +23.8281pp |
| stability2048 | 66.7969% | +23.6328pp | [19.5288, 28.3203]pp | +23.0469pp, +24.2188pp |

All four individual seed×pool Top512 point differences are strictly positive. Both pool-specific paired95% interval lower bounds are strictly above0. These checks use **Top512 as primary**; BA did not replace it. All-wrong false positives remain higher than E1 at the fixed variance threshold, while all-correct false positives are markedly lower; the two error types are reported rather than hidden by the successful primary ranking result.

The saved intervals use2000 bootstrap replicates, seed20261023, the same resampled question indices across E1 and both training seeds, and Top512 re-selection in each replicate. They quantify paired-question uncertainty, **not uncertainty over training seeds**. The mean is the mean of two independently trained model metrics, not a score ensemble or a best-seed result. This local check recomputed count/fraction/mean/delta/BA arithmetic and inspected saved interval settings; it did not reopen prediction rows or recalculate bootstrap samples.

Freeze and identity checks passed locally: freeze SHA `2606a04636e282df2e011c962fe47c29c00be4460f3581f3fc03b58fd5d428ca`; prospective-rule SHA `8f4fead8f40a7e14f7ddaffb2950bea3f3e405d435c544388519c507e7ce3a27`; the raw complete bytes match the remote complete proof SHA `c7b747427b1fafe89c90afcd4f41ee2794ea8d7252b336658cd889386bb20622`. Submission, final-use marker, both pool completions and final completion all reference this same freeze. Their accepted gate fixes the same E9/raw checkpoints, two distinct training seeds/plans and identical accepted physical training input. Per-pool model questions/class totals match, both original ownG5 plan hashes are preserved, and the paired output has a bound prediction proof. Row-level identity is supported by the completed immutable scorer’s same-view input/merge checks; this local verification did not reread individual qid/trajectory/EM rows.

Original fixed ownG5 plan hashes:
- confirmation2048: `e27cebfe1e426b228d84f5ab366f44f10a0f245979c50097e56a47968c9f5f2b`.
- stability2048: `96918d21d48d86dd7df0167d5eda00be39779d330b9706d71d55699738451276`.

Thresholds remain fixed at E1 probability0.5 and candidate population variancev0.08 (=4v score0.32). Variance is not a calibrated Mixed probability. No secondary state, threshold fitting, favorable trajectory-order choice, seed selection or new parameter search was introduced.

This confirms the overall benefit of the frozen full169028 L15/P80/MSE training-and-scoring configuration against original E1. It does not isolate a causal benefit of MSE, layer choice, Prefix80 or a cache window. WINDOW2018 has a separate matched comparison whose interval touches0; this final result is not evidence for a window effect.

**Both final pools have now been used. They must not be reused for tuning or future confirmatory candidate selection.** Future changed candidates require new independent confirmation data.

Sources: [original raw completion](actual-final-complete-raw.json), [actual final metadata/accounting](actual-final-live.json), [frozen candidate bytes](actual-candidate-freeze-raw.json), [machine-readable primary acceptance](FINAL-PRIMARY-ACCEPTANCE.json). This report was produced locally from the authorized small metadata only; no network, checkpoint load, individual predictions, Slurm action or new experiment.

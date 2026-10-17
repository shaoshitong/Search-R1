# Period10: actual paired probe diagnostics

Both arms complete10/200, same native H10011 executions. Each four-rank actor has200 actual Adam writes. A selection probe stays frozen at0. B performs exactly50 new synchronous probe/EMA updates,90→140 per rank, LR5e-6, EMA.99, no extra selected-subset fit. Both enter outer11; actual score11 head adoption is pending separately.

This is current-actor prefix-probe classification, not a QA benchmark. Training uses the original unresampled2560 question groups; diagnostic evaluation uses fixed512 identities with freshG5 from the same sampling actor9. A and B have their own actor trajectories and labels. All pre/post B rows use exactly the same B data.

## B original training pool

| State | Ordinary accuracy | Mixed balanced accuracy | Trajectory BCE | Mixed BCE | Total (trajectory+5×mixed) |
|---|---:|---:|---:|---:|---:|
| Raw before | 63.98% | 52.02% | 0.9079 | 0.9518 | 5.6671 |
| Raw after | 78.05% | 77.91% | 0.3965 | 0.5193 | 2.9929 |
| EMA before | 71.17% | 65.05% | 0.6197 | 0.7002 | 4.1209 |
| EMA after | 73.67% | 69.83% | 0.5122 | 0.6108 | 3.5659 |

## B fresh fixed-question diagnostic

| State | Ordinary accuracy | Mixed balanced accuracy | Trajectory BCE | Mixed BCE | Total (trajectory+5×mixed) |
|---|---:|---:|---:|---:|---:|
| Raw before | 61.52% | 49.38% | 1.2999 | 1.2173 | 7.3867 |
| Raw after | 58.79% | 59.61% | 0.6588 | 0.7989 | 4.6532 |
| EMA before | 62.11% | 52.29% | 0.9727 | 0.9030 | 5.4877 |
| EMA after | 61.72% | 54.73% | 0.8309 | 0.7924 | 4.7927 |

B diagnostic contains191 mixed/321 nonmixed groups, majority baseline62.70%. EMA balanced accuracy improves52.29%→54.73% and raw49.38%→59.61%, while ordinary accuracy falls slightly for EMA62.11%→61.72% and raw61.52%→58.79%. Mixed recall rises, with lower nonmixed specificity. The heldout ordinary accuracies remain below the majority baseline. This is an early paired observation, not a significant or causal treatment-success claim.

## A frozen comparison

A frozen head has training mixed BA74.17%, ordinary76.33%; fresh diagnostic BA56.87%, ordinary59.57%, trajectory BCE.7501 and mixed BCE.8006. Its fresh diagnostic has201 mixed/311 nonmixed groups. Its pre/post values are identical because weights are frozen. A's reporting loss uses its original weight1, so its total is not compared directly with B's weight5 total. Components and classification metrics remain explicit.

## Sampling, true selection and cost

B original counts are912 mixed/1648 nonmixed. Fitting resamples1280/1280 completeG5,1577 unique questions,38.3984% repeated draws. Raw and EMA pre/post metrics above evaluate all original2560 groups, not the repeated fitting sequence. Every global step mean is weighted over actual question counts and reduced across all4 ranks. The50 loss records and10-step block means are in the paired JSON; different batches make the curve nonmonotonic.

Full-pool actual mixing: A898/2560=35.08%, B912/2560=35.63%. Both selected512 actual-mixed groups with no fallback. Selected100% reflects the required periodic actual-EM-priority rule, not EMA success.

Whole period10 wall time including sampling, diagnosis, actor/probe work and persistence: A1085.7319s (18m06s), B1074.7698s (17m55s). B's actual50-update loop is6.8503s; that is not the whole probe phase or whole outer. No QA10 is scheduled. QA50/100/150/200 and final51713 remain pending; full200 each remains active.

# Period20: fresh paired probe diagnostics

Both same native h100-11 processes have completed20/200 and begun preparing21.
Each of four actor ranks has400 actual Adam updates; frozen A has0 online
probe updates and balanced B has280 actual Adam/EMA updates. The twenty
completed-outer records and280 sequential B global loss rows are retained in
`through20-small-metrics-and-loss-curves.json`. Runtime2e397623 is unchanged.

## Data and actual execution

| Period20 | A frozen | B balanced + EMA |
|---|---:|---:|
| Own current actor snapshot |actor0019|actor0019|
| Full candidate questions / trajectories |2560 /12800|2560 /12800|
| Actual mixed / nonmixed groups |810 /1750|795 /1765|
| Selected actual mixed /512 |512|512|
| Fallback groups |0|0|
| Fresh fixed-question diagnostic trajectories |2560|2560|
| Diagnostic mixed / nonmixed groups |160 /352|161 /351|
| New probe Adam per rank |0|50|
| Whole wall, including persistence |1073.249s|1077.305s|

All four full-pool workers returned3200 trajectories per arm. All four fresh
diagnostic workers returned640 prefixes and640 completed trajectories per arm.
The diagnostic receipt identifies the same fixed512 questions (manifest
`1ebb2d88013ef335275ca60d86e423fd75e33a7149136525a92f1f2308a2dbbd`),
each arm's own actor0019, native actor hidden and no diagnostic training.
These are new trajectories for those fixed questions, not reused teacher labels.
A and B actors and outcomes already differ, so the causal paired comparison
here is each B head before versus after on its own unchanged period20 data.

The periodic selection rule explicitly uses actual mixed-first ranking after
all candidates complete. Its100% selected mixed is an oracle-rule outcome,
not proof of perfect probe prediction. Ordinary rounds remain label-free.

B resampled only this original2560-group pool:795mixed/1765nonmixed became
1280mixed+1280nonmixed,1539 unique groups,39.8828% repeated draws. It made
50 updates once at LR5e-6, updated EMA.99 after each actual write, and did not
add a second fit of the selected512. The50-update loop took7.035s; that is
not its whole17m57s iteration time.

## Same-data paired results

| Distribution / head | BA before | BA after | Traj BCE before→after | Mixed BCE before→after |
|---|---:|---:|---:|---:|
|train / raw|48.4179%|72.8079%|0.564980→0.397319|0.978678→0.620858|
|train / ema|66.9601%|68.1460%|0.415928→0.409803|0.703171→0.686222|
|test / raw|50.8627%|53.8824%|0.833552→0.714609|1.166199→1.077880|
|test / ema|53.1843%|53.6887%|0.720116→0.716683|1.045489→1.053752|

Train means all original, unresampled2560 groups after complete fitting, not
the balanced draw sequence. Test means the fresh512-question G5 diagnostic,
not the QA benchmark. B EMA diagnostic ordinary accuracy stayed46.875%, below
its68.5547% nonmixed majority baseline. Mixed recall70.19%→72.05% improved,
nonmixed recall36.18%→35.33% declined. EMA diagnostic mean predicted mixing
rose.610406→.615702 against the actual161/512=.314453 prevalence. Its mixed
BCE and weighted total loss worsened, even as training metrics improved.
There is no generalization-success claim.

A remains frozen: original-pool BA66.4423%, fresh-diagnostic BA50.6534%,
diagnostic ordinary accuracy54.8828%. Its pre/post metrics are identical.
A diagnostics contain160mixed groups whereas B contains161; both are based
on their own changing actor trajectories. Neither A nor B probe metrics are
QA EM/F1. A total loss uses group weight1, B weight5, so compare components
rather than raw total loss across arms.

## B actual global loss curve

| Actual steps | Trajectory BCE mean | Mixed BCE mean | Weighted total mean |
|---|---:|---:|---:|
|1–10|0.560143|0.750394|4.312114|
|11–20|0.531616|0.676326|3.913244|
|21–30|0.521133|0.637088|3.706571|
|31–40|0.513001|0.606262|3.544312|
|41–50|0.526676|0.597479|3.514071|

Each row is already question-weighted across all four ranks. Ten-step means
describe different minibatches, not repeated evaluation of one fixed dataset.
All280 stored global rows through20 match sequential actual Adam/EMA clocks
and total=trajectory BCE+5×mixed BCE within FP32 aggregation error.

## Continuing comparison

For ordinary6–19 excluding periodic10 (13 observations), selected mixed
means are A65.8053%/B59.5853%; whole-wall means A784.440s/B792.873s.
These averages exclude oracle periodic rounds and do not assert uncontinued
candidate accuracy. They remain early descriptive results.

QA5 remains A EM27.5391%/F1 36.1807%, B EM28.5156%/F1 37.3150%.
There is no scheduled QA20. QA50/100/150/200, final51713, normal checkpoints
and final resource/state acceptance are pending. The20 state is the latest
recovery candidate until a later joint completes; it has not received an
extra large-tensor audit because20 is not a normal checkpoint-audit milestone.
The same processes continue200/no201.

Small sources:

- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix5-ab-small-report-through020-20261018T011254939526Z.json`, SHA256 `068f66b867a119a999d5cd018336c67fe29fdbd4e390ab10a6cee5b4310ff36e`.
- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix5-ab-periodic20-phases-20261018T011254299138Z.json`, SHA256 `d45411338486fe79b852c11536b7674f5a47b5bd90286dcc41ccff5c3060395f`.
- All original rank-journal facts16–20: `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix5-ab-h10011-completed16-20-actual-facts.json`.

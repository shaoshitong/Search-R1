# V5 supplementary: first-day exposure of genuinely new works

This supplement is specified after the cross-old-work analysis, at the parent's
explicit request. It does not change that analysis, its primary comparison or
its protocol. Its own machine-readable protocol is frozen before fitting.
It is an exploratory extension of an already explored historical non-AI dataset.

## Target and common candidate pool

Use the same source file and four calendar origins as the completed V5 analysis:
14 training days, seven calibration days, seven test days; origin start indices
0, 14, 28 and 35 in the 63 source dates. This supplement does not impose the
63-day survivor requirement. A candidate is a unique source video with a recorded
row on its actual upload date, `date == upload_dt`, in calibration or test.
Every model uses the same observed first-day candidate set on that date. This
is not the full platform's new-work pool. Missing upload-day rows remain missing.
Observed zero exposures remain zero.

Keep all authors, including accounts unobserved during training. `seen` means
at least one source row in the current 14-day training window, regardless of
positive exposure. `unseen` is not a newly registered account and does not prove
an owner has created a new identity.

## Frozen information before calibration/test

- Historical catalogue stock Q: count unique source video IDs already observed
  by training end and uploaded by then, per account. This avoids retrospectively
  discovering an old item from a future row and adding it to training stock.
- Publication flow F: among that past-observed catalogue, count videos uploaded
  during the 14 training dates. Q/F are selected-catalogue proxies, not complete
  production or historical active inventory. They remain frozen within origin.
- Old-work history H: use only training rows for videos uploaded strictly before
  the training window began. Its denominator is the number of actually observed
  old-video-days, never a calendar grid filled with zeros. With total old-work
  exposure E, observed item-days D, global old-item-day mean mu and prior item-days
  kappa, `h=(E+kappa*mu)/(D+kappa)/mu`. No available old history implies h=1.
- Duration base b: using only training videos' observed upload-day rows, estimate
  first-day exposure means in fixed duration bins [0,7s,18s,30s,60s,infinity].
  Missing duration gets its own group. Each bin shrinks with 20 video-equivalents
  to the training global first-day mean. Test-work duration is a static metadata
  feature; its exposure, plays, likes or later feedback are not inputs.

Calibration new-work first-day outcomes choose only global grid settings.
They never update account histories or the frozen Q/F catalogue. Test new works
use none of their own feedback, at any date, as features.

## Limited, nested global grids

All scores are per candidate video and normalized within its common date pool.
Compare uniform videos; duration b; b*(1+Q)^betaQ; b*(1+F)^betaF;
b*(1+Q)^betaQ*(1+F)^betaF; b*h; and b*h*(1+Q)^betaQ*(1+F)^betaF.

Both beta grids are fixed at [-1,-0.5,0,0.5,1]. History prior item-days are
1,7,28,100, plus an explicit `no_history` candidate h=1. Thus H contains the
duration base, and HQF contains both H and QF. Choose each model's setting by
lowest calibration log loss on the joint pool, with deterministic grid order
for ties. Do not fit per-account coefficients, add fixed effects, choose by test,
or expand the grid after observing results.

## Reporting and comparison

The prespecified primary increment for this supplement is HQF minus H in held-out
exposure-weighted conditional cross entropy. Report per origin and pooled, with
paired date-block and four-origin-block bootstrap intervals. Also report the
duration-only and uniform references. All are predictive comparisons, not causal
production-cost or ranking-alpha estimates.

Evaluate `seen` and `unseen` accounts under the same globally calibrated models.
For each group, report its joint-pool log-loss contribution and a separately
labeled within-group conditional loss, renormalizing only for the latter.
Report group exposure mass as well. Joint contributions include between-group
allocation; within-group losses assess allocation inside the group. Single-video
dates, and singleton within-group dates, are listed separately and excluded from
the corresponding primary multi-candidate metric. Whole-day zero-volume pools
are retained in counts but have no exposure-weighted loss contribution.

No result identifies new registration, economic owners, account splitting,
intrinsic quality versus reinforcement, AI adoption, causal alpha, or equilibrium.
The new-work design removes the target video's own historical-feedback feature;
it does not remove catalogue selection, endogenous publication or persistent
account/content differences.

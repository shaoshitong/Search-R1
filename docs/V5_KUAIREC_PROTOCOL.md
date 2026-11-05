# V5 KuaiRec cross-work transfer protocol

Planning began 2026-11-05 02:42:07 UTC (2026-11-05 10:42:07, Asia/Shanghai).
This display copy renders the local date in the client's current timezone.
The frozen source protocol and its original hash remain unchanged.
The machine-readable `protocol.json` is frozen before the V5 analysis runs.
This is a prespecified extension of an already explored dataset, not an
independent preregistered confirmatory study.

## Question and units

Does a creator account's exposure history on one group of old works predict
future exposure to different old works by the same account? Compare this with
the target works' own history and with equally flexible shuffled account labels.
The outcome is conditional author-account exposure share among B works in a
selected panel. A work is a video; an account is `author_id`, never an economic
person or a verified owner of several accounts. No AI-cost or equilibrium effect
is identified.

## Cohort and fixed split

Use the V3 source `v3/kuairec/item_daily_features.csv`, SHA256
`45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b`.
Reproduce the metadata-defined pool: videos uploaded by 2020-07-05 with a row
on every one of the 63 observed dates. This gives 925 videos and 867 accounts.
Retain the 46 accounts with at least two pool videos. This deliberately uses
future row presence and is a retrospective, selected-survivor panel; it is not
an operational forecast of the full platform.

For each account, sort videos by SHA256(`v5-kuairec|<integer video_id>`), breaking
ties by video ID. Even ranks go to A, odd ranks to B. Both sides are nonempty.
The split depends only on IDs and account membership, never exposure outcomes.
Do not filter accounts for exposure in any train, calibration or test window.
Do not replace missing item-day rows with zeros. The complete source panel must
already contain each required row; observed zero exposures remain zeros.

## Four chronological origins

Each origin uses 14 training days, seven calibration days, then seven test days.
Starts in the sorted source-date array are fixed at indices 0, 14, 28 and 35.

| Origin | Training history | Calibration | Test |
|---|---|---|---|
| R1 | July 5–18 | July 19–25 | July 26–August 1 |
| R2 | July 19–August 1 | August 2–8 | August 9–15 |
| R3 | August 2–15 | August 16–22 | August 23–29 |
| R4 | August 9–22 | August 23–29 | August 30–September 5 |

All dates are in 2020. Histories are frozen within each origin, with no updates
from its calibration or test outcomes. Later origins may use data that have
become past observations; this is planned sequential reuse, not independent
replication across four datasets. The four test windows do not overlap.

## Models and calibration

Let `E_Aa` be training A exposures, `n_Aa` the fixed A inventory and `D=14`.
Let `mu_A = sum(E_Aa)/(D*sum(n_Aa))`. With prior-equivalent days `kappa`, use

`r_Aa = (E_Aa/n_Aa + kappa*mu_A)/(D+kappa)`.

Predict B share proportional to `n_Ba*r_Aa`. Candidate kappas are fixed at
0.5, 2, 7 and 14; choose the one with the lowest B calibration log loss.
No B training exposures enter this transferred rate. Global calibration uses
B calibration exposures, and this limited target information is disclosed.

Comparators:

1. Uniform account share.
2. Linear B quantity share, proportional to `n_Ba`.
3. Free B quantity share, proportional to `n_Ba^alpha`, with alpha in [0.05, 2].
4. B's own frozen training history, with the same per-item-rate shrinkage and
   the same calibration grid. This intentionally has target-work information:
   it is a richer-information memory benchmark, not a fair same-information
   contest against A-only history.
5. Quantity + demand + age: `n_Ba^(alpha-1) * sum_B(q_v^beta * (1+age_vd)^(-gamma))`.
   The B demand proxy is training `long_time_play_cnt/show_cnt`, shrunk with
   50 exposures to its training global mean and clipped to [1e-6,1-1e-6]. It is
   an ordinary-traffic demand proxy, not intrinsic quality. Fit alpha, beta and
   gamma on calibration outcomes only, using bounds [0.05,2], [0,4], [0,2].
   Use three fixed starting points and choose by calibration loss. If B quantity
   is constant, fix alpha=1 and report it as unidentifiable from this slice.

## Same-complexity identity comparison

Generate 999 fixed account permutations with seed 20261104, independently
within metadata strata `(n_Aa,n_Ba)`. Use each same permutation across the four
origins. It shuffles A rates, then applies the recipient's B inventory. Every
permutation selects kappa from the same four candidates on the same calibration
data. Thus matching information and calibration capacity are preserved; no
permutation is selected using test results. Report immovable strata and the
fraction of accounts eligible to move.

Primary reference: pooled test log loss of the correct mapping versus this
conditional permutation distribution. The one-sided reference rank is
`(1 + number of permuted losses <= correct loss)/1000`. This is an exploratory
label-association comparison conditional on the strata, not a causal test of
ability versus algorithmic reinforcement. Report each origin as well.

## Prespecified secondary: content-adjusted residual transfer

Added at the parent's explicit request before any V5 results were run; retain
the original raw-transfer comparison as primary. The raw transferred rate has
not controlled source-item age or demand, so it is only a coarse association.

Estimate each A/B video's ordinary-traffic demand proxy from training only,
with the same 50-exposure prior. Fit the content/quantity/age score's global
parameters on A calibration exposure only, using the same fixed bounds and
starts as above. Let `E_Aa` be its expected A training exposures, conditioning
on observed training-day A volume, and `H_Aa` the actual A training exposures.
For `rho_a=(H_Aa+kappa)/(E_Aa+kappa)`, predict B share proportional to the
content-base B share times `rho_a`. Candidate kappas are fixed multipliers
0.01, 0.1, 1 and 10 of the mean expected training count per account. Select on
B calibration only. Every within-inventory-stratum identity permutation uses
the same residual values and repeats that same calibration choice.

This secondary model uses target B training feedback for its content base and
is not an A-only-information model. The demand proxy is endogenous ordinary
feedback, so residual transfer still cannot identify ability versus algorithmic
memory. Report its results separately; do not replace a failed primary result
with a successful secondary result.

## Evaluation and interpretation

Report exposure-weighted conditional cross entropy and total variation of
aggregate B shares. Keep accounts with zero exposures. Preserve daily loss
contributions and use 2,000 paired date-block bootstrap resamples for differences;
also report origin-block resampling because there are only four origins. These
intervals describe limited temporal variation, not billions of independent
observations or uncertainty over all creators.

Correct A mapping beating shuffled mappings supports shared account-associated
predictive information across different works in this selected sample. It does
not identify latent quality, fan base, account reputation, recommendation memory,
economic ownership, costs, a causal alpha or a Nash equilibrium. B-own-history
beating A transfer additionally indicates useful target-work-specific history;
it does not by itself disprove a shared account component.

## New-upload feasibility audit

Using source metadata only, count newly uploaded catalogue items in the observed
date range; compare upload dates with first/last recorded dates and audit missing
item-day rows. Describe catalogue-restricted upload flow and cumulative stock
proxies. Count new test-window items whose account has an older observed work
in the relevant training window, without fitting a new cold-start model. Do not
equate these with full production or active inventory, do not use `visible_status`
as a historical availability record, and do not impute missing exposure rows.
No owner linkage, AI indicator or exogenous production cost is available.

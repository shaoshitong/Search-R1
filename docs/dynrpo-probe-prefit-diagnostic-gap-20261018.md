# Probe pre-fit diagnostic gap: observed facts and unresolved cause

These are Mixed/nonMixed classifier metrics (`q>=0.5`), not QA EM. Both sides
are newly sampled from the same frozen actor: actor50 at the CP50 refresh and
actor59 for the outer60 sampling. Pre-fit means before this fit, not an untrained
head; retained online Adam counts were250 and400, respectively.

| Pre-fit node | train total loss | heldout total loss | train accuracy | heldout accuracy | train mean q | heldout mean q |
|---|---:|---:|---:|---:|---:|---:|
|50|1.313850|1.914122|60.3125%|42.1875%|0.411047|0.620298|
|60|1.301267|1.906900|64.4531%|45.7031%|0.381319|0.597895|

At60, actual Mixed prevalence is454/2560=17.7344% versus100/512=19.5313%.
The nonMixed false-positive rates are665/2106=31.5764% versus251/412=60.9223%.
Positive Mixed recall is46.0352% versus73%; balanced accuracies57.2294% and
56.0388%. Thus the large raw-accuracy gap includes a large operating-threshold/
calibration shift, not just a large difference in balanced accuracy. Both BCE
components also worsen: trajectory0.554319 vs0.827756, group0.746948 vs1.079144.
The total loss gap is therefore not merely a threshold-display artifact.

Checked directly against the saved sampling records:

- All12800 train and2560 diagnostic label/prefix/feature/actor identities match
  at each node; exact qid overlap is zero. Native layer27, Prefix20/width293,
  sampler and interaction contracts agree; no independent official encoder is
  used. No evidence of stale actor replacement or shifted label joins was found.
- Source NQ/Hotpot proportions, prompt length means and canonical source-index
  deciles are close. These checks do not explain the q shift.
- Independent recomputation from all pre-fit candidate logits and own labels
  exactly reproduces the reported train loss, confusion counts and mean q.
- Previously fitted P10 question overlap is141/2560 at50 and256/2560 at60,
  excluding initial warmup. Removing these questions leaves train accuracies
  60.2315% and63.8021%, so prior P10 question memorization alone does not explain
  the gap. Initial warmup overlap has not been quantified by this audit.
- Train mean q remains around0.35--0.46 across capture ranks/640-row buckets;
  no single rank or first/last capture bucket accounts for the shift.

The numerical failure is established: the retained head assigns too much Mixed
mass to the fixed diagnostic cohort, producing many false positives. The root
cause of that systematic score shift is NOT yet established. Do not describe it
as an expected random train/test difference, merely this fit's overfitting, or a
proven absence of hidden representation bugs. A possible amplification is that
uncertain per-trajectory probabilities yield high `1-prod(p)-prod(1-p)` (five0.5
values yield0.9375), but aggregate mean q alone does not prove this mechanism.

No heldout threshold tuning, scorer relaxation, loss change or head reset was
performed. The separately human-authorized two-stage actor selection uses probe
ranking to shortlist1024, then authentic strictEM to prioritize actual Mixed512;
it does not use this classifier threshold as its final actor acceptance rule.

Actual scalar evidence remains in the task output
`outputs/hpc3-prefix5-switch-20261017/probe-fit50-60-diagnostic-distribution-audit.json`.
Raw rollouts, tensors and checkpoints remain remote. Audit operator:
`docs/hpc3-probe-diagnostic-gap-audit-20261018.py`.

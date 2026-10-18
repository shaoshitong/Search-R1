# Probe pre-fit diagnostic gap: exact warmup question coverage

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

The decisive historical difference was then verified against all1744 real
warmup feature shards' `groups.json`, checking each file against its manifest SHA.
Only metadata was read; no feature tensors, model weights or raw rollout payloads
were transported. The cache manifest SHA exactly matches the retained EMA5 head's
feature binding and the imported HPC3 reference.

| Question cohort | Current questions | Previously present in EMA5 warmup training |
|---|---:|---:|
|50 fit pool|2560|2560 (100%)|
|60 fit pool|2560|2560 (100%)|
|Fixed probe diagnostic|512|0 (0%)|

The445440 warmup G5 sampling events cover168510 distinct normalized training
questions: the entire eligible source pool. Mean warmup G5 event counts for the
current50 and60 fit questions are2.6273 and2.6313 per warmup epoch. EMA5 then
traversed the warmup bank for five epochs. Its saved train/test Mixed accuracies
were86.2536%/61.5234%, balanced85.0335%/57.0872%, and total losses0.638514/1.369222.
A substantial train/heldout gap was already in the inherited initialization.
These historical scores use the historical bank's labels, not current actor EM.

Therefore fresh actor rollouts do not make the training questions unseen by the
probe. The two current fit/diagnostic sets are disjoint, but only the fixed
probe diagnostic is held out from the full warmup history. This is a confirmed
structural reason why a large gap can already exist BEFORE a new fit; the prior
P10-only overlap check was insufficient because it omitted the much larger
warmup bank. Do not describe current pre-fit train accuracy as performance on
unseen questions or attribute the whole gap solely to the current3-epoch fit.

The actual diagnostic score shift and false positives remain real. This overlap
audit establishes historical exposure, not a controlled quantitative attribution
of every loss difference or a proof that every implementation detail is bug-free.
No stale actor or misaligned label join was found by the stated identity checks.
The q formula can amplify uncertain probabilities (five0.5 values yield0.9375),
but mean q alone does not prove that it is the sole cause.

No heldout threshold tuning, scorer relaxation, loss change or head reset was
performed. The separately human-authorized two-stage actor selection uses probe
ranking to shortlist1024, then authentic strictEM to prioritize actual Mixed512;
it does not use this classifier threshold as its final actor acceptance rule.

Actual scalar evidence remains in the task output
`outputs/hpc3-prefix5-switch-20261017/probe-fit50-60-diagnostic-distribution-audit.json`.
Raw rollouts, tensors and checkpoints remain remote. Audit operator:
`docs/hpc3-probe-diagnostic-gap-audit-20261018.py`.

Exact membership evidence: `outputs/hpc3-prefix5-switch-20261017/probe-warmup-current-question-overlap-exact.json`; operator `docs/hpc3-warmup-question-overlap-20261018.py`.

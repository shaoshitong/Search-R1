# R4 P170: three epochs, probe CPU qualification and actual171 use accepted

Job733898 continues on ACD1-6 using four shared H100 GPUs. The immutable
training runtime is2000d46a530c64d78be7802bf11175a68b65effe. This report
records an intermediate result; the200-outer and final-evaluation goal remains
active. No training source, allocation, scorer, loss or optimizer was changed.

## Actual training and fit

Actor170 completed20 actual Adam writes per rank with zero skips, reaching3400
and scheduler170. The student state was saved before the first probe update.
The retained Prefix80 probe continued2050->2100->2150->2200 through three
epochs,150 actual writes per rank, without resetting the block, head or Adam.

The2560-question G5 training pool and independent512-question G5 diagnostic
were generated from frozen actor0169. These are current actor features and
real strict-EM labels, not substituted initial warmup features. The P170 pool
contained276 strict Mixed groups; all were selected with236 nonmixed fills
for the512-question actor batch,53.90625% Mixed. This selection preceded the
fit, used the2050-step probe and had a larger full pool than ordinary1024
rounds. It does not measure a benefit of the newly fitted2200-step head.

Joint170 SHA:
`6b4a3a939a4ce923d4a044449fdbe6d460a996e7e47feb09fa7a583befc8efb5`.
Joint-to-joint time was1159.848s. The fit loop took49.564s; its full phase,
including startup and evaluations, took115.357s.

## Same-data fitting curve

Accuracy here classifies strict Mixed with q>=0.5; it is not formal QA EM.
Every row evaluates the same2560 training and512 diagnostic groups.

| Epoch | Adam/rank | Train loss | Diagnostic loss | Train accuracy | Diagnostic accuracy |
|---|---:|---:|---:|---:|---:|
|Before|2050|1.364720|1.976750|67.3828%|49.4141%|
|1|2100|1.181282|1.997214|73.6719%|50.9766%|
|2|2150|0.885313|1.968198|83.7109%|47.4609%|
|3|2200|0.783101|1.962064|85.5078%|50.1953%|

Train balanced accuracy rose56.7156%->87.8966%; diagnostic balanced accuracy
changed59.2695%->54.6985%. Training fitting improved, but diagnostic results
do not establish improved generalization. Eligible training questions were
historically exposed during EMA5 warmup; diagnostic questions were excluded.
No diagnostic threshold tuning or strict-scorer relaxation was performed.

Metrics SHA:
`9df52544f7cc6da469198af64366efa503dba0d8515cb5c1fae3955bd2dfb6f5`.
Local scalar artifact:`two-stage-periodic-000170-fit-curve.json`.

## Independent probe payload qualification

The existing66f3dd86bc37 CPU operator passed full SHA and actual CPU loading
of the fitted probe, buffers,14 Adam states and all four rank RNG states.
It checked three epoch boundaries and the source joint, and pinned the
immutable probe with a remote hardlink. CUDA was not initialized; peak RSS
was1,553,297,408 bytes. This qualifies the probe payload, not full actor CPU
loading, GPU forward or stopped-writer recovery. All payloads remained remote.

Head SHA:
`2b5ac21ef624f266f6b116209d342c2edc88f1e0008e4b86c3354c1863cdf4b3`.
Remote proof:`preparation/live-probe-000170-v1-full-sha-cpu.json`.
Proof SHA:
`d0628087f3b01fccbfea8ab9a9f58490405063640c91a96c1299fc59a7f5dec4`.
Its exact3242 bytes were mirrored locally and verified.

## Actual171 new-head use and actor continuity

The separate acceptance checked all150 probe updates per rank, all three
epoch boundaries, actor170's20 real writes per rank and student-save-before-fit
ordering. It verified the source joint and CPU stat seals, and independently
read the stored strict-G5 pool and selected membership. Intermediate epoch
payload full SHA and full actor CPU loading were not repeated.

Actual171 scored using the exact2200-step head and frozen saved actor0170,
with Prefix80 and head capacity353 tokens. Its score-manifest SHA is
`6c6f551f7f1db7011b1b6eeda4cd1357b0fcf5b06e42fb0bc464e38f52124f46`.
Its ordinary budget is5120 prefix questions, then1024 complete original G5s,
then512 outcome-selected questions. It does not use the superseded dynamic
candidate formula. All four actor ranks restored Adam3400/scheduler170 and
FP32 master parameters, then each performed three actual writes to3403 at
acceptance. These are live complete-line ledger snapshots, not a claim that
actor171 or its producer-final ledger had already completed.

Acceptance status:
`P10_THREE_EPOCH_CPU_QUALIFIED_AND_NEXT_REAL_NEW_HEAD_ACTOR_USE_ACCEPTED`.
Remote proof:`preparation/two-stage-periodic-000170-actual-next-use.json`.
Proof SHA:
`619f6cb8512842e657856883663df325900274735dc6a90743cc17d85697824e`.

The remaining180/190 probe updates,200 milestone checkpoint/QA, final51713
and resource closure remain required. The full goal is not complete.

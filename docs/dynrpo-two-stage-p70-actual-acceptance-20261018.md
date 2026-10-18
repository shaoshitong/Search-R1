# R4 P70: three-epoch fit and actual next-outer use

Actual job733898, ACD1-6, four shared physical H100s; immutable runtime
2000d46a530c64d78be7802bf11175a68b65effe. This is intermediate acceptance
within the active200-outer objective, not completion of training or final QA.

P70 used2560 distinct training questions, each with five complete authentic
rollouts, and the fixed512-question G5 probe diagnostic. Both sets were sampled
from the frozen actor0069; the P70 student was subsequently saved before probe
fitting. Its four rank Adam counters are1400, scheduler70, with20 actual writes
and zero skips on each rank. Probe state continuously advanced550->600->650->700
through three epochs; no model or Adam reinitialization was performed.

## Full-data fit curve

These are Mixed-classification metrics (`q>=0.5`), not formal question-answer EM.
Each row evaluates the same full2560G5 training and512G5 diagnostic sets.

| Epoch | Adam/rank | Train total loss | Diagnostic total loss | Train accuracy | Diagnostic accuracy | Train balanced accuracy | Diagnostic balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
|Before|550|1.388688|2.000918|64.5703%|44.5313%|55.9226%|50.1339%|
|1|600|1.066640|1.953932|73.4375%|44.3359%|68.0686%|54.5869%|
|2|650|0.852490|1.940668|81.5234%|44.9219%|77.0601%|54.9398%|
|3|700|0.735268|1.921163|86.6797%|44.1406%|86.1032%|55.3834%|

Training fit improves strongly. Diagnostic total loss improves modestly and
balanced accuracy increases, while raw diagnostic accuracy does not improve.
This is not evidence that the large training/diagnostic generalization gap is
resolved. The separately established historical warmup exposure difference
remains relevant; neither these metrics nor new trajectories make the training
questions unseen by the retained probe. No diagnostic threshold tuning or
strict-scorer relaxation was performed.

The complete2560-question pool had K=0..5 counts1200/114/90/83/120/953,
where K is the number of strict-correct original responses. There were407
Mixed questions. All407 were retained in the512-question actor selection,
with105 nonMixed fills:79.4921875% actual Mixed. P70 completes a larger pool
than an ordinary1024-question continuation; this composition should not be
presented as a controlled before/after effect of probe fitting.

## Actual file and next-use evidence

The independent CPU operator verified full probe payload SHA, finite FP32
model/Adam and all four saved RNG streams at700 updates. No CUDA forward was
performed for this readback. The operator pinned the committed probe inode in
the same remote run, so later deletion of a rotating snapshot alias does not
invalidate the checked payload's existence. No tensor was copied to the Mac.

- Probe payload SHA:
  `85d6b7c3c73fb05dffe564f1ffd63dd2d5330f905cf36852cbba867fa3c5d698`.
- CPU proof:`preparation/live-probe-000070-v1-full-sha-cpu.json`, SHA
  `c3c85b8b0b796a110ffaafee940efd55ad35568a3686ad656f24b1d53c3c1ca9`.
- Next-use proof:`preparation/two-stage-periodic-000070-actual-next-use.json`,
  SHA`5a25c255b055eff9a35ff58f9398471ca668bc31754ddf0b38c5068d403c1a8c`.

The next-use verifier accepted the actual P71 score manifest bound to that
new probe SHA and Prefix20, plus the ordinary fixed5120 candidate budget. It
also checked each rank's actual restore from student70/Adam1400/scheduler70
with FP32 masters, followed by real writes1401..1411. The student save timestamp
precedes the first P70 probe update. Per-rank probe update journals contain all
150 writes, and the three epoch checkpoint receipts record600/650/700.

At that observation P71 was still updating. Its update-ledger digests are
explicitly complete-line snapshots of growing journals, not producer-final
file hashes or a claim that P71 had finished. The goal remains active through
all200 outers, remaining probe/prefix/grounding transitions, formal512 QA at
100/150/200, complete51713 final QA, checkpoints and own-resource cleanup.

Small scalar audit mirrors are in
`outputs/hpc3-prefix5-switch-20261017/two-stage-periodic-000070-fit-curve.json`,
`two-stage-live-probe-000070-v1-full-sha-cpu.json`, and
`two-stage-periodic-000070-actual-next-use.json`. Raw rollouts and state stay
remote. Formal final51713 acceptance is separately staged and has correctly
returned PENDING; it has not been accepted early.

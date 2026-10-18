# R4 P80: actual fit, file readback and next-outer use

Job733898 remains the same four-shared-H100 execution on ACD1-6, immutable
runtime2000d46a530c64d78be7802bf11175a68b65effe. This report accepts the
intermediate P80 update, not completion of the active200-outer objective.

P80 used2560 distinct training questions with five authentic complete rollouts
each, and the fixed512-question G5 probe diagnostic. Features and labels on
both sides came from frozen actor0079. Student80 was saved first, with four
rank Adam counters1600, scheduler80,20 actual writes and zero skips/rank.
The retained probe continued700->750->800->850 through three epochs; weights,
Adam and RNG were inherited, with no student/block reinitialization.

## Same-data fit curve

These accuracies classify true Mixed (`0<sum(strict_EM)<5`) at q>=0.5; they
are not formal QA EM. All rows evaluate the same full2560G5 and512G5 sets.

| Epoch | Adam/rank | Train loss | Diagnostic loss | Train accuracy | Diagnostic accuracy | Train balanced accuracy | Diagnostic balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
|Before|700|1.354990|1.949399|62.8906%|44.7266%|56.6586%|56.7831%|
|1|750|1.132336|1.958144|72.5000%|46.0938%|67.2951%|54.9157%|
|2|800|0.853406|1.948724|82.4609%|46.0938%|80.7650%|58.1221%|
|3|850|0.724821|1.950923|87.5391%|46.2891%|88.1205%|57.7026%|

Training fit improves strongly, while diagnostic loss does not improve.
The small raw/balanced diagnostic-accuracy changes do not establish that the
generalization gap is resolved. The previously verified warmup exposure
difference remains relevant: fresh trajectories do not make training questions
unseen by the inherited probe. No diagnostic threshold, loss, strict scorer or
running training source was changed in response to these metrics.

There were427 true Mixed training questions and77 diagnostic Mixed questions.
The full training-pool K=0..5 counts were1111/125/68/89/145/1022. All427
Mixed questions entered the512-question actor selection, with85 nonMixed fills:
83.3984375% selected actual Mixed. This P10 pool completes2560 questions,
whereas ordinary rounds complete the probe Top1024. Its selection happened
before fitting the P80 probe, so its higher Mixed fraction must not be attributed
to that new fit or presented as formal QA accuracy.

## Independent readback and actual next use

The actual probe payload underwent full SHA and CPU model/Adam/four-rank RNG
validation at850. Finite FP32 model/Adam states were checked; CUDA remained
uninitialized. The readback preserved a remote hardlink to the immutable
probe inode, preventing rotating-alias cleanup from invalidating the evidence.
No model tensors or raw rollouts were copied to the Mac.

- Probe payload SHA:
  `1d6f809ca9bfefc62e0aa8ec613e538483f58277a2caa3d9abbbe4a6b225e74b`.
- CPU proof:`preparation/live-probe-000080-v1-full-sha-cpu.json`, SHA
  `836bdf43386b9d593f49464e3cd00360847d93a186d3cd1a7812788dbe09090c`.
- Next-use proof:`preparation/two-stage-periodic-000080-actual-next-use.json`,
  SHA`8fa1c6fbd127468e572be8e5c83a751bd80846d49e2d6d195b2fd6650a19ba44`.

The next-use verifier checked the actual P81 score manifest against the new
probe SHA and Prefix20, plus the fixed5120 candidate budget/cursor. Each rank
actually restored student80/Adam1600/scheduler80 with FP32 masters and wrote
1601..1614. Student80's save preceded the first probe update. The150 probe
writes/rank and all three epoch receipts750/800/850 were checked independently.
P81 was still active at proof time; its ledger hashes are explicitly observed
complete-line snapshots, not final producer hashes or a completed-P81 claim.

The bounded target80 reader79974 expired normally before final joint submission.
Fresh reads of the same live Slurm job then confirmed actual joint80 and P81
progress. Training was not restarted. Read-only follower74471 now targets90;
no canceled automation was restored.

Small audit mirrors are in `outputs/hpc3-prefix5-switch-20261017/`:
`two-stage-periodic-000080-fit-curve.json`,
`two-stage-live-probe-000080-v1-full-sha-cpu.json`, and
`two-stage-periodic-000080-actual-next-use.json`. Runtime state, tensors and full
responses remain remote. Remaining work includes student through200, P10 and
Prefix40/80/grounding transitions, formal512 QA at100/150/200, final51713 QA,
checkpoint acceptance and own-resource cleanup.

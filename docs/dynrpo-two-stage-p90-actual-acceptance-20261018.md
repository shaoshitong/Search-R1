# R4 P90: actual fit, full probe readback and next-outer use

Job733898 continues on ACD1-6 with four shared H100 GPUs. The running source
remains immutable2000d46a530c64d78be7802bf11175a68b65effe. This accepts the
intermediate P90 update; the200-outer goal and final51713 QA remain incomplete.

## Data, order and continuous fitting

P90 sampled2560 distinct training questions x5 authentic trajectories, plus
the fixed512-question G5 probe diagnostic, from the same frozen actor0089.
Student90 was saved before the first probe update: four rank Adam1800,
scheduler90,20 actual new updates/rank and zero skips. The retained probe
continued850->900->950->1000 in three epochs,150 actual Adam writes/rank.
No student parameter copy, transformer-block reset or optimizer reset occurred.

## Same-data fit curve

These are Mixed classifier accuracies at q>=0.5, not formal QA EM. Every
epoch row evaluates the same full training and diagnostic sets.

| Epoch | Adam/rank | Train loss | Diagnostic loss | Train accuracy | Diagnostic accuracy | Train balanced accuracy | Diagnostic balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
|Before|850|1.433087|1.965020|60.7812%|47.4609%|52.0495%|55.0417%|
|1|900|1.110137|1.965843|69.8828%|44.5312%|64.7761%|54.0721%|
|2|950|0.869317|1.985583|81.4453%|43.5547%|77.8511%|51.8248%|
|3|1000|0.725197|1.971922|86.4062%|45.5078%|85.6061%|53.0210%|

Training loss improves substantially; diagnostic loss does not improve.
Diagnostic raw accuracy falls from47.4609% to45.5078%, and balanced accuracy
from55.0417% to53.0210%. Do not present this fit as a demonstrated improvement
in generalization. Earlier exact warmup membership evidence shows different
historical question exposure for train and diagnostic cohorts; fresh actor
trajectories do not by themselves make the questions unseen. No diagnostic
threshold, probe loss, strict scorer or running training source was changed.

The training pool had448 true Mixed questions; the diagnostic had94.
Training K=0..5 counts: 1065/117/104/96/131/1047.
All448 Mixed questions entered the selected512, plus64 nonMixed fills, giving
87.5% selected Mixed. Selection happened before fitting this P90 head and
used2560 full questions rather than the ordinary probe Top1024. It does not
establish a benefit of the new fit and is not formal QA EM.

## Independent CPU qualification and actual P91 use

The full probe payload SHA, finite FP32 model/Adam states, and all four rank
RNG states passed CPU readback at1000. CUDA was not initialized. A hardlink
pin retains the actual immutable remote probe inode across rotating-alias
cleanup; no tensors, raw rollouts or checkpoints were copied to the Mac.

- Probe payload SHA: `1a5cf4372d08b183ddd4afe018d9c24e753c096145dadae73a5eec9e9d157c68`.
- CPU proof: `preparation/live-probe-000090-v1-full-sha-cpu.json`, SHA
  `149509bdabe74f70ff6c637157b35551c7b0834d0ed718a910df586f7872b308`.
- Next-use proof: `preparation/two-stage-periodic-000090-actual-next-use.json`,
  SHA `78a7e685d317f7c1e1564669b1fea5b5c9f7477a3ba25e307e1576dc46f00e1b`.

The independent verifier matched the actual P91 score manifest to this exact
new head SHA and Prefix20, and checked the fixed5120 candidate budget/cursor.
All four actor ranks actually restored student90/Adam1800/scheduler90 with
FP32 masters, then wrote1801..1814. It also checked the student-save timestamp
before the first probe update, all150 actual probe writes/rank and all three
epoch receipts900/950/1000. P91 was active at proof time: its ledger hashes
are explicitly observed complete-line snapshots, not producer-final hashes
or a claim that P91 had already completed. This is probe CPU qualification
and actual actor restore/write evidence, not actor tensor CPU readback.

## Continuing execution

Foreground reader74471 ended normally upon actual joint90. Its replacement
68427 follows the same job with bounded read-only observations toward110.
No training restart, duplicate submission, hot source change or canceled
automation restoration occurred. Full200 training remains active.

Next requirements include the Prefix40/grounding100 transition, full100
checkpoint readback and fixed512 QA, the Prefix80/grounding150 transition,
remaining P10 fits through190, final200 checkpoint and fixed512 QA, full
51713 QA with seven-set macro EM reported separately from question-weighted
EM, and own-resource cleanup. Completion remains unproven until those pass.

Small evidence mirrors reside in `outputs/hpc3-prefix5-switch-20261017/`:
`two-stage-periodic-000090-fit-curve.json`,
`two-stage-live-probe-000090-v1-full-sha-cpu.json`, and
`two-stage-periodic-000090-actual-next-use.json`. All model assets remain remote.

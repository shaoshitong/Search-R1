# R4 P100: checkpoint, fixed512 QA and actual complete101 transition accepted

Job733898 continues on ACD1-6 with four shared H100 GPUs, immutable runtime
2000d46a530c64d78be7802bf11175a68b65effe. This is an intermediate milestone
report, not completion of the200-outer/final51713 objective. Round100 reward
grounding was fitted from saved actor100 and adopted by all four replicas.
The independent transition audit then passed against actual complete101;
outer102 had started Prefix generation at the latest observation.

## Student first, then new Prefix40 data and fitting

Actor100 completed20 actual Adam updates/rank, zero skips; counters2000 and
scheduler100 were durably saved before the prefix refresh. Actor100 selection
used Prefix20 and round50 reward-grounding. The extra probe-only collection
then used frozen actor0100:2560 distinct questions x5, plus fixed512G5 probe
diagnostic, all at Prefix40. Its separate reservation did not advance the actor
candidate pool. Student was not updated during collection or probe fitting.

The actual runtime migration checked that block parameters and native buffers
were equal across the readout expansion, retained old output coordinates and
Adam moments, zero-extended new output coordinates/moments, and retained Adam
steps1000 and all four RNG states. This is the checked runtime receipt; this
audit did not independently reload and compare both pre-fit tensor snapshots.
Subsequent three epochs continued training the existing probe, without copying
parameters from student or resetting the native block/optimizer.

Migration receipt SHA: `f285c41617b0c30237c020145cbcb03356c50dc882818772dc5b603d28e573f3`.
Expanded pre-fit Prefix40 probe SHA: `d71a50976987b448b4e349c5a4045620675bd48382d49b714f1c80f8a0a2a33f`.
Final fitted Prefix40 probe SHA: `8d389ecd5e5b9ec864d17761f78c83d767385b7a8a68b0b83f073446143b6a3f`.

## Same-data probe fitting curve

These accuracies classify real Mixed at q>=0.5 and are not QA EM. All four
rows evaluate the same current-actor2560G5 and512G5 datasets.

| Epoch | Adam/rank | Train loss | Diagnostic loss | Train accuracy | Diagnostic accuracy | Train balanced accuracy | Diagnostic balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
|Before|1000|1.376865|1.969125|64.4531%|44.7266%|55.6853%|55.5061%|
|1|1050|1.172638|1.995340|72.8125%|46.6797%|64.8973%|54.2765%|
|2|1100|0.900902|1.946817|81.4844%|42.9688%|79.8258%|55.0765%|
|3|1150|0.766654|1.943880|86.2109%|43.7500%|87.3991%|56.1209%|

Probe Adam continued1000->1050->1100->1150,150 actual writes/rank. Training
loss improved substantially. Diagnostic loss declined slightly1.969125->1.943880,
while raw diagnostic accuracy fell44.7266%->43.75% and balanced accuracy rose
55.5061%->56.1209%. This does not establish that generalization is fixed or
causally attribute the change to Prefix40: both actor snapshot and sampled
trajectories differ from the prior P90 data. No scorer/threshold/loss was tuned.

## Independent full checkpoint CPU qualification

The full33-file SHA/CPU audit passed against this R4 actual joint100. It checked
HF actor tensors, all four rank model/Adam/scheduler/RNG states, driver RNG,
candidate-pool position, paired Prefix40 probe1150 and the round50 reward
state actually used by actor100. Immutable file stat/filesystem seals were
rechecked before accepting the proof. This is live immutable-payload readback,
not a claim of a stopped-writer recovery rehearsal or GPU forward validation.

The audit ran in a detached CPU-only Slurm step of existing job733898, with
two CPUs, requested12G memory, gresgpu0 and empty CUDA_VISIBLE_DEVICES. It
allocated no extra GPU or training job. Actual CPU peak RSS was6.625GB.
The own PID/start identity, lock, intent, progress and sealed result prevented
duplicate launch after control timeouts. No tensor payload was moved to Mac.

CPU proof: `preparation/live-milestone-000100-v1-full-sha-cpu.json`, SHA
`2cdf46c5e9e79b21c8908b38206fdf7f0d59d4a0155d2ed8f6c5791e32766414`.

## Formal fixed512 QA at actor100

This is the bound dev512 set, NQ256 plus HotpotQA256, with strict saved-score
aggregation. It is separate from the probe Mixed diagnostic and from final
seven-dataset51713 QA. All eight rank receipt/compressed-output files underwent
fresh SHA checks, canonical qid/source coverage checks and stored-score
reaggregation. The parser/scorer was not relaxed; no extra generation or token
rescoring was performed by the audit. BF16 compute and four zero worker exit
codes were verified; no pending-QA entry remained.

| Set | Questions | Strict EM | F1 |
|---|---:|---:|---:|
|NQ|256|40.2344%|51.8719%|
|HotpotQA|256|52.3438%|56.9299%|
|Combined dev512|512|46.2891% (237/512)|54.4009%|

Actual parallel rank inference times (128 questions/rank): 10.293s, 10.343s, 10.143s, 10.664s.
Maximum parallel inference time10.664s; the full phase including initialization took86.878s.
These QA scores describe actor100, trained through100 using Prefix20 selection;
the new Prefix40 selection can affect actor updates starting101, not these
already-saved actor100 weights.

QA source-complete SHA: `65c3467b8facebf9e1be86338a42e05f9e4874f9b4b5028a9675c4dad84a0b5f`.
QA audit mirror: `preparation/fixed512-000100-fresh-sha-acceptance.json`, SHA
`30f21e02e276542c9ffd032deece923490a8c2050688030952c6cd45f927981a`.

## Grounding100 and actual complete101 transition

The new heads were fitted from actual actor0100 using the bound300-question
G5 calibration, with1497/1500 valid rollouts, both required CV folds and BF16
interaction. Commit CV AUC was0.809940; boundary CV AUC was0.759499. These
are grounding calibration metrics, not the selection probe or formal QA EM.
All four actual reward services adopted version
`cc0de51f2d68495c9664aa888074839c:100:0250f494eabf41f596a6d49f09586a45`.

The independent transition reader checked the fitted binary heads' full SHA
and finite CPU state, actor100 encoder binding, calibration identity/protocol,
four distinct rank reward endpoints, and the new version recorded in actor101's
actual reward receipt. Selection-probe weights were not used for reward.
It also checked actual101 Prefix40 generation from actor0100, scoring with the
exact fitted1150-step probe, and four rank restore/update ledgers. All four
actors restored2000 and wrote20 actual updates, reaching2020; scheduler101
completed and zero opportunities were skipped. The complete101 joint SHA is
`6063f792d156fabbdcf855b5b9ca06d04b79bfdafecf015bd9641c109646d212`.

Acceptance status: `ACTUAL_PREFIX_AND_GROUNDING_TRANSITION_USED_BY_COMPLETE_NEXT_OUTER_PASS`.
Remote proof: `preparation/transition-000100-actual-next-complete.json`, SHA
`fd96fa2d4e205a7bbd9237ac240fc0047f8784898a774a5f99cbc35a6b14901a`.
Grounding manifest SHA: `65afaea92d4a5955e2d9b51b366f1ac6b741550e0b340dc0ef3fb0fe35040dff`.
Four-service adoption SHA: `b713df76a6bebf6f6f9ca65dee8d644f5a767be52f34d6c106c93b0092766be7`.

The transition reader reused the sealed fullCPU100 proof after checking its
file stat/filesystem seals. It did not repeat the full calibration feature
payload hash scan or independently compare both pre-fit migration tensors;
the readout expansion assertions above remain runtime-checked evidence.
Formal fixed512 QA was accepted separately, as documented above.

The first ordinary Prefix40 round101 selected193/512 real Mixed questions
(37.6953%),154 all-wrong and165 all-correct, with319 nonMixed fills. Its
successful phase times summed678.078s; joint100-to-joint101 was1200.145s
and additionally includes post100 QA/grounding/orchestration. This single
round does not demonstrate improved Mixed selection from Prefix40. Round100's
73.4375% came from the larger2560-question P10 pool and is not a matched
ordinary-round comparison.

## Actual Prefix40 continuation and PPO projection

The separate read-only raw-record audit passed for all5120 trajectories in
round101's Top1024 pool and the2560 projected actor trajectories. Fresh SHA
was checked for17 raw-record files kept on remote storage. Each checked
prefix retained40 real policy tokens. Their IDs, original request/positions,
closed-turn tokens/masks/logprobs and partially completed action chunks matched
the real full continuations. All5123 recorded native hidden-flush events
discarded exactly one lookahead token. The final512 groups were independently
ranked from the stored strict labels and original q scores, with no rescoring.
Every original scientific field was unchanged in the projected PPO rows;
only the declared subset selection bookkeeping was rebound.

Remote proof `preparation/prefix-continuity-000101.json`, SHA
`b1004f785a9ee2737eec992713a58235ef428bb0748674471dc1ec5ef0ddf26e`.
Operator `docs/hpc3-two-stage-prefix-boundary-continuity-20261018.py`, SHA
`c4f20ed1150ae4ba72a910de4392e8fb57a12e053a4da8c37f3f94fdd2d95ab9`,
was committed/pushed as4239f20d7938d81360c6fab44043e8f2cc271042 and staged
as an immutable separate preparation tool, without changing runtime2000d46.
No hidden-feature tensor was read. All checked Top1024 prefixes reached40,
so this audit does not independently exercise short-prefix feature padding.
Its real151 entry returned PENDING; Prefix80 still requires actual151 evidence.

## Remaining work

Training through200, later probe updates, Prefix80/grounding150, fixed512
QA150/200, final51713 QA, all remaining checkpoint checks and own-resource
cleanup remain incomplete. Full goal remains active. No runtime source,
scorer, threshold or training configuration was changed for this audit.

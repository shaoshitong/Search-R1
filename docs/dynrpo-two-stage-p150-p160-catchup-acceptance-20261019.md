# R4 P150/P160 acceptance and continuous training through165

Fresh Slurm and runtime receipts identify the same job733898, mou176,
ACD1-6 and immutable runtime2000d46a530c64d78be7802bf11175a68b65effe,
using four shared physical H100 GPUs. The earlier observation handle was
unavailable; this did not cause a training restart or a new submission.
Fresh completed receipts132..165 are contiguous: every rank has20 new Adam
writes/outer and zero skipped opportunities. Complete165 has3300 writes/rank,
scheduler165 and the frozen2050-step Prefix80 probe. Actual166 generation
advanced from200 to5900+ trajectories/rank with no observed new errors.

This report records intermediate acceptance. Training through200, final51713
QA, final checkpoint/retention qualification and resource closure remain due.
No training source, scorer, loss, thresholds, allocation or optimizer was changed.

## Full milestone150 checkpoint

The retained milestone was independently read inside the existing733898
allocation using a separate CPU-only srun step,2 CPUs,12G requested memory
and no additional GPU allocation. The launcher was observed live, verification
advanced through13.6GB and25.9GB, then the worker completed with a sealed PASS.
No process was stopped or replaced to perform this audit.

Full SHA and actual CPU loading qualified33 files: HF model, all four rank
model/Adam states, scheduler150, driver/data-pool state, RNG, paired1900-step
probe and the grounding100 state used by actor150. Actor counters were3000
per rank,20 new writes, zero skips. CPU model tensors were finite; no model
forward or stopped-writer recovery rehearsal was performed.

Joint150 SHA:
`4865c19ffd1f03993e17fdff83fba48bf6d1de4747bf0249f192084370d17520`.
CPU proof: `preparation/live-milestone-000150-v1-full-sha-cpu.json`, SHA
`a8898cd4a5fe55d8c335b09c99a460a2484d4092803fd08a2f78bbbc0ebe544e`.
Its exact19492 bytes were mirrored locally and verified.

## Formal fixed512 QA at150

The independent QA reader freshly hashed all eight rank files, checked the
canonical512 identities and stored strict-score aggregation, and confirmed
BF16 computation and all worker exits0. This did not generate new answers
or relax/recompute the strict answer parser.

| Checkpoint | Strict EM | F1 |
|---|---:|---:|
|100|237/512 =46.2891%|54.4009%|
|150|249/512 =48.6328%|56.8096%|

The150 set contains256 NQ questions (EM41.7969%, F153.0370%) and256 HotpotQA
questions (EM55.4688%, F160.5822%). Rank inference times were9.532/9.857/9.512/
9.453s; the complete validation phase including initialization took80.240s.
The question-source SHA remains
`3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb`.

This is fixed512 QA, not the final seven-benchmark evaluation. Actor150 was
trained using Prefix40 and grounding100; its QA improvement cannot be
attributed to the subsequent Prefix80 or grounding150 refresh.

QA complete SHA:
`b30e2140e43ff0fe071d37f80ef978ec42334e6513c55882bca8883a8ffd13aa`.
Persisted independent QA proof:
`preparation/fixed512-000150-fresh-sha-acceptance.json`, SHA
`c854a46f3b990d23de02842347bbfda2a1965be5af8b48f7bdabd768bd5a7b18`.

## Actual Prefix80 and grounding150 adoption

After actor150 was saved, the Prefix80 fitting data came from frozen actor0150.
The probe continued1750->1800->1850->1900 for three epochs. Runtime migration
checks record retained block/buffers, zero-extension of the readout and Adam
slots, and retained clock/RNG. Both pre-fit tensor sets were not independently
recompared by the later transition reader.

The transition reader accepted actual complete151: all four ranks restored
actor150 and completed20 new writes, reaching3020 with zero skips; scoring
used the fitted1900-step Prefix80 head. New grounding heads were freshly
hashed, CPU-loaded as finite binary Pipeline classifiers, adopted by all four
services and used in actual151 rewards. Grounding source was actor0150.
Grounding version:
`cc0de51f2d68495c9664aa888074839c:150:888b73ca78f241f085bd4d03f04112fb`.
Boundary AUC was0.8544474 and commit AUC0.7839582; these are calibration
metrics, not formal QA results. Calibration feature payload hashes were not
independently rescanned by this transition check.

Transition proof: `preparation/transition-000150-actual-next-complete.json`, SHA
`9f00bbe6bd5cc09d1145fd7509d8d188106a9c38253c9a3399ef989b0ddba606`.

A separate raw-data audit freshly hashed17 actual151 files and checked5120
Top1024 trajectories' original tokens, positions, masks, sampling logprobs and
continuation chunks, then exact scientific-payload equality for2560 selected
PPO trajectories. Retained real prefixes range46..80 tokens, including2234
length80 trajectories;9888 lookahead flushes were discarded. Short prefixes
were not expanded into invented PPO tokens. Hidden-feature padding tensors
were not independently loaded by this raw-data audit.

Raw continuity proof: `preparation/prefix-continuity-000151.json`, SHA
`615248807ad90e57ffea75029c9af30dded61eacdb0452935d23b5838b721588`.

## P140 and P160 probe scope

P140's three epochs1600->1650->1700->1750 are present in its actual epoch
receipts, retained epoch files and metrics. Train loss1.359305->0.720937;
diagnostic loss1.919003->1.894930, with diagnostic accuracy46.2891% at both
ends. Its original non-milestone joint-probe snapshot was legitimately rotated
out. The retained epoch3 file has different bytes/SHA from that joint snapshot;
it was not substituted or claimed to pass original joint-probe CPU readback.
This differs from mandatory retained milestones5/50/100/150/200.

P160 used current actor0159 Prefix80 features,2560G5 training data and the
separate512G5 diagnostic, then continued1900->1950->2000->2050. Independent
full SHA/CPU loading qualified the actual final probe, buffers, Adam and four
rank RNG states, with CUDA uninitialized. The exact source head was pinned.
Actual161 scored with that head and all four ranks restored3200, completed
20 real updates and reached3220. Student-before-fit ordering and all150
probe writes/rank were checked. Full actor payload CPU loading was not repeated.

P160 train loss1.332339->0.734924 and accuracy66.6797%->88.5156%; diagnostic
loss1.946353->1.964433 and accuracy46.6797%->47.0703%. These fitting numbers
do not establish improved generalization or downstream selection.

P160 CPU proof: `preparation/live-probe-000160-v1-full-sha-cpu.json`, SHA
`c382a54f2497ae9626546b584e1269fd7689bb7c6424d0c3b50c8db4dd53ecf9`.
Actual161 use proof: `preparation/two-stage-periodic-000160-actual-next-use.json`, SHA
`aeb6ed1ce3a3cbad47f2c8638f426080e7e61a9ff9df65b409657ec51b0bfa56`.

All mirrored artifacts are small audit JSON only. Model, tensor and raw-rollout
payloads stayed remote. No canceled automation was restored. Later probe
updates170/180/190, full200 checkpoint and fixed512 QA, final51713 QA and own
resource cleanup remain incomplete; the full goal stays active.

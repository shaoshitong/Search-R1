# R4 P190: final probe fit, CPU qualification and actual191 use accepted

Job733898 continues on ACD1-6 with four shared H100 GPUs. The immutable
training runtime remains2000d46a530c64d78be7802bf11175a68b65effe. No training
source, allocation, loss, scorer or optimizer was changed.

## Actual completed190

Student190 reached3800 Adam writes per rank and scheduler190, with20 new
writes and zero skips per rank. The retained Prefix80 probe continued
2350->2400->2450->2500 through three epochs. This is the last scheduled
probe fit;200 saves and evaluates without another probe update.

The training pool contained2560 distinct questions,12800 original G5
trajectories and265 strict Mixed groups. Its512-question actor batch retained
all265 Mixed groups and247 nonmixed fills,51.7578125% Mixed. Selected K=0..5
counts were120/73/50/48/94/127. Both this pool and the512-question diagnostic
were generated using frozen actor0189. Selection preceded fitting and used
the2350-step head. Its larger pool differs from ordinary1024-completion
rounds, so this result cannot be attributed to the new2500-step probe.

Joint190 SHA:
`0fe0c975e61dad6fbbdd23c4202cbbd0997f7ebc5ae88ada99b6dd147981b6b6`.
Joint-to-joint time was887.930s; successful phases summed826.084s. The fit
loop took43.362s, while its full phase took94.188s. These scopes differ
because the cycle also includes storage and orchestration.

## Same-data fitting curve

The accuracies below classify strict Mixed using q>=0.5; they are not formal
QA EM. Each row evaluates the same2560 training and512 diagnostic groups.

| Epoch | Adam/rank | Train loss | Diagnostic loss | Train accuracy | Diagnostic accuracy |
|---|---:|---:|---:|---:|---:|
|Before|2350|1.406621|1.956455|64.1016%|48.0469%|
|1|2400|1.116746|1.962913|71.4453%|50.1953%|
|2|2450|0.911805|1.979303|82.8125%|46.8750%|
|3|2500|0.761802|1.959134|85.8594%|50.7813%|

Training balanced accuracy rose51.9402%->86.4389%; diagnostic balanced
accuracy changed54.2074%->55.0507%. The training fit improved, while diagnostic
loss slightly worsened. These measurements do not establish improved
generalization. Eligible training questions had historical EMA5 warmup
exposure; diagnostic questions were excluded. No threshold tuning or strict
scorer relaxation was used.

Metrics SHA:
`55b2d04f903d768934f92713f7a0a0b6344c7a65efb6d2d41b71f4aea635aa16`.
Local scalar artifact:two-stage-periodic-000190-fit-curve.json.

## Independent probe payload qualification

The existing66f3dd86bc37 operator passed full SHA and actual CPU loading of
the fitted probe, buffers, Adam and four rank RNG states. It checked three
epoch boundaries and the source joint, retaining an immutable remote hardlink.
CUDA remained uninitialized; peak RSS was1549602816 bytes. This qualifies
the probe payload, not full actor CPU loading, GPU forward or a stopped-writer
recovery rehearsal. All tensors remained remote.

Head SHA:
`cab9401fabf7bebd40a653917b7e45c0b0a514ec889c3c1bafee261077da9e97`.
Head bytes:933674012.
Remote proof:preparation/live-probe-000190-v1-full-sha-cpu.json.
Proof SHA:
`bdfad6ad691e5eabefa1ab275ddc679fc23a9715214b7111d94dfcb1b3e912c3`.
The exact3243 proof bytes were mirrored locally and SHA-verified.

## Actual191 new-head use and actor continuity

The separate acceptance checked all150 probe writes per rank, all three
epoch boundaries, actor190's20 real writes per rank, student-save-before-fit
ordering, qualified CPU stat seals, stored strict G5 labels and the selected
membership. Intermediate epoch payload full SHA and full actor CPU loading
were not repeated.

Actual191 scored with the exact2500-step head and saved actor0190, Prefix80
and head capacity353 tokens. Score-manifest SHA:
`1e5d0aea379345c0752af4378705f9a5d7c984213da679b7b40a83a0811bc1dd`.
Its ordinary budget is5120 prefix questions,1024 complete original G5s,
then512 outcome-selected questions; the old dynamic formula is superseded.
The selected batch contained131 strict Mixed questions and381 fills,
25.5859375% Mixed. All four actor ranks restored Adam3800/scheduler190 and
FP32 master parameters, then reached3811 at acceptance. These11 live writes
per rank and complete-line ledger snapshots do not claim a completed191
joint or producer-final ledger SHA.

Acceptance status:
`P10_THREE_EPOCH_CPU_QUALIFIED_AND_NEXT_REAL_NEW_HEAD_ACTOR_USE_ACCEPTED`.
Remote proof:preparation/two-stage-periodic-000190-actual-next-use.json.
Proof SHA:
`51d7e693e40941bbc5fc87483aac4cd613b890b1fe625b2b8282f6994ff46674`.

## Remaining final acceptance

The old bounded observer76043 ended normally at its240-sample limit;
sequential87251 then ended normally after the actual190 joint. Neither event
was a training failure or restart. The new read-only foreground helper
docs/hpc3-two-stage-follow-final-20261019.py is syntax-checked and its real
entry produced two observations showing191 prefix advancement. It performs
no submission or training control and alerts at the actual200 joint,
explicitly requiring the remaining independent audits.

At actual200, launch the existing full checkpoint CPU audit promptly while
the owned allocation is still live, then qualify fixed512 QA, final51713,
retained milestones and owned-resource closure. The full goal remains active.

# R4 P130: three epochs, probe CPU qualification and actual131 use accepted

Job733898 continues on ACD1-6 with four shared H100 GPUs and immutable runtime
2000d46a530c64d78be7802bf11175a68b65effe. This is an intermediate acceptance;
the full200-outer, milestone QA, final51713 and resource-closure goal is active.
Training source, scorer, loss, thresholds, allocation and optimizers were unchanged.

## Collection and continuous fitting

The P130 pool used2560 different questions x5 at Prefix40. Its independent
fixed512G5 diagnostic and training features/strict labels came from the same
frozen actor0129, rather than the subsequently updated actor130. Current
layer27 features were used; initial warmup tensors were not substituted.

Actor130 completed20 real Adam writes/rank, zero skips, reaching2600 with
scheduler130. Its student state was saved before the first probe update.
The retained probe continued1450->1500->1550->1600 through three epochs,
150 real writes/rank. There was no Prefix expansion, block copy or Adam reset.

The full2560 pool contained302 strict Mixed questions. All302 were retained
in the512-question actor batch, with210 fills (58.984375% Mixed). Selection
preceded this fit and used the1450-step probe. The larger P10 pool differs
from ordinary rounds'1024, so this is not evidence of a new-head benefit.

Actual joint130 SHA:
`890ceea71ce9d391de230ad627e321f48ccec5a7054a18d0955ac398f10a8b0f`.
Joint-to-joint time was769.675s. The fitting loop took43.034s; its complete
phase, including startup/evaluation, took85.817s.

## Same-data fitting curve

These accuracies classify strict Mixed using q>=0.5, not formal QA EM.
The same2560 training and512 diagnostic groups are evaluated in every row.

| Epoch | Adam/rank | Train loss | Diagnostic loss | Train accuracy | Diagnostic accuracy |
|---|---:|---:|---:|---:|---:|
|Before|1450|1.375023|1.936476|62.8516%|44.9219%|
|1|1500|1.135596|1.927867|72.3828%|45.1172%|
|2|1550|0.899571|1.925212|81.3281%|45.5078%|
|3|1600|0.782174|1.928496|86.7188%|47.4609%|

Train balanced accuracy rose53.8431%->86.8779%; diagnostic balanced accuracy
rose54.2728%->57.5469%. This diagnostic change is modest and is not a
controlled demonstration of improved downstream selection or QA. Historical
EMA5 warmup exposed eligible training questions and excluded the diagnostic
questions; fresh rollouts do not remove that exposure difference. No diagnostic
threshold tuning, loss change or strict-scorer relaxation was performed.

Metrics SHA: `29fff89b1990c3d00b8451aa739c1d4573de81050ddf4b28795105a7ea95116e`.
Scalar artifact: `two-stage-periodic-000130-fit-curve.json`.

## Independent qualification and actual131 use

The staged66f3dd86bc37 CPU reader passed full SHA and actual CPU deserialization
of the fitted native probe, buffers, Adam and four rank RNG states. It checked
all three epoch boundaries and source joint. A remote hardlink pins the
immutable payload. CUDA initialization was false; peak RSS was1,401,196,544B.
This qualifies the probe, not actor-payload CPU loading, GPU forward or a
stopped-writer recovery rehearsal. Model/tensor/raw-rollout data remained remote.

Fitted head SHA:
`ffa938feddb718cf08ca03cb2c3249fd32d7a0d22027197b9a1c728bb835cde3`.
Remote CPU proof: `preparation/live-probe-000130-v1-full-sha-cpu.json`, SHA
`4e3f362719c86eea83fd202fdef2bfdfe168c241190f8408ece3c9277183735f`.
The exact3242-byte proof was mirrored locally and its SHA verified.

Independent acceptance also checked actor130's20-opportunity ledgers, all150
real probe writes/rank, epoch counts, stored strict-G5 membership and Mixed
selection, student-save-before-fit ordering, and preserved CPU stat seals.
Actual131 scored using the exact1600-step head, sampled from saved actor0130,
and restored all four ranks at Adam2600/scheduler130 with FP32 master parameters.
At acceptance each rank had performed9 real writes and reached2609. These are
live complete-line ledger snapshots, not producer-final ledger SHA claims.
Intermediate epoch payload full SHA and full actor CPU loading were not repeated.

Acceptance: `P10_THREE_EPOCH_CPU_QUALIFIED_AND_NEXT_REAL_NEW_HEAD_ACTOR_USE_ACCEPTED`.
Remote proof: `preparation/two-stage-periodic-000130-actual-next-use.json`, SHA
`58df526fd5d74c152233985e2dc07141a5999e5d36e4ca41426270c07e16d234`.
Its exact4677 bytes were mirrored locally and verified. Actual131 score SHA:
`6e75f91d9527991607247089d969805402e1de58e7c393398969c6a54d831a11`.

The next formal fixed512 QA milestone remains150. Later P10 updates,
Prefix80/grounding150, remaining checkpoint acceptance, final51713 QA and
resource cleanup remain incomplete. The full goal is not complete.

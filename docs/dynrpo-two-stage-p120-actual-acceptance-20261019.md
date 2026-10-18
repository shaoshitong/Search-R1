# R4 P120: three epochs, probe CPU qualification and actual121 use accepted

Job733898 continues on ACD1-6 with four shared H100 GPUs and immutable runtime
2000d46a530c64d78be7802bf11175a68b65effe. This is an intermediate acceptance;
the full200-outer, milestone QA, final51713 and resource-closure goal is active.
No training source, scorer, loss, threshold, allocation or optimizer was changed.

## Collection, student save and continuous probe fitting

P120 used2560 different questions x5 at Prefix40. Its independent fixed512G5
probe diagnostic and training features/strict labels were generated from the
same frozen actor0119. They do not describe the subsequently updated actor120.
Initial warmup feature tensors were not substituted for current actor features.

Actor120 completed20 actual Adam writes/rank, zero skips, reaching2400 with
scheduler120. Its student state was saved before the first probe update.
The retained probe continued1300->1350->1400->1450 through three epochs,
150 actual writes/rank. No Prefix expansion, block copy or Adam reset occurred.

The full2560 pool contained318 strict Mixed questions. All318 were retained
in the512-question actor batch, with194 fills (62.109375% Mixed). This selection
preceded the new fit and used the1300-step probe. Its larger full-pool budget
differs from ordinary rounds'1024; it cannot establish a benefit of the new head.

Actual joint120 SHA:
`f141a1d19b5f3afc40f21889ebf5c5ed0bef1309271c148165638489cc0b2151`.
Joint-to-joint time was798.864s; the fitting loop took42.285s and its complete
phase, including startup/evaluation, took82.853s.

## Same-data fitting curve

These accuracies classify strict Mixed using q>=0.5, not formal QA EM.
The same2560 training and512 diagnostic groups are evaluated in every row.

| Epoch | Adam/rank | Train loss | Diagnostic loss | Train accuracy | Diagnostic accuracy |
|---|---:|---:|---:|---:|---:|
|Before|1300|1.388913|1.955522|63.0469%|43.9453%|
|1|1350|1.079150|1.966783|72.8516%|43.5547%|
|2|1400|0.852036|1.964556|83.3984%|43.7500%|
|3|1450|0.753050|1.949270|87.4219%|44.5313%|

Train balanced accuracy rose56.0994%->89.3107%; diagnostic balanced accuracy
was54.9095%->54.6477%. Fitting improved strongly on training data, while the
diagnostic result changed little. This does not establish improved generalization.
Historical EMA5 warmup exposed the eligible training-question pool and excluded
the diagnostic questions; fresh rollouts do not remove that exposure difference.
No diagnostic threshold tuning, loss change or strict-scorer relaxation was made.

Metrics SHA: `1d5e2bcfd801a56d00a2644c6890321741b3f5a1bebd9a8719340e11ddca52a9`.
Scalar artifact: `two-stage-periodic-000120-fit-curve.json`.

## Independent qualification and actual121 use

The staged66f3dd86bc37 CPU reader passed full SHA and actual CPU deserialization
of the fitted native probe, buffers, Adam and all four rank RNG states. It
verified the three epoch boundaries and source joint. A remote hardlink pins
the immutable payload. CUDA initialization was false; peak RSS was1,397,055,488B.
This qualifies the probe, not actor-payload CPU loading, GPU forward or a
stopped-writer recovery rehearsal. Model/tensor/raw-rollout data remained remote.

Fitted head SHA:
`a49d88ba2ff5459e4cdae8e70cc2792abb124f318cf9a8ca3b14ec6d891c899d`.
Remote CPU proof: `preparation/live-probe-000120-v1-full-sha-cpu.json`, SHA
`0903c5a63fcbc867d00a9e2f48363c851caaf1d3903a1dd9e016ca43c86ab14b`.
The exact3240-byte proof was mirrored locally and its SHA verified.

The independent periodic acceptance also checked actor120's20-opportunity
ledgers,150 real probe writes/rank, epoch counts, strict-G5 membership and
Mixed selection, student-save-before-fit ordering, and preserved CPU stat seals.
Actual121 scored using the exact1450-step head, sampled from saved actor0120,
and restored all four ranks at Adam2400/scheduler120 with FP32 master parameters.
At acceptance, each rank had performed11 real writes and reached2411. These
are live complete-line ledger snapshots, not producer-final ledger SHA claims.
Intermediate epoch payload full SHA and full actor CPU loading were not repeated.

Acceptance: `P10_THREE_EPOCH_CPU_QUALIFIED_AND_NEXT_REAL_NEW_HEAD_ACTOR_USE_ACCEPTED`.
Remote proof: `preparation/two-stage-periodic-000120-actual-next-use.json`, SHA
`e5df65c775b5c99e283223ad0f46749c501c15f8fc4523d6e99e52d4714fc4fe`.
Its exact4682 bytes were mirrored locally and verified. Actual121 score SHA:
`3894218d473492ac60ae91f665c2938ae49f1adcf48a3db112443f01746b39d4`.

The next formal fixed512 QA milestone remains150. Later P10 updates,
Prefix80/grounding150, remaining checkpoint acceptance, final51713 QA and
resource cleanup remain incomplete. The full goal is not complete.

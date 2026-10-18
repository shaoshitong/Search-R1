# R4 P110: three epochs, probe CPU qualification and actual111 use accepted

Job733898 continues on ACD1-6 with four shared H100 GPUs and immutable runtime
2000d46a530c64d78be7802bf11175a68b65effe. This is an intermediate milestone,
not completion of the200-outer/final51713 objective. No runtime source, loss,
scorer, selection threshold, allocation or optimizer was changed for this audit.

## Real collection and continuous fitting

The P10 pool used2560 different questions x5, at Prefix40. The fixed512G5
probe diagnostic was independently generated from the same frozen sampling
actor0109. Both feature/label sets therefore describe actor0109, rather than
the subsequently updated actor110. These are current native layer27 features;
initial warmup tensors were not reused for this fit.

Actor110 completed20 actual Adam writes/rank, zero skips, reaching2200 with
scheduler110. Its student checkpoint was saved before probe optimization.
The existing probe continued1150->1200->1250->1300 through three epochs,
150 actual writes/rank. The native block, readout, Adam and RNG were retained;
there was no new Prefix expansion or student-parameter reinitialization at110.

Actor110 selected368/512 actual Mixed questions (71.875%) with144 fills from
the full2560 pool. This selection used the pre-fit1150-step probe. It cannot
be attributed to the new1300-step probe or compared directly with ordinary
rounds' smaller1024-question full pools.

Actual joint110 SHA:
`636ab3c1a1bca0baef982ae4423494c1af9f343e89a4fc8b4d6c7e05d8f6aea0`.
The joint-to-joint cycle took861.392s; the fitting loop itself took41.384s,
and its full phase including startup/evaluation took88.125s.

## Same-data fitting curve

These accuracies classify strict Mixed at q>=0.5. They are not formal QA EM.
The same2560 training groups and512 diagnostic groups are evaluated in all rows.

| Epoch | Adam/rank | Train loss | Diagnostic loss | Train accuracy | Diagnostic accuracy |
|---|---:|---:|---:|---:|---:|
|Before|1150|1.416704|1.901002|62.9297%|43.9453%|
|1|1200|1.216577|1.998841|70.2734%|48.8281%|
|2|1250|0.885672|1.928766|81.6016%|43.3594%|
|3|1300|0.768283|1.931539|86.6797%|43.7500%|

Training fitting improved substantially, while diagnostic loss worsened slightly.
Diagnostic balanced accuracy declined55.6796%->54.4785%; training balanced
accuracy rose55.9674%->87.5863%. This is not evidence that generalization is
fixed. The two current sets are disjoint, but historical EMA5 warmup covered
the eligible training-question pool and excluded the diagnostic questions;
fresh rollouts do not erase that historical exposure difference. No heldout
threshold tuning or scorer relaxation was performed.

Metrics SHA: `947a5ec946e38260fc28f5e8c3eda62fd43d5aa71b518ba167e6d32f5118131b`.
Scalar artifact: `two-stage-periodic-000110-fit-curve.json`.

## Independent full probe SHA and CPU readback

The staged66f3dd86bc37 periodic reader passed full SHA and actual CPU
deserialization for the immutable fitted head, native buffers, Adam and four
rank RNG states. It checked three epoch boundaries and source joint identity.
A remote hardlink pins the successful probe payload against rotating aliases.
Actual CUDA initialization was false; peak CPU RSS was1.397GB. This qualifies
the stored probe, not actor tensor readback, GPU forward or a stopped-writer
recovery rehearsal. No model/tensor/raw-rollout payload was moved to Mac.

Fitted head SHA:
`b2dc5883438f4df32474baf8196641cfefbf668ff546028ee93d702a300b463a`.
Remote CPU proof: `preparation/live-probe-000110-v1-full-sha-cpu.json`, SHA
`8773c5ad43859eb47ebdb1fa0ffd38a561519d5e0794976a10813a780a42190d`.
The exact3240-byte proof was mirrored locally and its SHA verified.

## Actual111 use accepted

The separate periodic reader accepted the actual111 score binding to the exact
fitted1300-step head, normal5120-prefix/1024-full/512-actor budget, sampling
from saved actor0110, and all four real restores from student110. Each rank
restored Adam2200, scheduler110 and FP32 model master parameters. Subsequent
actual write counts observed were6/7/7/7, reaching2206/2207/2207/2207. These
are live complete-line ledger snapshots; the slight count difference is an
observation-time boundary, not a claim that complete111 ranks diverged.

It also independently checked the20-opportunity actor110 ledgers, all150 real
probe writes/rank and epoch counts, stored strict-label G5 membership and
actual Mixed selection, student-save-before-probe ordering, and the preserved
CPU-proof stat/filesystem seals. It did not repeat full SHA/CPU loading of the
actor checkpoint or intermediate epoch tensor payloads.

Acceptance status: `P10_THREE_EPOCH_CPU_QUALIFIED_AND_NEXT_REAL_NEW_HEAD_ACTOR_USE_ACCEPTED`.
Remote proof: `preparation/two-stage-periodic-000110-actual-next-use.json`, SHA
`c1510f1b8d10ce55e5546c351dd8099125898c950c28a27b7c0ab83a7b5319ca`.
Actual111 score manifest SHA:
`a2e53d616441461c0a93eebb788ffce1fd44272a93af7144c64d12a424336aa5`.

This is not formal fixed512 QA; the next formal QA milestone remains150.
Training through200, later probe updates, Prefix80/grounding150, final51713
QA, remaining checkpoint qualification and resource cleanup remain incomplete.
The full goal remains active.

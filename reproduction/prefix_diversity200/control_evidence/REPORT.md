# Search-G1 Top1024 continuation / EM-diversity Top512

Actual completed joint training: **3/200**. Latest submitted continuation is **3421 on h100-10 /10.33.0.18**, actually completed outer3 and advanced into outer4. Actor and probe durable Adam counters are60 per each of8 ranks; probe has an additional20 failed unsaved writes/rank from the earlier diagnostic failure. No earlier complete outer is recomputed.

| Complete outer | Continued1024 mixed | Actor512 mixed | Actor512 mean EM variance | Probe fixed512 ordinary / balanced |
|---|---:|---:|---:|---:|
|1|66.60%|100%|.216875|61.52% /60.24%|
|2|66.70%|100%|.2134375|58.79% /58.96%|
|3|67.09%|100%|.21625|61.52% /60.24%|

These are training-selection and probe-classification metrics, not held-out QA gains. Actor-selected512 contains only mixed positives in both rounds, so its balanced classification accuracy is undefined; the full1024 probe training set has both classes. Its post-update mixed balanced accuracy is50% in both rounds because all selected groups are predicted mixed at threshold.5. The fixed512 diagnostic fluctuates. Initial actual retainedCP100 balanced accuracy was59.87592805%; no new warmup was run.

The probe loss has20 global points/outer. Outer1 first/last1.65360072/1.44913451; batches differ, so this is not a monotonic same-batch improvement claim. Complete per-iteration scalar evidence and CSV remain beside this report. Normal outer2 six-stage wall total417.907778971s (6.97min), excluding startup, failed attempts and engineering/repair delays. Those overheads remain in original remote native/phase logs and terminal/cost proofs.

Scientific protocol:5120 diagnostic-excluded questions x5 same-actor20-policy-token prefixes; stable predicted mixed probability Top1024; continue exactly5120 original trajectories; select512 intact G5 by population strict-EM varianceK(5-K)/25, frozen prediction then original ordinal for ties. Actor uses unchanged Search-G1 grounding soft reward and20Adam/mini128/epochs1/initial globalmicro128 local16. Probe uses ALL1024 continued G5 naturally once,20Adam/rank, averageglobalbatch256, LR1e-6, trajectory-mean BCE plus G5 mixed BCE with weights1/1. Uncontinued4096 have no outcome labels. Both5120 candidate questions and total budgets remain fixed. Data/caches/weights/results stay remote.

Failure history:3416 failed before any optimizer write due diagnostic array absolute paths, fixed with explicit shared aliases.3417 completed actor20 and physically wrote probe20 but failed1024-group diagnostics before saving probe; that probe20 was recomputed once and is accounted separately.3418 actually committed joint1 and joint2, then failed while pruning an imported historical actor-directory alias. State2 was preserved. The new joint importer retains only original actor receipt references, probe/state and exact candidate cursor10240; it starts3 and does not copy rotating history directories. Current frozen source9ce4b14ae1f8fec52ec3e21214a0a50867587bbb was normally committed/pushed and all1240 deployed Git files matched SHA. No running code or another agent task was modified.

No new QA results for this experiment yet. Existing Search-G1 QA protocol remains8TP1/FP32/greedy/full500 action/raw500 observation/four turns; fixed512 at prior milestones, full final seven datasets51713. Checkpoint saves rolling and5/50/100/150/200. Goal completion remains pending actual200, fullQA, HF/tokenizer/native identity and finite/update checks, costs, and owned cleanup. The previously completed Tree200 experiment is preserved separately; do not conflate its53.82%F1/46.94%EM with this new run.


Actual resume acceptance: all8 original actor3 ledgers read ACTUAL_RESTORE Adam40/scheduler2/FP32 master, then20 writes to60; probe3 ledgers each20 natural writes40-to60. Joint3 SHA a4aef5933ac1e71e0f0089d306a996560e1139cb3cd54d714b1af5e09a9c2b2e and next outer4 actual stage are recorded in JOINT3_DIRECT_RESTORE_PROOF.json. Normal outer3 six-phase total419.492678102s=6.99min. No newQA score yet. Goal remains active.

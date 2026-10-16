# Search-G1 predictedTop1024 / actualEM-varianceTop512

Actual joint training complete12/200. Same3422 RUNNING on h100-10 /10.33.0.18 at outer-000013-actor-92b3eb4f830241e58a107b82a1e8def6. Saved actor/probe240 per each8rank; physicalprobe260/rank includes historical20unsaved writes. Actual11–12 original8rank actor/probe ledgers each20newwrites verified; no completedouter recomputation/reset. Prior restored6–10 plus actual11generation acceptance remains valid. Healthy training continues200.

QA5 complete512: EM24.60937500%, F130.60965402%; all8 actual worker exit0. Exact fixed validation SHA3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb. NQ256 and Hotpot256,8TP1 FP32 greedy/full500 action/raw500 observation/four search turns, no answer reserve or early final. This is the current experiment first measured QA; no final-effect claim. Full outputs stay remote; scalar proof QA5_SUMMARY.json.

| Complete outer | Continued1024 mixed | Actor512 mixed | Actor512 meanEM variance | Probe fixed512 ordinary / balanced |
|---|---:|---:|---:|---:|
|1|66.60%|100.00%|0.216875|61.52% / 60.24%|
|2|66.70%|100.00%|0.213438|58.79% / 58.96%|
|3|67.09%|100.00%|0.216250|61.52% / 60.24%|
|4|65.33%|100.00%|0.212500|62.70% / 56.67%|
|5|65.53%|100.00%|0.214375|62.30% / 60.36%|
|6|62.99%|100.00%|0.212188|60.74% / 59.75%|
|7|61.23%|100.00%|0.207656|56.25% / 59.65%|
|8|59.08%|100.00%|0.203125|62.30% / 57.57%|
|9|57.32%|100.00%|0.203437|62.11% / 56.82%|
|10|59.77%|100.00%|0.205156|60.55% / 60.09%|
|11|56.35%|100.00%|0.203125|59.57% / 60.42%|
|12|54.20%|100.00%|0.198437|62.89% / 55.85%|

Probe diagnostics classify group mixing and are distinct from QA EM/F1. Actor512 allmixed makes its balanced classification accuracy undefined; no artificial positive claim from trivial ordinaryaccuracy1. Full1024 online train has both classes, but postupdate allpredictedmixed at threshold.5, balancedaccuracy50%. Uncontinued4096 have no outcome labels.

Normal six-phase totals outer2/3/4/5:417.91/419.49/416.20/412.37s (~6.9–7.0min), excluding setup, QA, failure and engineering delay. True consecutive outer intervals will be recorded from resumed6 onward. Per-outer20globalAdam loss points are retained; differing natural minibatches do not imply monotonic fitting.

Protocol:5120 diagnostic-excluded candidate questions ×5 original20-policy-token prefixes, frozen actor/probe while selecting stable predictedmixedTop1024. Continue those original5120 trajectories, keep authentic token/context/sampling records. StrictEM varianceK(5-K)/25 selects512 intactG5, ties by frozenh then originalordinal. Actor uses original Search-G1 grounding softreward PPO, mini128/epochs1/globalmicro128 local16/noaccum; only actualtypedOOM permits budget-preservingmicro fallback. ActorBF16compute/FP32master andAdam, LR1e-6 with57completeouterwarmup. Probe uses ALL1024/5120 naturally once,20synchronousAdam/rank, globalbatch255/260 average256, LR1e-6, trajectorymeanBCE+ownG5mixedmeanBCE weights1/1. Best actualretainedwarmupCP100 model-only initialization, no rewarmup. OfficialQwen2.5-3B-Instruct revisionaa8e72537993ba99e69dfaafa59ed015b17504d1.

Current frozen runtime2c1219defaebaf69467fce18efad331ce68cd6ec,1246files allremoteSHA matched. Actual configSHA2973c362cf039cb60d844e1be8d9e4332cdcf5f9d756ea6d4976adea463948e6. Local read-only watcher is latercontrolcommit e068855a and does not hotchange runtime. All weights/optimizer/feature/rollout/fullresults stay sameH100cluster remote.

Failurehistory:3416 preAdam diagnostic path failure;3417 actor20saved but probe20unsaved diagnostic1024failure (only probe20 recomputed/accounted);3418 joint2saved then imported-history prunerfailure;3421 actuallycompleted3–5 thenQA5hit H100 FP32 Triton MMA compilerassert. Source5 is independently fullSHA/CPU accepted (proofbe541a6076e1c2e0f2c022ba7eb2f31a10341bb0dc8d19e0e758904e61409159):435FP32HFtensors,8rankmodel/Adam/scheduler/RNG,driver,probe,cursor25600/next6. CPU acceptance is not executor/CUDA exactresume. Original38terminal logs/fullSHA and1089births/cgroup/GPU release preserved in terminalproof f805e7d897ac3dee8863e804a2c7809b214c4958ca3d59853cf4198b00904d1a.

3422 imported joint5 unchanged, nowactualQA5success after switching only incompatible prefix/chunkedFP32kernel to fullprefillXFormers and adding PythonheaderCPATH. PendingQA is explicitly retained/retried before finalacceptance, rather than cancelling valid training. No completedtrainingrecomputed. Restored6–10 and actual11generation verified in RESUMED_FIVE_ACCEPTANCE.json; reader25091 exited0, training3422 remainsrunning.

Goal remains ACTIVE/incomplete: actual200, milestone50/100/150/200, completefinalseven51713QA, native/HF/tokenizer state verification, finite/nonzeroupdate/cost/source/split identity and ownresourcecleanup remain. OldTree200/51713 finalF1 53.8154%/EM46.9398% is completed separately and is not thisrun’s score.

Additional cost evidence: completed1–12 actualsampledtokens11,982,868; native prefix/newcontinuation requests and6phasewall perouter retained in COST_SNAPSHOT.json. Failed attempts/QA/nativeidle/setup remain separate. Recent fulljoint intervals[418.8847117424011, 418.30745673179626, 421.82969784736633, 419.43331837654114], mean419.614s (~6.994min).

Passive resource sampler g1-resource5s-f3ee1435e501-3422 is independently staged outside frozenrepo, source9f3cf693c91aad312d0badaa0fd0d7451eee36fd/readerSHA80e0e3383b1455abfa6f0c908192b12e471d865cf4edc60f3bf651ca386fefee, CPUQuota10%/Memory128M/Nice19, noGPU allocation/forward or trainingconfig change. Two actualobservations prove9->32samples and14038->50163bytes, noerrors; full5sJSONL staysremote. GPUpeak scope is whole physicalcards only since telemetry creation, not continuous/historical peaks; already-observed earlier80GB-nearcapacity snapshots remain valid. CPU is an explicitly incomplete lowerbound over observedownedbirth utime+stime because Slurm exposes onlyfreezer, nojobCPUaccounting. Short-lived/earlierexited processes may be missing; never report this as fullCPUcost. RESOURCE5S_ACCEPTANCE.json andlatestsummary record coverage. Fullretrieval/costaccounting remains outstanding.

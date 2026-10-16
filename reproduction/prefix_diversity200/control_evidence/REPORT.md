# Search-G1 predictedTop1024 / actualEM-varianceTop512

Actual complete18/200; job3422 RUNNING on h100-10/10.33.0.18, stage outer-000019-prefix. Saved actor/probe Adam per8rank: [360, 360, 360, 360, 360, 360, 360, 360] / [360, 360, 360, 360, 360, 360, 360, 360]. Completed joint counters: [360, 360, 360, 360, 360, 360, 360, 360] / [360, 360, 360, 360, 360, 360, 360, 360]. Historical failed unsaved probe20/rank remains separate; never sum8replicas as globalupdates.

Latest QA: outer5,512questions EM24.6094% / F130.6097%. QA uses unchanged8TP1 FP32 greedy/full500/raw500/foursearch/noanswerreserve/noearlyfinal. Classification diagnostics below are distinct from QA EM/F1. Full final51713 remains required.
Recent actual same3422 completejoint intervals[441.612713098526, 441.2780020236969, 445.81118178367615, 438.8169014453888]; mean441.880s=7.365min. Startup,QA,failure/repair are separate.

Latest actor512 mixed91.6016%, meanEMvariance0.17875000, selected frozen meanpredictedmixed0.73965986. Continued1024 mixed45.8008%; uncontinued4096 have no outcome labels.

|Complete outer|Continued1024 mixed|Actor512 mixed|Actor512 variance|Probe train ordinary / balanced|Probe fixed512 ordinary / balanced|
|---|---:|---:|---:|---:|---:|
|1|66.60%|100.00%|0.216875|66.60% / 50.00%|61.52% / 60.24%|
|2|66.70%|100.00%|0.213438|66.70% / 50.00%|58.79% / 58.96%|
|3|67.09%|100.00%|0.216250|67.09% / 50.00%|61.52% / 60.24%|
|4|65.33%|100.00%|0.212500|65.33% / 50.00%|62.70% / 56.67%|
|5|65.53%|100.00%|0.214375|65.53% / 50.00%|62.30% / 60.36%|
|6|62.99%|100.00%|0.212188|62.99% / 50.00%|60.74% / 59.75%|
|7|61.23%|100.00%|0.207656|61.23% / 50.00%|56.25% / 59.65%|
|8|59.08%|100.00%|0.203125|59.08% / 50.00%|62.30% / 57.57%|
|9|57.32%|100.00%|0.203437|57.32% / 50.00%|62.11% / 56.82%|
|10|59.77%|100.00%|0.205156|59.77% / 50.00%|60.55% / 60.09%|
|11|56.35%|100.00%|0.203125|56.35% / 50.00%|59.57% / 60.42%|
|12|54.20%|100.00%|0.198437|54.20% / 50.00%|62.89% / 55.85%|
|13|55.96%|100.00%|0.199375|55.96% / 50.00%|62.30% / 57.09%|
|14|52.93%|100.00%|0.200469|52.93% / 50.00%|63.09% / 59.15%|
|15|50.29%|100.00%|0.195937|50.29% / 50.00%|62.30% / 57.33%|
|16|50.59%|100.00%|0.197656|50.59% / 50.00%|63.48% / 59.21%|
|17|48.54%|97.07%|0.189531|49.12% / 50.57%|64.06% / 55.30%|
|18|45.80%|91.60%|0.178750|47.27% / 51.22%|60.16% / 58.45%|

20-point probe losses and all6stagewall durations are in COMPLETED_ITERATION_SCALAR_EVIDENCE.json and ITERATION_METRICS.csv. Different natural batches mean first/last loss is not a monotonic same-batch fitting claim. Actor-selected allpositive balancedaccuracy is undefined when there is no negativeclass.

Protocol: officialQwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1; best actuallyretainedwarmupCP100 model-only probe initialization, no rewarmup.5120distinct diagnostic-excluded questions/G5prefix20, frozenactor/probe stable predictedmixedTop1024, continue authentic same5prefixes, population strictEMvarianceK(5-K)/25 selects512 intactG5 with frozenh/originalordinal ties. No extra sampling to force mixing. Actor originalSearch-G1 groundingsoftreward PPO20Adam/mini128/epoch1/globalmicro128 local16/noaccum, BF16compute/FP32masterAdam, LR1e-6/warm57outer. ProbeALL1024/5120 naturallyonce,20synchronousAdam/rank/globalbatch255/260avg256, LR1e-6, trajectorymeanBCE+ownG5mixedmeanBCE weights1/1. Fixeddiagnostic512 stays excluded. Only realtypedOOM permits preserving-budget micro fallback.

Frozen runtime2c1219defaebaf69467fce18efad331ce68cd6ec/actualconfigSHA2973c362cf039cb60d844e1be8d9e4332cdcf5f9d756ea6d4976adea463948e6. Later controls/results do not hotchange training. All models/native/optimizer/raw/fulloutputs stay sameH100cluster remote; local artifacts are small scalar/code summaries.

History:3416 preAdam pathfailure;3417 actor20saved/probe20unsaved diagnosticfailure (onlyprobe20recomputed/accounted);3418 savedjoint2 then imported-history prunerfailure;3421 savedjoint5 thenFP32H100TritonMMAcompilerassert inQA.3422 imported5 unchanged, compatiblefullprefillXFormers/CPATH fixedQA. Source5 independentfullSHA/CPU PASS:435FP32HFtensors,8nativeAdam/scheduler/RNG/driver/probe/pool; proofbe541a6076e1c2e0f2c022ba7eb2f31a10341bb0dc8d19e0e758904e61409159, acceptance5/. CPUload is not exactCUDAexecutorresume. Restored6–10+actual11generation accepted; failednative/log/birth/cost evidence retained. OldTree200/51713 is complete separately, not thisrun score.

Cost: COST_SNAPSHOT.json contains existing native token/request metadata and phasewall with explicit coverage; failed attempts/QA/setup/idle separate. Passive resource5s unit runs outside frozenrepo (CPU10%,Memory128M,Nice19), noGPUforward or trainingcontrol. Wholecard sampledpeaks cover only since22:49:59CST, not historical/continuouspeaks or amplecapacity. CPU observedbirth utime+stime is an incomplete lowerbound (freezeronly, past/shortlived CPU may be missing), never fullCPUcost. Raw telemetry staysremote. Fullretrieval and combinedGPU/CPU accounting remain outstanding.

Completion remains unproven: actual200, normal50/100/150/200checkpoint/QA, fullseven51713, HF/tokenizer/nativefinite/nonzeroupdate/source/split identity, cost/ownedcleanup and finalpublication. GoalACTIVE; no stop/restart is performed by this reader.

Latest passive telemetry515 samples, last observed1792164777.029278; errors[], wholecard maxused80663MiB/minfree419MiB (coverage above).

At completeouter18, only469/1024 continuedgroups are strictEMmixed. All469 are selected plus43zeroEMvariance groups to keep512G5; actualactormixed469/512=91.6015625%. Outer17 had497mixed/512=97.0703125%. No extra sampling/drop/relabel/epsilon was used to force100%mixing. This measures EMdiversity, not equality of continuousSearch-G1softrewards or a heldoutQA gain.

CPU accounting evidence remainsexplicit: sacctdisabled; sstatNTasks0/overflow-likeAveCPUinvalid andunused. Actualjobfreezer stable11PID own+waitedCPU42156.72coresec @1792164092.3334658 is an observedcumulativeprocess-tree snapshot, not final/completeCPUcost. SeeCPU_ACCOUNTING_EVIDENCE.json;5sownedbirthlowerbound andwholecardpeaks stayseparate.

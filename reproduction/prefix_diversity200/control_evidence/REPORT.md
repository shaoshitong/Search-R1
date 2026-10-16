# Search-G1 predictedTop1024 / actualEM-varianceTop512

Actual complete27/200; job3422 RUNNING on h100-10/10.33.0.18, stage outer-000028-continue. Saved actor/probe Adam per8rank: [540, 540, 540, 540, 540, 540, 540, 540] / [540, 540, 540, 540, 540, 540, 540, 540]. Completed joint counters: [540, 540, 540, 540, 540, 540, 540, 540] / [540, 540, 540, 540, 540, 540, 540, 540]. Historical failed unsaved probe20/rank remains separate; never sum8replicas as globalupdates.

Latest QA: outer5,512questions EM24.6094% / F130.6097%. QA uses unchanged8TP1 FP32 greedy/full500/raw500/foursearch/noanswerreserve/noearlyfinal. Classification diagnostics below are distinct from QA EM/F1. Full final51713 remains required.
Recent actual same3422 completejoint intervals[433.3234586715698, 435.53052711486816, 432.6748833656311, 429.9025390148163]; mean432.858s=7.214min. Startup,QA,failure/repair are separate.

Latest actor512 mixed73.2422%, meanEMvariance0.14156250, selected frozen meanpredictedmixed0.77782712. Continued1024 mixed36.6211%; uncontinued4096 have no outcome labels.

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
|19|45.21%|90.43%|0.176719|45.21% / 50.00%|62.30% / 55.76%|
|20|46.00%|91.99%|0.180469|46.19% / 50.17%|62.50% / 58.45%|
|21|45.02%|90.04%|0.172813|45.02% / 50.00%|62.30% / 57.09%|
|22|41.31%|82.62%|0.161250|43.85% / 52.06%|63.48% / 54.97%|
|23|41.11%|82.23%|0.158594|42.19% / 50.55%|62.11% / 57.42%|
|24|39.06%|78.12%|0.150469|39.06% / 50.00%|63.87% / 57.09%|
|25|39.16%|78.32%|0.154688|39.26% / 50.08%|63.09% / 55.64%|
|26|35.35%|70.70%|0.135781|36.43% / 50.83%|63.48% / 54.73%|
|27|36.62%|73.24%|0.141563|39.55% / 51.80%|63.67% / 54.52%|

20-point probe losses and all6stagewall durations are in COMPLETED_ITERATION_SCALAR_EVIDENCE.json and ITERATION_METRICS.csv. Different natural batches mean first/last loss is not a monotonic same-batch fitting claim. Actor-selected allpositive balancedaccuracy is undefined when there is no negativeclass.

Protocol: officialQwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1; best actuallyretainedwarmupCP100 model-only probe initialization, no rewarmup.5120distinct diagnostic-excluded questions/G5prefix20, frozenactor/probe stable predictedmixedTop1024, continue authentic same5prefixes, population strictEMvarianceK(5-K)/25 selects512 intactG5 with frozenh/originalordinal ties. No extra sampling to force mixing. Actor originalSearch-G1 groundingsoftreward PPO20Adam/mini128/epoch1/globalmicro128 local16/noaccum, BF16compute/FP32masterAdam, LR1e-6/warm57outer. ProbeALL1024/5120 naturallyonce,20synchronousAdam/rank/globalbatch255/260avg256, LR1e-6, trajectorymeanBCE+ownG5mixedmeanBCE weights1/1. Fixeddiagnostic512 stays excluded. Only realtypedOOM permits preserving-budget micro fallback.

Frozen runtime2c1219defaebaf69467fce18efad331ce68cd6ec/actualconfigSHA2973c362cf039cb60d844e1be8d9e4332cdcf5f9d756ea6d4976adea463948e6. Later controls/results do not hotchange training. All models/native/optimizer/raw/fulloutputs stay sameH100cluster remote; local artifacts are small scalar/code summaries.

History:3416 preAdam pathfailure;3417 actor20saved/probe20unsaved diagnosticfailure (onlyprobe20recomputed/accounted);3418 savedjoint2 then imported-history prunerfailure;3421 savedjoint5 thenFP32H100TritonMMAcompilerassert inQA.3422 imported5 unchanged, compatiblefullprefillXFormers/CPATH fixedQA. Source5 independentfullSHA/CPU PASS:435FP32HFtensors,8nativeAdam/scheduler/RNG/driver/probe/pool; proofbe541a6076e1c2e0f2c022ba7eb2f31a10341bb0dc8d19e0e758904e61409159, acceptance5/. CPUload is not exactCUDAexecutorresume. Restored6–10+actual11generation accepted; failednative/log/birth/cost evidence retained. OldTree200/51713 is complete separately, not thisrun score.

Cost: COST_SNAPSHOT.json contains existing native token/request metadata and phasewall with explicit coverage; failed attempts/QA/setup/idle separate. Passive resource5s unit runs outside frozenrepo (CPU10%,Memory128M,Nice19), noGPUforward or trainingcontrol. Wholecard sampledpeaks cover only since22:49:59CST, not historical/continuouspeaks or amplecapacity. CPU observedbirth utime+stime is an incomplete lowerbound (freezeronly, past/shortlived CPU may be missing), never fullCPUcost. Raw telemetry staysremote. Fullretrieval and combinedGPU/CPU accounting remain outstanding.

Completion remains unproven: actual200, normal50/100/150/200checkpoint/QA, fullseven51713, HF/tokenizer/nativefinite/nonzeroupdate/source/split identity, cost/ownedcleanup and finalpublication. GoalACTIVE; no stop/restart is performed by this reader.

Latest passive telemetry1316 samples, last observed1792168789.2039602; errors[], wholecard maxused81065MiB/minfree17MiB (coverage above).

## Native reward and cost evidence

Original saved actor grounding soft reward on selected512G5: outer1 mean0.72182053; outer27 mean0.94058778, min-0.10000000, max1.59700072. Version15bb15365346401da1d919e9cde8f813:0:a72f2e62b8674787bee00266a8eef79e. All eight rank statistics finite,2560 trajectories per outer, prefix correctness head not used as reward. This is selected training-distribution reward, not heldout QA and not proof of probe-selection benefit. TRAINING_REWARD_HISTORY.json binds each row to its original joint receipt SHA256; no additional forward was run.
Committed generation through outer27: 25398999 sampled tokens; logical retrieval through outer27: 147805 calls, backlog[]. Inherited prefix searches subtracted; actor512 subset not added. Failed attempts, QA, HTTP retries/cache misses and physical FAISS calls remain separately scoped.
Native reservation wall: historical terminal3416/3417/3418/3421 total6.004444 GPU-hours, including useful completed updates and failures/setup/QA; current running3422 brings observed reserved total to28.108889 GPU-hours, lower bound at native observation. Do not add phase wall again or label whole failed jobs wasted. Reserved CPU-core-hours are capacity, not CPU usage. sacct disabled/sstat invalid; actual CPU process-tree/birth snapshots remain partial. Full cost report and goal completion remain outstanding.

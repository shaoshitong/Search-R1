# Search-G1 predictedTop1024 / actualEM-varianceTop512

Actual complete68/200; job3427 RUNNING on h100-10/10.33.0.18, stage outer-000069-prefix. Saved actor/probe Adam per8rank: [1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360] / [1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360]. Completed joint counters: [1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360] / [1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360]. Historical failed unsaved probe20/rank remains separate; never sum8replicas as globalupdates.

Latest QA: outer50,512questions EM28.3203% / F134.4268%. QA uses unchanged8TP1 FP32 greedy/full500/raw500/foursearch/noanswerreserve/noearlyfinal. Classification diagnostics below are distinct from QA EM/F1. Full final51713 remains required.
Recent actual completejoint intervals within unchanged execution base/source[415.6608910560608, 401.252818107605, 415.1195182800293, 412.10737466812134]; mean411.035s=6.851min. These wall intervals include any preceding milestoneQA and scheduling overhead; pure per-outer phase durations are recorded separately. Startup and terminal failure/repair costs are separate.

Latest actor512 mixed38.0859%, meanEMvariance0.07312500, selected frozen meanpredictedmixed0.83396964. Continued1024 mixed19.0430%; uncontinued4096 have no outcome labels.

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
|28|34.57%|69.14%|0.136875|36.62% / 51.37%|65.43% / 58.67%|
|29|33.11%|66.21%|0.125156|40.82% / 51.00%|64.65% / 54.43%|
|30|35.06%|70.12%|0.136563|40.33% / 52.65%|65.23% / 56.46%|
|31|33.40%|66.80%|0.131094|42.58% / 54.12%|63.87% / 54.06%|
|32|31.54%|63.09%|0.121719|35.45% / 50.93%|64.45% / 55.00%|
|33|33.69%|67.38%|0.129688|34.57% / 50.31%|65.04% / 57.76%|
|34|31.35%|62.70%|0.119688|39.75% / 51.97%|64.45% / 54.27%|
|35|31.64%|63.28%|0.123281|40.62% / 54.00%|64.06% / 54.46%|
|36|30.86%|61.72%|0.122188|44.63% / 54.96%|64.45% / 54.15%|
|37|32.62%|65.23%|0.125000|41.60% / 52.57%|63.67% / 53.79%|
|38|30.08%|60.16%|0.116719|36.62% / 52.37%|63.87% / 54.79%|
|39|30.37%|60.74%|0.118594|33.01% / 51.17%|65.82% / 57.88%|
|40|30.27%|60.55%|0.117344|35.45% / 50.24%|64.45% / 55.12%|
|41|26.37%|52.73%|0.099687|27.93% / 51.06%|65.62% / 57.61%|
|42|26.46%|52.93%|0.102500|27.15% / 50.35%|63.67% / 57.06%|
|43|26.76%|53.52%|0.103281|26.76% / 50.00%|63.67% / 56.94%|
|44|26.17%|52.34%|0.102344|27.64% / 50.15%|64.84% / 56.52%|
|45|27.34%|54.69%|0.104688|27.44% / 50.07%|65.04% / 57.64%|
|46|26.66%|53.32%|0.102500|39.84% / 53.98%|63.67% / 53.91%|
|47|26.17%|52.34%|0.103125|47.66% / 55.28%|64.06% / 53.12%|
|48|23.63%|47.27%|0.091406|26.56% / 51.06%|64.65% / 56.37%|
|49|23.14%|46.29%|0.089688|30.27% / 53.16%|65.04% / 56.55%|
|50|23.54%|47.07%|0.091563|32.91% / 52.68%|64.06% / 54.46%|
|51|23.44%|46.88%|0.090938|29.88% / 51.03%|65.04% / 55.94%|
|52|22.75%|45.51%|0.088125|27.64% / 51.50%|64.65% / 56.49%|
|53|21.68%|43.36%|0.083906|22.75% / 50.52%|63.67% / 56.46%|
|54|22.95%|45.90%|0.088438|30.76% / 52.83%|64.26% / 54.97%|
|55|23.05%|46.09%|0.088750|25.29% / 50.12%|63.09% / 57.33%|
|56|21.00%|41.99%|0.080625|21.00% / 50.00%|63.67% / 56.58%|
|57|21.00%|41.99%|0.080625|48.05% / 56.19%|64.06% / 53.24%|
|58|21.88%|43.75%|0.084062|38.18% / 52.24%|63.87% / 53.45%|
|59|22.95%|45.90%|0.089375|27.05% / 51.02%|64.26% / 55.94%|
|60|23.34%|46.68%|0.090000|40.23% / 50.69%|64.06% / 53.49%|
|61|22.27%|44.53%|0.086094|30.18% / 52.74%|64.06% / 55.43%|
|62|20.12%|40.23%|0.076719|27.54% / 52.65%|63.87% / 54.67%|
|63|22.75%|45.51%|0.088750|24.61% / 50.90%|63.09% / 55.03%|
|64|22.85%|45.70%|0.088438|23.83% / 50.48%|63.48% / 56.06%|
|65|20.90%|41.80%|0.080469|29.39% / 52.96%|65.04% / 56.31%|
|66|22.27%|44.53%|0.085938|22.36% / 48.50%|63.28% / 56.52%|
|67|22.46%|44.92%|0.086250|22.66% / 50.13%|63.87% / 57.70%|
|68|19.04%|38.09%|0.073125|19.14% / 50.06%|63.28% / 55.91%|

20-point probe losses and all6stagewall durations are in COMPLETED_ITERATION_SCALAR_EVIDENCE.json and ITERATION_METRICS.csv. Different natural batches mean first/last loss is not a monotonic same-batch fitting claim. Actor-selected allpositive balancedaccuracy is undefined when there is no negativeclass.

Protocol: officialQwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1; best actuallyretainedwarmupCP100 model-only probe initialization, no rewarmup.5120distinct diagnostic-excluded questions/G5prefix20, frozenactor/probe stable predictedmixedTop1024, continue authentic same5prefixes, population strictEMvarianceK(5-K)/25 selects512 intactG5 with frozenh/originalordinal ties. No extra sampling to force mixing. Actor originalSearch-G1 groundingsoftreward PPO20Adam/mini128/epoch1/globalmicro128 local16/noaccum, BF16compute/FP32masterAdam, LR1e-6/warm57outer. ProbeALL1024/5120 naturallyonce,20synchronousAdam/rank/globalbatch255/260avg256, LR1e-6, trajectorymeanBCE+ownG5mixedmeanBCE weights1/1. Fixeddiagnostic512 stays excluded. Only realtypedOOM permits preserving-budget micro fallback.

Frozen runtimee3463cfddca71a22ac684d75c875dbe123e8bc43/actualconfigSHA384cf0f516e7bf9ce3623bdf48e87a078f8eceaa5af5e03f238e04e86ae59ad8. Later controls/results do not hotchange training. All models/native/optimizer/raw/fulloutputs stay sameH100cluster remote; local artifacts are small scalar/code summaries.

History:3416 preAdam pathfailure;3417 actor20saved/probe20unsaved diagnosticfailure (onlyprobe20recomputed/accounted);3418 savedjoint2 then imported-history prunerfailure;3421 savedjoint5 thenFP32H100TritonMMAcompilerassert inQA.3422 imported5 unchanged, compatiblefullprefillXFormers/CPATH fixedQA. Source5 independentfullSHA/CPU PASS:435FP32HFtensors,8nativeAdam/scheduler/RNG/driver/probe/pool; proofbe541a6076e1c2e0f2c022ba7eb2f31a10341bb0dc8d19e0e758904e61409159, acceptance5/. CPUload is not exactCUDAexecutorresume. Restored6–10+actual11generation accepted; failednative/log/birth/cost evidence retained. OldTree200/51713 is complete separately, not thisrun score.

Cost: COST_SNAPSHOT.json contains existing native token/request metadata and phasewall with explicit coverage; failed attempts/QA/setup/idle separate. Passive resource5s unit runs outside frozenrepo (CPU10%,Memory128M,Nice19), noGPUforward or trainingcontrol. Wholecard sampledpeaks cover only the current sampler window recorded by started_unix and last_observed_unix in RESOURCE5S_SUMMARY_LATEST.json, not historical/continuouspeaks or amplecapacity. CPU observedbirth utime+stime is an incomplete lowerbound over observed owned process births; past/shortlived CPU may be missing, never fullCPUcost. Raw telemetry staysremote. Logical retrieval coverage is recorded separately in RETRIEVAL_COST_SNAPSHOT.json; final combinedGPU/CPU accounting remains incomplete.

Completion remains unproven: actual200, normal50/100/150/200checkpoint/QA, fullseven51713, HF/tokenizer/nativefinite/nonzeroupdate/source/split identity, cost/ownedcleanup and finalpublication. GoalACTIVE; no stop/restart is performed by this reader.

Latest passive telemetry1346 samples, last observed1792187045.8439436; errors[], wholecard maxused80815MiB/minfree267MiB (coverage above).

## Original reward and scoped costs

Original saved actor grounding reward outer68 selected512G5/2560 mean1.00830855, min-0.10000000, max1.59498024. All8rank statistics finite; prefix correctness head not used as reward. Original joint receipt hashes bind history1–68; no new forward. Selected training reward is distinct from heldout QA or proof of selection benefit.

Committed generation/retrieval through68/68: 59152862 sampled tokens, 358058 logical searches; backlog[]. Inherited prefix searches not doubled; actor512 subset not added. Failed attempts, QA, HTTP retries/cache misses and physical FAISS calls separately scoped.

Historical terminal reservations 51.802222 GPU-hours include useful saved updates and failure/setup/QA overhead. With current job, observed reserved total66.911111 GPU-hours is a lower bound for running jobs at native observation. Do not add phase wall again. Reserved CPU capacity is not actual CPU use; sacct disabled/sstat invalid, CPU snapshots partial. Goal200/final51713 and final cost report remain incomplete.

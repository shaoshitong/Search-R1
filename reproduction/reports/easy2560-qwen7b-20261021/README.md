# Easy2560 + Co rollout: official Qwen2.5-7B-Instruct

**Current evidence, 2026-10-23 22:04 CST:** production750491 has completed and
sealed outer100 on the same four H100; outer101 has saved all2560 candidate questions.
All4 have2000 actual Adam writes. Runtime remains frozen at dde1452; no restart,
new GPU allocation, runtime edit or extra monitor forward was introduced.

Fixed512 evaluation uses the same ordered NQ256/HotpotQA256 occurrences and
parquet SHA across milestones: **5=193/512 (37.6953%),50=259/512 (50.5859%),
100=280/512 (54.6875%)**. At100 NQ is119/256 (46.4844%) and HotpotQA161/256
(62.8906%):21 more correct than50, or4.1016 percentage points. All100 accepted
training log rows are present;91–100 mean training EM64.82%, sequence reward0.8448.
These changing selected512x5 pre-update samples are distinct from fixed512 tests.
No claim of final-quality equivalence or attribution to an individual optimization
follows from these curves.

All29 ordinary outers71–99 average **1493.029s (24m53.03s)**, versus old65–69
1846.455s:19.141% less elapsed time /1.2367x. Outer100 includes fixed512 validation
and its full adjacent-boundary wall is2256.530s (37m36.53s); do not silently mix
that milestone into ordinary-outer timing. All30 outers71–100, including100,
average1518.479s (25m18.48s). Whole wall includes checkpoints/seals/inter-outer
work; validation/checkpoint/inter-outer overhead at100 is not independently
decomposed. Different policy/data prevent strict same-input timing causality.

The100 full archive is accepted: **all1351 core files /121,925,546,599 bytes**
passed incoming SHA, independent fresh HKUST3 readback/fsync and fresh HPC3 source
readback. The eight-file post-callback boundary and seven transferred dependency
files are accepted; the dependency closure covers all manifests and binds the
already accepted core SHA. The saved boundary includes post-fixed512/grounding100
refit RNG and reward version, with callbacks complete. Both original archive
controllers have completed100 and moved to waiting for150, without failure.
This byte/state archive acceptance is not actual final CUDA recovery. The same
training job continues;101 has made all20 writes/rank (cumulative2020), while its
ordinary checkpoint seal remains pending at this observation.
See [full core/boundary/dependency receipts and scope](evidence/milestone100-archive-current-summary.json).
The200/no201 target, fixed150/200, exact native51713/seven scores, full milestone
archives and actual final native4 model/Adam/scheduler/RNG/dataequal recovery
remain required.

See [all100 training/fixed512 metrics](evidence/training-em-reward-fixed512-through100-20261023.json),
[through100 wall summary](evidence/combined-through100-summary.json) and
[actual timing/counter records](evidence/combined-through100-timing.json).

## Historical twenty-outer observation, 2026-10-23 17:12 CST

The following dated observations retain their original scope. Statements about
pending100 or a running older job describe that historical point, not current state.

**Current evidence, 2026-10-23 17:12 CST:** production750491 on ACD1-14/four H100
completed and sealed **all twenty outers71–90**; outer91 has saved512 candidate
questions. Whole-outer wall mean is **1459.074s (24m19.07s)** versus prior65–69
mean1846.455s (30m46.45s): **20.980% less elapsed time / 1.2655x speed**.
All twenty consecutive outers are included. Measurements use adjacent actual
before-generation boundaries, include checkpoint/seal/inter-outer work and exclude
startup restore. Different policy/data make this an observational production
comparison; final quality remains pending.

| Completed outer | Whole wall, seconds | Candidate fill, seconds |
| --- | ---: | ---: |
| 71 | 1410.022 | 652.194 |
| 72 | 1389.232 | 657.690 |
| 73 | 1405.625 | 662.507 |
| 74 | 1399.299 | 650.532 |
| 75 | 1379.569 | 649.508 |
| 76 | 1378.740 | 635.872 |
| 77 | 1402.034 | 655.441 |
| 78 | 1361.076 | 635.968 |
| 79 | 1509.812 | 640.881 |
| 80 | 1505.601 | 669.349 |
| 81 | 1539.334 | 670.684 |
| 82 | 1515.201 | 662.940 |
| 83 | 1472.430 | 647.380 |
| 84 | 1477.304 | 645.458 |
| 85 | 1487.279 | 639.741 |
| 86 | 1499.351 | 649.223 |
| 87 | 1519.126 | 659.902 |
| 88 | 1517.174 | 647.489 |
| 89 | 1489.827 | 665.544 |
| 90 | 1523.447 | 674.058 |

Actual all4 model/Adam/scheduler/RNG, driver and reconstructed-stream readbacks
passed from CP70. All4 now reached1800 Adam writes/scheduler90: every completed
outer performed20 actual updates and zero extra monitor forwards, used compact
actor FA3 Q28/KV4, and synchronized weights once across five candidate phases.
Native4 checkpoint seals were observed at every outer71–90; rolling latest is90.

Candidate fill mean is653.618s versus old1075.965s.
Actor mean is144.216s; adv199.701s and
checkpoint write191.086s. These component values must not
be summed to substitute for whole wall. The latest ten outers81–90 averaged
1504.047s (25m04.05s); checkpoint writing averaged217.041s,
versus165.130s in71–80. Residual sealing/inter-outer time is not independently
decomposed, so the recent slower wall is not attributed exclusively to writing.

Natural GPU retrieval service750471 completed341803 actual requests with0 failures
at its last observation. Both original archive-controller process births remain
live waiting for100, with no recorded failures. Legacy745198 remains untouched.
Selected mixed fraction across these twenty training pools is80.840%; this is
not heldout accuracy. Training81–90 mean EM is62.64% and reward0.8214,
from the changing selected512x5 pre-update trajectories. Current fixed512 tests
remain5=193/512(37.6953%) and50=259/512(50.5859%) on identical ordered questions;
100 has not run, and no new quality-equivalence claim is made.

The run remains active toward **200/no update201**. Fixed100/150/200 validation,
exact ordered native51713/seven-source metrics, final full archives and actual
final native4 model/Adam/scheduler/RNG/dataequal recovery remain required. Twenty
complete post-restore outers are an operational milestone, not goal completion.

See [twenty complete outer evidence](evidence/combined-twenty-complete-outers-summary.json),
[all timing/counter records](evidence/combined-twenty-complete-outers-timing.json),
and [training EM/reward and fixed512](evidence/training-em-reward-fixed512-through90-20261023.json).
The earlier [ten-outer evidence](evidence/combined-ten-complete-outers-summary.json)
and [five-outer evidence](evidence/combined-five-complete-outers-summary.json) remain historical evidence.

The following candidate-only tests are historical qualification evidence.

Search-only stop reduced512-question generation158.192→94.030s (40.56%)
with async retrieval, but only180.867→163.918s (9.37%) with the production-like
serial retrieval path. Both pairs completed0:0 on4H100. Raw tokens fell about68%
in both, while decode calls barely changed in the serial pair.

The prepared stop-only target is **superseded and must not be activated/submitted**.
The combination of the tested async CPU proxy, singleton-E5/FAISS16 batching
and search-only stop is active in the explicit new profile. The legacy
retrieval745198 remains running. Separate7day2GPU service750471 is READY on
ACD1-4 and completed actual64-query GPU document/score/hash outputs. The
combined target runs frozen runtime dde1452 and explicit750471
binding. Replacement750491 resumes from the sealed70 boundary; actual
recovery and five full-outer follow-up measurements are accepted. The old prepared
artifact/evidence is retained. Candidate-only pairs are separate from the
new whole-outer evidence and do not prove final-quality equivalence. See
[serial versus async actual pairs](evidence/search-action-stop-serial-vs-async-summary.json),
[superseded stop-only decision](evidence/search-stop-only-target-r1-superseded-decision.json)
and [actual CP70 transition](evidence/combined-cp70-transition-summary.json).

The candidate results below are historical diagnostics. Idle private services
750396 and750432 were released after all their consumers completed; checkpoint,
model, source and saved-result deletion was not requested. See
[actual release](evidence/private-diagnostic-services-release-summary.json).

The full native4 candidate A/B750423 completed0:0. For all2560 parents and
12800 trajectories, generation893.886s became841.931s (5.81% less time), and
input I/O + generation + scoring + saving1032.909s became980.517s (5.07% less).
New raw token work was0.287% larger; the average actual FAISS batch was1.485
rather than the configured upper limit16. This is a candidate-pipeline result,
not an entire training-outer ratio. The earlier512-parent startup comparison
reduced generation179.024s to165.799s (7.39%). Production has not been cut over
on the basis of this limited batching gain. See
[complete full-pool result](evidence/candidate-retrieval-full2560-final-summary.json).

Private16ms pair750433 completed0:0:512-parent generation174.703→159.506s
(8.70% less). Search-only stop pair750449 also completed0:0 on fourH100:
generation158.192→94.030s (40.56% less), pipeline184.048→120.846s
(34.34% less), raw tokens1,059,962→338,192. Both arms used the same async2ms
retrieval mode. All4 ranks executed FA3 and the enabled arm recorded5368 actual
search stops. These are candidate-chunk results, not entire training-outer or
quality acceptance. Formal750029 remains unchanged. The prepared stop-only target is superseded;
a combined async/batched continuation is being prepared. See
[actual four-GPU search-stop result](evidence/search-action-stop-live-native4-summary.json).

Both probes use hardlinked frozen HF policy66 and fixed parent chunks. They
regenerate trajectories; saved parents came from outer66 under pre-update
policy65. They load no actor optimizer and perform no training updates.
Production750029 and retriever745198 continue unchanged. Private2ms service
750396 uses immutable source0f48a49; full candidate probe750423 usesbf15a8c;
private16ms service/candidate use2303dd4.

Natural candidate-generation counters identify a concrete remaining serial path:
the local proxy and external retriever synchronously handle single-query calls.
In an85.95-second window,961 backend calls consumed39.796 seconds;473 of1434
logical calls were exact cache hits. This is46.3% serial backend occupancy in
that window, not a whole-outer critical-path fraction or GPU utilization.
Independent2GPU probe750360 onACD1-20 completed128 saved-query comparisons.
The conservative512-query probe750363 onACD1-4 also completed0:0: with E5
unchanged singleton calls, FAISS batch16 reduced the component median from
21.363 to4.650 seconds (4.59x); all1536 top3 slots match, max score error2.38e-7.
Neither called or modified production retriever745198 or training750029.
Full candidate-phase timing is now complete; production integration remains pending. See
[conservative actual512-query result](evidence/retrieval-batching-r2-final-summary.json),
[natural counter windows](evidence/leanmetrics-natural-retrieval-window-summary.json)
and [complete phase breakdown](evidence/leanmetrics-phase-breakdown61-65.json).

The current native4 async final path retains exactly51713 ordered occurrences;
it does not produce the legacy native8 fixed seven-row tail padding. Do not
delete seven rows from current final output. After real outer200, internal
`global_steps=201` enters final validation and returns; it is not an additional
training update. Final actual native4 restore-only is a separate required future
action, not something the archive controllers automatically execute. See
[current final entry bindings and remaining execution](evidence/native4-final-entry-readiness750029-20261023.json).

Previous job `749598`, immutable runtime
`c7976da3afc863a1c20b99daa7cc99de3bb61f15`, completed and sealed CP60 on four
H100 at `ACD1-14`: all four ranks reached Adam1200/scheduler60. Its complete
checkpoint is91,405,595,410 bytes, manifest SHA
`b1dbae539f4a1e1513293b930009bac19f84a83358c90d3d427958860ee31ed5`.
The completed54 native4 model/Adam/scheduler/RNG/data restore and complete55–59
performance history remain valid. Complete outer55–59 wall time averaged **29m02s**,
including checkpoint write/seal and inter-outer work. This is an observational
12.5% time reduction relative to old outer52, not a same-input causal comparison.
Actor and candidate FA3 execution, compact GQA, one weight sync for five candidate
phases, and zero extra monitor forwards were observed in actual worker reports.
See [five complete outers](evidence/native-prefix-five-complete-outers-summary.json)
and the [speed attribution report](SPEED-DIAGNOSIS-20261023.md).

The remaining gradient max/post-norm measurement pass is removed in immutable
runtime `2d5183455551145c52d1783a38663e2fcf170ceb`. Its independent
actual-native4 probe749913 completed0:0: all20 paired clips on all4 ranks had
bitwise-equal full gradients and preclip norms. Old/new20-clip bodies took about
5.01/2.05 seconds per rank; the approximately3-second saving is a component
measurement, not a whole-outer2.5x gain. Prior probe749812 failed in CUDA
initialization onACD1-13 before any clipping or Adam; evidence is retained.
After CP60 was sealed, the existing human conditional authorization of
2026-10-22 13:34:40UTC was exercised: old job749598 was cancelled at03:17:44CST;
at03:18:35 its terminal state was `CANCELLED by14090 / 0:0`. The existing CPU
full-state recovery read ran03:18:51–03:23:30 (about279 seconds) and passed
`EASY2560_NATIVE4_FULL_SHA_CPU_STATES_PASS`, receipt SHA
`6651c5ff154eb43fb644e3bc754dbf2536bf67a6e77360b9d09955b81de51cd5`.
This preserves CP60 and its Adam state; the continuation does not restart from
zero or reset Adam. CPU readback is not actual GPU restoration acceptance.

The unique continuation750029 was allocated four H100/48CPU/512G onACD1-14 at
2026-10-23 03:33:27CST, in `emergency_acd` for seven days. The03:35 observation
shows `RUNNING` and runtime identity matching
`2d5183455551145c52d1783a38663e2fcf170ceb`. The later actual native4 CP60 restore
passed: model collective readback and all four Adam/scheduler/RNG readbacks match,
rank0 checked339 model tensors individually, and four distinct CUDA UUIDs were
observed. Each rank restored Adam1200/scheduler60; driver and reconstructed stream
both read back cursor153600 with data equality. All five subsequent outers61–65
completed and sealed, and actual generated data from66 is saved. Restore and
follow-up acceptance do not prove final quality or a causal speed ratio.
Retriever745198 remains running. See
[actual CP60 native4 restoration](evidence/leanmetrics-actual-cp60-native4-restore-summary.json).

The complete2d-runtime outers61–65 averaged1789.972171 seconds (**29m49.97s**),
measured between actual before-generation boundaries. Candidate generation
averaged1032.846189s (17m12.85s,57.7% of the whole), actor152.3142s
(2m32.31s,8.51%). These five outers are about48 seconds slower on average than
the historical55–59 mean29m02s; policy and data differ, so this is not a causal
comparison of the final approximately3-second clip patch.

Actual candidate/actor FA3 with Q28/KV4, five-phase sync `[1,0,0,0,0]`, zero
extra monitor forwards and latest-write post-clip norm `null/measured=false`
were confirmed in every completed outer. CP65's sealed23 checkpoint plus7
boundary files total91,405,597,196 bytes, manifest SHA
`1bb578ccd19217265e6046656920a30dac972d668b5f7bc5574e02a025a5501c`.
Mixed selection counts were464/472/467/469/487 out of512, weighted92.1484%.
Training EM logs were56.3/55.3/57.1/56.7/54.3%, rewards0.732/0.731/0.747/0.751/0.712;
these are training data, not fixed-validation results. No whole-outer multiplier
is established by the clip patch. See
[five complete post-restore outers](evidence/leanmetrics-five-complete-outers-summary.json)
and the historical [first complete outer61](evidence/leanmetrics-first-complete-outer61-summary.json).
The independently bound archive pair for750029 milestones100/150/200 has CPU
core640634/companion640635 live at the same recorded birth identities, with
polling advancing and no errors. See
[checkpoint transition progress](evidence/leanmetrics-transition-progress.json).

Node-health probe750051 independently failed1:0 after8 seconds onACD1-13:
all four ranks saw `cudaDeviceCount=3` despite `CUDA_VISIBLE_DEVICES=0,1,2,3`;
no model, forward or NCCL step ran. Formal750029 keeps its node exclusion.
At the earlier scheduling observation, no healthy node supplied four idle GPUs
(the maximum observed was three), so lowering CPU/RAM requests would not resolve
that measured GPU shortage.
See [node13 health result](evidence/node13-health-final-summary.json).

The last patch saves only about3 seconds per outer at the measured component
level. That did not justify surrendering the existing allocation and queuing
again: **the timing of this transition was a judgment error**. The complete
training state is retained, but the queue delay is real and is not offset by
calling the clip component2.5x faster.

The remaining sections below retain the original Oct21 preparation history;
their descriptions of optimizations as unimplemented are historical. The
200/no201, fixed-validation milestones, native51713, seven-source metrics, full archive and actual final
native4 recovery requirements are still pending.

This is a fresh 7B actor experiment, adapted from the 3B Easy implementation at
6956d9ffa0ec497c13a66061bd6460a471e9aa0d. The original 3B checkout is preserved. Original job734689 was subsequently
stopped at the user’s explicit request, as recorded below. No other thread is messaged or modified.

Official source: Qwen/Qwen2.5-7B-Instruct, revision
a09a35458c702b33eeacc393d103063234e8bc28. The complete existing HPC3 asset is
`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/assets/qwen25-7b-instruct-a09a35458c70`.
Its 11 files / four BF16 shards total 15,242,788,168 bytes. The exact pinned
manifest is `reproduction/easy2560_joint4/official7b.json`. No learned selection
head, teacher labels or actor checkpoint is inherited from the source thread.

## Same scientific recipe

- 2560 candidate questions, five complete original trajectories per question.
- Real EM chooses K=1..4 first, then K=5 in original order to 512. Never choose K=0.
- Reuse the selected 512x5 tokens and actual sampling log-probabilities; no second rollout.
- Same grounding-augmented GRPO, virtual0=false; no question-selection probe.
- Same 4096 total context, max four searches, 500 action/observation limit and
  500 final-answer reserve. Same temperature1/top_p1 and original retrieval.
- Four H100, 200 outer, global mini128, physical micro8/rank, logical micro4,
  accumulation4, 20 optimizer opportunities per outer. LR1e-6, fixed57-outer
  warmup, PPO epoch1, clip.2, dual clip3, grad clip1, entropy/ref-KL coefficients0.
- Same BF16 compute and FP32 master/optimizer, FSDP offload, TP1. Preserve the
  original vLLM cache ratio0.1. In this vendored worker it allocates a fraction of
  *free memory after profiling*, not a total GPU cap including model weights.

7B-specific work is identity / architecture / shard binding. The grounding
calibration recipe is unchanged: train-derived fixed300, G5, layers21/24/27,
correction feedback, two-fold grouped CV. Round0 is genuinely regenerated and
fitted using official7B, rather than inheriting incompatible3B labels/heads.
Rounds50/100/150 retain the same actor-dependent refits. All these layers exist
in the28-layer7B; features and classifier input dimensions are determined natively.

Full native4 model/Adam/scheduler/RNG/driver state is saved after every completed
outer; publish the new complete boundary before retiring the preceding ordinary
checkpoint. Preserve milestones5/50/100/150/200, fixed512 QA and final native51713
per-source/macro/weighted EM/F1, and perform actual full-state CUDA restore.
There must be no actor update201. Raw outputs/checkpoints stay remote.

## Launch and current limitations

Dedicated base: `/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021`.
The existing once-only `control submit --base BASE --commit FULL40 --actor-micro 8`
uses jxu957/emergency_acd, four GPUs,48CPUs,512GiB. Code is published and deployed
before submission. A job id or RUNNING is not an accepted training result.

The retained personal storage budget is decimal2TB. The last full Oct20
cross-account visible measurement was2,354,561,650,688 bytes (not a fresh quota
reading). New training writes cannot be claimed budget-compliant while this
remains unresolved. A new cleanup question is pending; never delete other active
jobs' assets or silently raise the budget. Preparation and publication continue.

Local focused tests cover real Hydra composition in the existing HPC runtime,
model/reference binding, native4 geometry, real7B round0 routing and pinned shards.
The inherited common-contract fixture test `test_reader_preserves_eight_logical_micro_mass_for_physical32`
already fails identically in the unchanged3B checkout because it omits tensor
hashes. This is retained as a pre-existing test issue, not relaxed in runtime.

## Current user-directed stop and infrastructure candidate

At2026-10-21 17:55 CST the user explicitly stopped original3B734689 at completed145
(durableCP100). Slurm cancellation/extern release were verified. This supersedes
its previous200-outer objective and does not stop new7B745022. New7B745022 actually
started onACD1-22 at17:57:39, fourH100, runtimed6690e86152cbd6f9400317646de04d9f36ac2e1.
Source identity and real9 CPU configuration/calibration-routing tests passed;
retrieval staging79GB subsequently completed. A completed actor outer remains
pending. This is initialization, not a finished or accepted training run.

The approved oldQProbe cleanup removed exactly32 old warmup/initial-head tensor
files and independently verified their absence. Last-link allocated release was
16,865,304,576 bytes, not the29.98GB sum that double-counted hardlinks. Existing
raw rollouts, metadata and archives were not targets. Full per-UID root usage
scans remain pending, so this does not claim the2TB budget is already satisfied.

The user's subsequent infrastructure request has a separate opt-in candidate:
`infra_optimizations={selected_policy_logits:true,exact_retrieval_cache:true}`.
Missing flags retain the original path. Current immutable job745022 has no such
flags and is not hot-edited. `easy7b_infra.flags` restricts activation to the
pinned7B native4 Easy scope. Scientific recipe, generation and sampling LP remain
unchanged. No larger physical micro has yet been measured or selected.

- Selected policy projection: install a temporary LM-head pre-hook before FSDP;
  normal outerFSDP forward and complete decoder context remain intact. Under
  packedSP1, map original authored-target loss_mask onto their prediction rows,
  project only those rows, perform original full-vocabulary softmax and scatter
  back to the original response layout. No new weights/checkpoint keys/kernel.
  Existing PPO sanitizes masked observations before ratios, so unused0 entries
  are never claimed as measured probabilities and never enter the objective.
  If entropy is requested or no loss_mask is available, retain all response
  projection rows. Three actual tinyQwen CPU tests in a GPU0 substep of745022
  passed (selected log-prob/PPO/all parameter gradients, gradient-checkpointing,
  error cleanup),75.9s wall/72.3s test execution. The first login child timed out
  at40s and was retained; it is not a model/training failure. Packed-offset and
  empty-mask fallback checks passed locally. Full7B CUDA/FSDP equivalence,
  memory/throughput and larger-micro qualification remain unproven.
- Exact retrieval reuse: bounded4096-row LRU on the original FAISS initialization
  thread. Cache key includes exact query/ordered originalbatch/row/topk/scoremode
  and encoder/index/corpus hashes. A partial miss dispatches the ENTIRE original
  batch unchanged. Deep-copy values preserve caller mutability; errors are not
  cached, and no micro-batching/re-ranking/new FAISS thread is introduced. Logical
  search counts and outputs stay unchanged; health exposes independent cache
  counters.11 focusedCPU cases passed. Actual hit-rate/speedup is not measured.

Already-existing optimizations are not new savings: sampled oldLP reuse, zero-KL
reference forward skip and zero-entropy skip. Same-actor five-phase weight-sync
reuse, unused reference initialization removal and minimal grounding-head request
routing are further measured-work candidates; they are not silently enabled.

## Launchable optimization candidate and measured progress update

A future independent run can explicitly select `control submit --infra-profile
sparse-exact-v1`. The single profile is pinned through submit claim, shell and
prepared run manifest; baseline remains default and its launch argv is unchanged.
This is code preparation, not activation on745022. It does not raise micro8.

`search_g1/benchmark_selected_qwen_head.py` measures only a randomly initialized
7B-shape LM head in RAM (3584 hidden,152064 vocabulary). Both modes use the same
full vocabulary, positions, labels and signed objective. It reports CUDA forward/
backward time, synchronized wall, allocation/reservation peaks and full LP/loss/
all-gradient differences. Dense OOM is reported independently without reducing
size or fabricating a speed ratio. No GPU benchmark has been run yet, and this
script cannot establish complete7B FSDP throughput or safe micro16.

At18:16CST the actual7B calibration had fitted both heads after1500 rollouts,
1180 valid; its feature NPZ was48,956,084 bytes. Commit CV AUC0.892 and boundary
CV AUC0.691 are calibration measurements, not final QA performance. At18:17:32
the pipeline entered GRPO_1_TO_200 and began actor initialization. No completed
actor outer was present in the latest receipt. Two measured checks showed
generation completion, feature publication and final fitting rather than merely
a Slurm RUNNING flag.

The completed Oct21 root scan exposed at least5,095,177,170,944 allocated bytes
under the shared experiments root as jxu957, despite permission gaps. The
secondUID scan exposed4,991,155,515,392; these overlap and must not be summed.
The priorOct20 2.354TB observation is historical. The32-file approved cleanup
released16.865GB but does not resolve the global2TB cap. Other directories and
raw rollout/feature contents are not authorized by this checkpoint cleanup. A
low-priority per-directory distribution audit is prepared; no broad deletion.

Further reviewed synchronization opportunity: preserve five original RNG/KV
phases but retain the same frozen vLLM policy weights between candidate chunks,
reducing each rank’s sync calls5→1 (20→4 across four ranks). This is80% fewer
weight syncs, not80% faster training. Current offload replaces GPU weight storage
with CPU placeholders, so simply skipping sync would be incorrect. This change
is not implemented or enabled. Removing idle reference initialization and
requesting only reward-required grounding features remain additional candidates.


### Search-only action stopping: experimental, production unchanged

Saved-token CPU replay completed15,832 actual generation calls. In10,722 search
calls the new optional checker could stop at the original retained boundary;
591,019 retained logprobs and all compared token/text/boundary content were exact,
with no mismatches or missing eligible stops. It could avoid1,455,969 of2,128,927
raw tokens (68.39%) on these saved streams. No model or CUDA was loaded, and
this does not establish live RNG equality or a speed ratio. Actual step750432.0
completed0:0. The optional checker is installed only by the independent benchmark
flag; production750029 has not been changed. The old parser prioritizes search
over any earlier answer, so answer early stopping is deliberately excluded.

Independent4GPU pair750449 has been submitted using source7bbc459. Both512-parent
cases use the same asynchronous2ms private retrieval route and a fresh cold LRU;
only the search-stop flag differs. Dependency afterany750423 prevents overlapping
its shared private retriever with the existing full candidate A/B. Actual GPU
performance remains pending. See
[real saved-token replay](evidence/search-action-stop-saved-cpu-replay-summary.json)
and [raw versus retained accounting](evidence/candidate-raw-retained-token-accounting.json).

[Combined target and actual GPU service readiness](evidence/combined-batch-search-stop-prepared-summary.json).

# Easy2560 + Co rollout: official Qwen2.5-7B-Instruct

**Current evidence, 2026-10-23:** continuation `750029` completed and sealed
all five consecutive outers61–65; outer66 has actually generated and saved2560
questions. All four ranks performed100 new Adam writes, reaching1300/scheduler65,
with zero extra monitor forwards. The required post-restore five-outer follow-up
is accepted. The overall200-outer experiment and final acceptance remain incomplete.
See [five complete post-restore outers](evidence/leanmetrics-five-complete-outers-summary.json).

Natural candidate-generation counters identify a concrete remaining serial path:
the local proxy and external retriever synchronously handle single-query calls.
In an85.95-second window,961 backend calls consumed39.796 seconds;473 of1434
logical calls were exact cache hits. This is46.3% serial backend occupancy in
that window, not a whole-outer critical-path fraction or GPU utilization.
Independent2GPU probe750360 onACD1-20 now replays the same128 saved queries at
batch1/2/4/8/16/32, including fixed-singleton-embedding FAISS-only comparisons.
It does not call or modify production retriever745198 or training750029.
Numerical and speed acceptance are pending. See
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

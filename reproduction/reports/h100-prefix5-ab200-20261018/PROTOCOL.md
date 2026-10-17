# H100 Prefix5 A/B actor200 — approved execution protocol

This run implements the activated thread goal, after the preparation-only reference audit. The common baseline is staged `7679bb8ced2c13922fa9f5c9050ed0ce7fa54dde` and its published preparation/audit `8bd6a7661cc904b8f4e938df80807fed2c3f9420`. No old actor or online optimizer is imported. Both actors start from the same official Qwen2.5-3B-Instruct revision; selection heads start from exactly the same retained warmup epoch5 EMA model-only export (SHA `06ffd8c21b6208ff5a0beb11cded62da7b9551c3e1357a1d7321e69eb06d994e`). No new warmup.

## Preserved common science

Each arm has four physical ranks, colocated exact-index FAISS/E5 and grounding services, independent UUID leases, run directories, ports, pool/optimizer/RNG states. A uses0–3; B uses4–7. Foreign tasks are preserved. No new Slurm/GRES or global policy changes.

Current sampling actor question+5 policy token index27 postblock hidden is used, preserving actual IDs/masks/positions. The sixth sampled token is discarded after processing token5. The copied native block35 and flattened Linear remain unchanged. Rank score is h=1-product(p)-product(1-p); deterministic ties and original prefix continuation remain unchanged. No independent encoder forward.

Candidates: outer1–5=2560; ordinary6+=5120; periodic10/20/...190=2560 complete G5 with original actual-mixed-first/probe-score-fill actor selection. Outer200 is ordinary. Actor always512G5 with original Search-G1 reward/PPO, 20 update opportunities/rank, original virtual8/micro2 equal-mean objective on four physical ranks, global mini128, LR1e-6/57-complete-outer warmup, original finite-state safeguards and actual-write accounting. No data augmentation/new reward/loss changes to actor.

Boundary/commit grounding probes retain qualified round0 weights and actual50/100/150 refits/adoption. They are independent from the selection heads. Unified new BF16 interaction/QA protocol and fixed512 QA at5/50/100/150/200 plus final51713 remain unchanged. No diagnostic question enters candidates.

## A — frozen warmup

No selection-probe optimizer, parameter update or online EMA. Original periodic candidate completion/selection and diagnosis still run. Probe actual Adam stays0. Actor and grounding updates remain enabled.

## B — current-outer balanced raw training, EMA selection

Raw and EMA start equal. Only raw receives Adam updates at constant5e-6. After each actual Adam write, all floating EMA state follows0.99*EMA+0.01*raw; integer state is copied. Nonpersistent native buffers are retained. EMA selects and its pre/post evaluation is primary; raw paired metrics are also recorded.

Ordinary data are this outer's selected512G5; periodic data are this outer's entire2560G5. No historical data or extra rollout. Group labels are mixed iff0<K<5, includingK1. Resample whole G5 within current classes with replacement to256+256 ordinary groups or1280+1280 periodic groups. One pass over this sequence gives10 or50 actual synchronized writes; mean global batch256 (255/260), original true-global DDP weighting. If a class is absent, record a skip, preserve head/EMA and keep actor progressing. Final200 gets ordinary probe training. Maximum probe/EMA writes per rank=2760, reduced only by recorded missing-class skips.

Loss = mean own-trajectory BCEWithLogits +5*mean tied mixed-event BCE. Same32-pattern log-domain computation; no clamp/detach, no independent head, DIST, mixup, auxiliary KL or new soft labels. Threshold remains0.5.

Full unresampled current data are evaluated with raw/EMA before and after fitting. Periodic fixed512 questions receive fresh G5/hidden/labels from the same actor; paired metrics share those trajectories and never train. Record ordinary/balanced accuracy, both recalls and loss terms. Balanced-sample metrics do not substitute for original-distribution metrics. Uncontinued ordinary candidates have no claimed accuracy.

## State and verification

Every joint commits actor/masterAdam/scheduler/RNG/pool and selection head state. B also saves EMA/Adam/actual EMA clock and complete sampling indices, multiplicities and RNG. Sampling metadata have separate hashes; large state stays remote. Normal5/50/100/150/200 state readback uses CPU-only `ab_checkpoint_once`; no GPU gate or repeated completed large reads. Same-run recovery preserves latest genuine complete joint and actual clocks, never rolls back for an old hardcoded helper. Original code remains immutable once started.

Implementation: `ab_policy.py`, `ab_probe.py`, `ab_lifecycle.py`, `ab_checkpoint_once.py`, `ab_control.py`; explicit opt-in integration in probe/launch/storage and same-run recovery readers. Original reference mode remains available unchanged unless AB config is explicitly present.

Focused CPU validation: 37 tests passed for resampling/wholeG5/DDP equivalence, EMA state and integer/nonpersistent buffers, weighted mixed loss/K1 target, actual skip clocks, A/B joint commits including outer200, original actor recipe, candidate selection, grounding lifecycle, interaction protocol and recovery. Native GPU progression remains an execution acceptance item, not a CPU-test claim.

## Actual zero-update startup repair

Both initial native attempts imported the exact warmup head correctly, then exited before any candidate/actor/probe write because HF `load_dataset` refused a new Arrow build on a volume with ordinary available bytes0. The existing retained corpus cache has the same archive source/size/mtime and extracted contentSHA43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d. Reuse its already complete Arrow dataset; never bypass the space check or regenerate the corpus.

Actual h100-8 local physical free was255870287872B (ordinary0). Existing `10.33.0.18:/mnt/data` NFS mounted at `/mnt/data/sihe-share5/workspace/training_data_node_10` had2074849837056B ordinary available. New AB outputs use its existing `shaoshitong` directory, under the same isolated replica basename. Compute, sampling, E5 and exact FAISS stay on assigned h100-8 cards; this is storage IO, no cross-node inference RPC. No mount/reserved policy/foreign process changes. Four path-admission guards admit this single explicit remote output parent. Code is copied remotely across devices with full targetSHA; initialized raw/EMA state is copied once across devices with streaming sourceSHA plus full targetSHA, retaining inode aliases on target. No model passes through Mac.

The zero-outer retry controller verifies native exit/owned cleanup and original inode logSHAs, proves no actor/pool state exists and probeAdam0, reads the retained corpus cache, then carries the exact initialized head (no re-warmup, optimizer reset or lost update). Original failed directories and logs stay intact. Failed starts are not completed iterations.

The cache repair completed actual retrieval initialization for both arms (four GPU shards, dimension768, exactly21015324 documents). First-prefix workers then failed during vLLM initialization with actual `# GPU blocks: 0` and `No available memory for the cache blocks`; no prefix completion, sampled-token progress, actor write or probe update occurred. Installed vLLM0.6.3 profiles its own peak as `init_free-current_free`; the actual32768-token/1024-sequence profile exhausted the per-process free-minus-reserve budget. This is a concrete KV initialization capacity failure, not an inferred loss anomaly or a claim of CUDA OOM.

The retry retains the exact pending outer1 reservation (same2560 indices/order/pool after-state) and all warmup head bytes. Only inference concurrency limits fall to8192 batched tokens/256 sequences on both arms. Temperature, top-p/top-k, per-trajectory token budgets, Prefix5, G5, candidate counts, actor512G5 and all optimization settings remain unchanged. No training algorithm or installed vLLM code is edited. Old native/Ray-free process evidence and original inode logs remain. No active or foreign process is killed by this repair controller.

## Actual actor-stage colocation repair

Both arms produced all12800 first-round prefixes and all2560 selected complete trajectories. Actor forward then hit typed CUDA OOM at globalmicro16, followed by native retry at8; A additionally reported `CUBLAS_STATUS_ALLOC_FAILED` as a distinct non-OOM RuntimeError (not relabeled a numeric skip). The logs show approximately49GiB foreign usage plus9.6GiB own FAISS/E5 and9GiB own grounding residency per card. Neither arm committed an actor outer; failed-attempt returned writes are retained individually and never stitched.

The hardware-only lifecycle now releases only its own retriever after sampling/diagnostics and before actor attempts, retaining all grounding reward services. It restores the exact same retriever before subsequent sampling, every QA call and final51713. No foreign signals, reward changes, index/encoder changes or alternate online RPC. The actual OOM justifies globalmicro8/local2/accum16; globalmini128, reference micro2 equal-mean objective and20 retained actor updates remain unchanged. Existing39 focused CPU tests pass, including the original objective/gradient equivalence.

The startup recovery keeps the exact pending first pool and authentic prefix/score/full phase files, initial head/EMA and source labels by same-filesystem hardlinks, after original writer exit and original-inode log SHA evidence. Actor restarts at its actual latest complete boundary0 with original official weights; all failed unsaved writes are separately accounted for. No teacher generation or duplicate first-round rollout.

Frozen-arm metrics are retained without fitting: training metrics reuse authentic GPU scoring logits on original labeled groups only; periodic diagnostics perform normal frozen-head scoring on the already generated same-actor512G5. The paired frozen before/after values are identical. This adds neither an optimizer nor an extra rollout, and uncontinued ordinary candidates have no claimed accuracy.

## User-authorized move to h100-11

The user explicitly directed moving to the node with the most actual free VRAM,
then authorized reducing only per-card micro-batch and increasing accumulation.
Both h100-8 leaders were pidfd-signalled at 2026-10-17T20:32:05Z; their cleanup
receipts prove all owned subprocesses gone and no committed first outer. Partial
Adam writes remain in original journals but are discarded, never stitched.
The exact completed first-round prefix/score/continuation, initial heads and
candidate reservation are reused. The destination keeps two four-card slots,
uses the target node's actual UUIDs, local micro1/global micro4/accumulation32,
and preserves global mini128, objective and20 update opportunities.

Existing NFS mounts expose all saved trajectories and source assets on h100-11.
A task-private mount namespace preserves original absolute asset paths and
immutable receipt hashes. No global mount, foreign process or existing source
is modified. Only the exact Python environment, official model, E5 and GPU
retrieval libraries missing from the target are copied remotely to shared
storage; no large artifact passes through Mac. Submission remains the existing
once-only native shared launcher: h100-11's eight GPUs/CPUs are already Slurm
allocated to another task, so this is not falsely reported as a new Slurm GPU
allocation. The user's existing same-card shared execution authorization applies.

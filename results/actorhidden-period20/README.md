# Current actor hidden + periodic probe, fresh200

User goal updated1792163508 supersedes the stopped3420 experiment. This new run starts the official actor from0 and imports only the best retained warmupCP100 probe weights (SHA8df6bd445945ffbbf6a94171f24934495f0ce3ff17a3d2577bbd53d4b8fefd56). Probe Adam starts0. Old run/source tensors remain unchanged.

## Native representation

Capture the actual sampling actor layer27 post-block residual sum with real prepared request IDs, token IDs and positions. Native Qwen returns (MLP,residual), so copy their sum before the following fused RMSNorm mutates residual. The copied head keeps the293x2048 question273/policy20 layout and actual masks/positions. No independent encoder model is loaded or forwarded for scoring. Head construction uses the official architecture; its learned weights come from the probe checkpoint.

The user explicitly approved sampling21 and keeping20. A paused native request processes token20 on its21st forward; discard the sampled lookahead token. Freeze an earlier closing-action boundary similarly and flush its final retained token. Original prefix IDs/oldLP stay in PPO; no sampled lookahead is used by the classifier, environment action, or resumed prefix. Capture engines use eager execution/no prefix-cache hits to obtain every question hidden; full continuation keeps normal CUDA graphs/cache. This has a real profiling cost and no speedup is claimed before actual iteration timings. All captured BF16 tensors are streamed to remote CPU-backed uint16 mmap shards and consumed directly by the head.

## Clocks and labels

Each outer samples5120 distinct diagnostic-excluded questionsG5, scores Top512 with the pre-update probe, continues the same prefixes and performs20 actor Adam writes per rank. Actor/reward/PPO budget and legacy FP32 QA remain unchanged. Probe is frozen on other rounds; only20,40,...200 fit all5120 original candidate groups once,100 Adam per rank at1e-6, ending1000 probe Adam per rank. The original owntrajectoryBCE + ownG5mixedBCE loss weights1/1 remain, with mixed0<K<5 includingK1. WholeG5 stays on one rank/step;80 steps255 trajectories and20 steps260 average256 and visit all25600 once.

At a periodic outer, fix Top512 before any full labels, complete the full5120G5 pool once, and project the exact preselected2560 rows/oldLP for actor PPO. No selected trajectory is generated twice. Heldout512 question identities stay fixed, but G5 outcomes/native hidden are fresh from the exact same actor snapshot as the training pool. Report paired pre/post train and heldout mixed accuracy/BA/recalls/confusion/loss, with class counts; these are not fixed QA EM/F1.

## Evidence and operations

Targeted CPU tests cover native residual aliasing, authentic query ordering, off-by-one20th token, future-token masking, whole-pool natural schedules and exact no-generation subset projection. Actual native GPU capture, correct actor writes and iteration timings remain unverified until the new production job advances. Normal checkpoint readback uses checkpoint_native_once.py; it handles actor20N and probe100*floor(N/20), including the zero-optimizer probe before outer20. No standalone GPU gate or extra baseline is added.

Model/Adam/hidden/trajectories/full predictions stay remote. Rotate only consumed current-run hidden buffers and unreferenced nonmilestone actor states after joint commit; retain original questions/tokens/labels and checkpoint metadata. No old data is deleted. Native exit0,200/no201, scheduled512QA and final51713/count, all normal checkpoints/final state and resource cleanup remain required; this README is not completion evidence.

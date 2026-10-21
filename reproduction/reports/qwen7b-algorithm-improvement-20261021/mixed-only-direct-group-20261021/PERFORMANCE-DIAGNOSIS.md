# Cached feature training performance diagnosis — 2026-10-22

The original production job746083 uses B32 questions/320 native rows/128 balanced G5 views, but micro_questions=1 creates32 serial native/head forward-backward calls. Its storage reader is synchronous inside that loop. No asynchronous producer/consumer feature queue was deployed. The actual head is two attention layers/d256 in addition to the233M-parameter native Qwen block.

An owned, read-only twelve-sample GPU observation found utilization0–22%, mean10.5%, memory8285MiB; this is a short observed interval, not a whole-run average. The prior accepted-update window was7.94s/update. The primary mistake was failure to profile/tune the data pipeline and GPU microbatch before declaring the initial slow rate reasonable.

Independent H100 job746155 completed0:0. With320 BF16 rows of the exact native293×3584 geometry already onGPU, forward/backward plus gradient checks measured median0.6376s for micro1 and0.3322s for micro32, peak allocation35.095GB. This is a synthetic compute benchmark, excluding IO/Adam/EMA; it is not production-step speed. Different BF16 batching gradients differ numerically, relativeL2 .00217 atmicro32, with identical objective/loss.

For consecutive320 authentic retained rows, old reader warm0.9518s versus optimized reader0.3842s. All feature bits/masks/positions are identical, and retained-row SHA remains mandatory. Consecutive rows isolate decoder overhead and do not represent random shared-storage latency. A second independent job746186 restores the accepted CP128 and compares full actual random-next-batch steps, including IO/Adam/EMA.

The new pipeline prepares up to8 future B32 batches (~5.4GB BF16 feature payload) with8 CPU producer threads while GPU consumes the current batch. Its lookahead is a disposable sampler copy limited to the current epoch; real sampler/RNG/accepted counters are unchanged. Actual next-batch tokens must match. It uses bounded per-worker caches and ordered bulk H2D transfers. CPU fixtures cover identical row tensors, own deltas, padding, duplicate requests, corruption detection, parallel ordering/RNG, and epoch-boundary lookahead identity.

Continuation must use a full highest-current checkpoint after graceful handoff; no CP128 rollback is allowed. Models, EMA, two Adam states, RNG, sampler/data position and total21132-update budget are preserved. No healthy main source is hot modified. The new immutable worker/config must explicitly distinguish the inherited all-payload seal from fresh metadata/each-used-row SHA; no repeat-fresh276GB SHA is claimed.

## Actual asynchronous production acceptance

Old746083 gracefully completed0:0 at highest acceptedCP373. New746220 restored exactly all nine component hashes into its immutable fast namespace; no state reset or prefix rollback. Initial fast update, independent fullCP374 CPUreload, and durable-next-update replay all passed. Six genuine accepted fast updates were qualified, then two observations advanced413→474 (101 genuine new updates since373). No failure was present.

A checkpoint-free actual production window385→413 completed28 updates in26.69617s: end-to-end0.95343s/update, body mean0.69712s/update,2013.77questions/minute (20137.7 genuine native-row visits/minute). CPU future-queue waiting averaged0.00017763s/update, proving that later-row reading now overlaps current-GPU work. Startup/periodic checkpoint-save overhead is explicitly outside this normal-window figure. A twelve-sample GPU window measured mean58.5% (range0–100%) and41449MiB resident memory; this is a sampled interval, not a whole-run/whole-epoch utilization claim.

Training remains active toward21132 accepted global updates/four epochs and independent Mixed-discrimination/selection acceptance. Performance validation does not establish scientific improvement or complete the overall goal. Latest runtime source: original21 files unchanged plus root readerc2fdc959, worker/updater24c68933, controllerd89eb901; exact publication manifest7750B SHAdc34dcc787b3a52a5dbfe19d6d1fa39c9599ff158654bc4acdda56e71a8ace4a.

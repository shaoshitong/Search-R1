# B CP50 native eight-rank prefix-diversity fork

Latest human assignment: continue the reference B from its genuine full CP50;
change each selection question from five to ten actual prefixes, choose five
without probe outputs or labels, pass only those five to the probe. Use an
independent directory and eight GPUs, reducing gradient accumulation rather
than increasing per-device micro-batch. Total target remains outer200.

Reference: Codex thread 01a0fcbb-863f-7450-926d-db537f648265, immutable B run
`d9e5385ebec24ea8ab6cb8db8d2f825a`, original full CP50 joint SHA256
`92bd03998fef4a63bd1e181717f55acbc459cc6a598041c252ece587c4542db9`.
The reference process is never stopped or edited by this fork.

- Prefixes20/40/80 for outer51–100/101–150/151–200; ordinary5120 candidates,
  Top768/1024/1536 original G5 continuations, reference actual-mixed-first
  projection to512 actor groups; periodic pool and fixed diagnostic unchanged.
- Exact policy-token sequence deduplication first, deterministic farthest-first
  among distinct sequences using mean positional mismatch and bigram-Jaccard
  distance. Retain min(5, unique_count) unique prefixes; fill unavoidable
  duplicates only if fewer than5 distinct samples. Preserve original native
  IDs, seeds, tokens and log-probabilities; canonical five slots are0..4.
- This is heuristic diversity selection, not a guarantee of mixed labels or
  unbiased conditional-policy sampling. Existing B PPO loss is retained;
  no fabricated selection propensity or new probability correction is claimed.
- Ordinary prefix generation5120*10=51200; retained probe inputs5120*5=25600.
  The other25600 receive no classifier forward. Generation costs include all10.
- Mini128, each rank20 opportunities, per-device micro1; global micro4→8,
  gradient accumulation32→16. Original equal-virtual-micro2 reduction retained.
- Actor true CP50 Adam1000 and scheduler50 retained; exact contiguous halves of
  each proven unpadded FSDP flat moment become8 shards. Tied HF embedding/head
  aliases must be bitwise equal before counting the single optimizer parameter.
  No new Adam write occurs during conversion. Original4-rank write journals
  remain original4-rank history, not synthesized eight-rank historical writes.
- Probe raw/EMA, Adam700 and EMA700 retained. Same reference20-token expansion
  zero-initializes only new Linear coordinates and their moments. LR5e-6,
  EMA.99, class balancing and trajectory-BCE+5*mixed-BCE retained.
- Even child RNG inherits parent; odd child derives a deterministic independent
  stream. This is not an exact eight-executor replay. Pool order/cursor is CP50.
- Historical4-replica grounding adoption is verified as historical4. All8 live
  replicas must actually reload/read health/adopt the same existing CP50 fit.
- Original first startup runtime c3ced09047dd4ccd66e5d232f36b5952c178a7ce.
  A published external CPU metadata finalizer completes an optional missing
  source provenance record; original failed importer terminal is retained.
  It does not rewrite tensors or frozen training source.

All model/data/index/tensor outputs remain on H10018 (h100-10/10.33.0.18).
Only code and small audit summaries are stored locally. Follow real51–55 and
positive56 before startup acceptance; continue the same process to200.
Initial QA is inherited from unchanged CP50, not rerun. Actual new writes,
skips, elapsed time, mixed ratio, prefix diversity and five-probe probability
variance are read from genuine outputs. Acceptance is not final quality success.

Actual first GPU attempt restored all8 ranks but wrote0 Adam updates: the
monitor sampler still required two virtual16 blocks per physical rank. The
repaired runtime ff04ea81e4f3d7783d9c84c558faa515e5eca79f admits one block per
rank on world8 and preserves the identical ordered32 global monitor IDs
(independent CPU36/36 cases passed). It uses an explicit immutable checkpoint
binding compatibility proof to c3ced090; model/Adam blobs are not rewritten.
The zero-write outer51 native prefix/score/continuation artifacts are reused.
Original failure, cleanup, journals and GPU time remain separate evidence.
The repair has one native leader PID2994909/start1938322513 on H10018; the first
leader PID2974049 and all owned children are confirmed gone. No reference A/B
process, live source, threshold, sampled sequence or optimizer history changed.

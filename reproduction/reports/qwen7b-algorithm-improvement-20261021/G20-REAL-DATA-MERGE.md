# Real G20 source acceptance

`g20_real_merge.py` is an independent CPU data consumer. It does not change the
active G10 collector, submit a job, train a probe/actor, or choose virtual G5
subgroups. Default output is
`algorithm-improvement-20261021/training-G20-matched6528-20261021/materialized`.

All generic completion, extra-G5 completion and adapter-controller receipts from
all twelve offset5/10/15 shards must already exist before large-source reads or
output creation. Partial phases fail closed. Actor revision, original source,
protocol, prompt/token input, cohort and independent run/attempt identities must
agree. The merger reuses the qualified immutable extra-shard verifier and checks
its own bytes against the collector's bound source hash.

At final acceptance, original selected source input/receipt/full-rollout SHAs are
freshly checked once. The actual original and all three new full-response seed
journals are streamed; their seeds are rederived and must be mutually disjoint
per original question, including within an event. All 130,560 own strict-EM labels
are recomputed with the frozen official CPU tokenizer; no model tensor is loaded.
The final-question manifest is not read or hashed.

`bank/groups.json` contains `train` rows with exactly twenty unique authentic
`trajectory_ids` and twenty own binary `labels`, ordered original0..4, extra5..9,
extra10..14, extra15..19. It preserves the original fixed-five identities/labels.
`bank/feature-events.json` records four independent native G5 feature events;
`bank/features.json` is their flat native manifest-path list. No tensor packing is
synthesized or joined. New-event packed payloads have full fresh SHA verification;
original accepted packed-feature manifests and identity-index SHAs are checked,
with original bulk feature SHA explicitly marked not freshly repeated. Native
FeatureReader checks reconstructed row hashes on actual consumption.

`accepted_g20(reference)` is a lightweight consumer: SHA checks sealed source
receipts, rejoins original and all three new own-ID/EM banks, and returns
`(groups, native_manifest_paths, complete)`. It never selects a G5 using labels.
Training-only random/balanced virtual grouping is a separate module.

CPU fixtures cover twenty distinct real-event identities, own-label translation,
actual seed derivation and cross-event collisions, missing/corrupt full rows,
strict-EM mismatch, authentic feature-event membership/payload SHA, and the
all-twelve-shards completion gate. These are implementation tests, not a claim
that real G20 data or GPU consumer qualification has finished.

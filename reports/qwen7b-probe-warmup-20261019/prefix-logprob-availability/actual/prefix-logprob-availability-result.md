# Finite native actor Prefix20 chosen-logprob availability check

The existing accepted full journals contain the actual generation chosen-token behavior logprobs. The prefix-only JSON/HF feature projection does not contain them directly. They can be projected from the full journal's native chunks using the first 20 *retained policy* token positions, without a new actor forward, GPU job, new rollout or model update.

A bounded read-only check selected the first identity in the frozen original8192 plan (old3 + newseed3 authentic trajectories) and the first question's existing fixed G5 in the viewed validation2048 plan. Identity selection used neither labels nor model scores. Eleven trajectories / 220 Prefix20 policy token values passed:

- Native `behavior_turns[].actual_sampling_chunks[].selected_logprobs` matched the chosen token IDs and the turn-level `selected_logprobs` exactly.
- The first20 retained policy values matched `response_log_probs` at `response_info_mask == 1` exactly. Retrieval/correction positions are stored as zero but were excluded from the projected policy confidence.
- Tokens, positions and all causal prefix segments matched the existing sealed Prefix20, including its SHA. No future full-answer logprob or discarded native generation tail entered the projected20 values.
- All 220 projected values were finite, negative and nonzero (220/220); zero values were not being used as a placeholder in these selected policy positions.
- Actual sample settings recorded temperature1, top_p1, top_k−1, repetition penalty1, presence/frequency penalties0 and requested logprobs1. Native snapshots select the logprob belonging to the actual sampled token and commit its token/logprob pair together. Some contributing turns also have continuation chunks with larger max_tokens; none of their post-cutoff token values are included in Prefix20 confidence.

Actual completed sample counts advanced 3 → 6 → 11; the remote command exited0. Only these11 target records were analyzed. Their bound source batch files were SHA streamed and target rows located; this is not a full validation score/scan or an assertion that every trajectory's logprob coverage was newly revalidated here. The previously accepted source receipts and exact descriptors remain bound in the audit.

This establishes availability and causal alignment in the finite samples. It does not establish any correctness correlation, Mixed balanced accuracy, calibration or selection benefit. A sampled chosen-token logprob is an observable of that behavior policy; logprobs1 does not supply the full distribution needed for vocabulary entropy, and no entropy was invented. The aggregate `p_j` correctness estimate from the probe is a different quantity. Existing scoring/training recipes were not changed and no confirmation/final pool was read.

Complete: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix20-chosen-logprob-availability-v1/complete.json`, 20593 bytes, SHA256 `d38529092bb5fc653cefb2cc80181cd06c08fd9951ef36425aa09a924f5e1959`.

Source `790d31bf3e58c330b90c2a9b738d58ab3198d164`, branch `codex/prefix20-chosen-logprob-availability-20261021`, remote SHA verified; four CPU boundary tests passed (mask/observation zero exclusion, clock/value mismatch rejection, future suffix invariance, missing/nonfinite/positive value rejection).

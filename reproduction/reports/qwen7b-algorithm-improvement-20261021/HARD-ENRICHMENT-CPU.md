# Train-only hard enrichment CPU preparation

Implementation: `reproduction.operations.qwen7b_warmup.hard_enrichment`. This is a pure CPU plan preparer/selector, not a trainer, GPU scorer, or submission tool. No real data have been selected by the local tests.

The source is the canonical 169,028-question training bank. All of it was already exposed during initial warmup; these candidates are outside the current 6,528-question controlled-fit set, not claimed unseen in previous warmup or fine-tuning. Identity-only exclusions cover current fit, fixed 512 diagnostic, 2,048 selection validation, and frozen final identities. The module opens only the small exclusion audit, never its final-source provenance files.

The frozen E4 scorer contract is checked by a cross-module synthetic test: candidate rows contain exactly `qid`, `question_hash`, five `trajectory_ids`, five own binary integer `labels`, and `split: train`. Top-level `checkpoint` and `source_checkpoint` bind the same E4 descriptor. Candidate identity selection is hash-based and precedes joining own EM.

Remote configuration references use `{path, bytes, sha256}`. Required prepare keys are `source_bank`, `matched_plan`, `features`, `checkpoint`, `exclusion_refs`, and integer `seed`. Every exclusion audit has role `training_mining_exclusions`, identity-only `groups: [{qid, question_hash}]`, and actual `source_refs`; source refs' role fields or top-level `scopes` explicitly cover `matched_fit`, `diagnostic512`, `selection2048`, and `frozen_final`. The source bank's entire `train` is never treated as an exclusion set.

```sh
python -m reproduction.operations.qwen7b_warmup.hard_enrichment \
  --action prepare --config /data/.../config.json --output /data/user/jxu957/experiments/.../prepared
```

The output `candidates.json` freezes exactly 32,768 question identities and 163,840 own trajectories. After the independent GPU scorer completes, add SHA-bound `candidate_plan` and `scoring_complete` references to the config and run `--action select`. The selector verifies complete coverage, ordered source IDs, own EM, all five finite probabilities, their group-product formula, and the actual E4 scoring receipt before selecting anything.

Targeted supplement, 4,092 questions per stratum:

- Highest q among true all-wrong questions.
- Highest q among true all-correct questions.
- Lowest q among true Mixed questions.
- A hash-random anchor from the remaining true Mixed questions.

The random control uses hash selection within the same macro classes: 4,092 all-wrong, 4,092 all-correct, 8,184 Mixed. Each branch adds the same original 6,528 core questions, resulting in 22,896 unique questions and 477 one-pass B48 updates, with the same seed. The actual K=1/2/3/4 distribution is reported, not assumed matched. “High” and “low” are ranks, not an implicit 0.5 eligibility filter; actual false-positive/false-negative counts at 0.5 are separately reported.

Insufficient classes produce `quota-unreachable.json` with actual counts and the maximum common reachable quota. There is no invented label, repeated question, partial-pool selection, or validation fallback. `groups.json` contains the training branch only; a downstream trainer must reference the unchanged diagnostic fixture separately, not move it into training.

Verification: 9 hard-enrichment tests plus 8 scorer tests, 17/17 passed on CPU. Covered label-blind candidate selection, scorer schema compatibility, exact coverage/own IDs/EM/q, source binding, deterministic strata, equal macro-class budgets, score-independent random control, and explicit insufficient-quota failure. Model/feature/scoring files remain remote.

The parent published commit `dfb183ddf35a7116ef118d1d2e1e5b49dcbb21c7`; the CPU candidate preparation has now completed remotely, using an isolated source file SHA `7b13a4cb6ac126a9c3c972ab66c8458be2af3c6bf9191df132707c11f4415ca1`. Two independent readbacks 41.17 seconds apart verified candidate SHA, source bank/features, all own IDs/EM, unique counts, and unchanged output mtimes. The durable complete receipt exists and the process has exited with no failure receipt and a free lock; its process exit code was not captured, so completion rests on the full artifact/readback acceptance rather than a claimed exit code.

Remote output root: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hard-enrichment-20261021`.

- `candidates.json`: 15,436,509 bytes; SHA `b0118d67ed31187ed0e380cd81d62649852a2806fc9f5d8f16af381f8c4b2832`.
- `prepare-complete.json`: 3,991 bytes; SHA `b07eaa03189e15e401cfaddb1deaa9a242a9bf4ef775ed17bbea86f81d072770`.
- `identity-exclusions.json`: 1,660,462 bytes; SHA `fb114f35d59407bcd87297ba97493cc61381d9922c846153a3668c219bb9d7f5`; 13,184 disjoint identities across four scopes.

The eligible canonical pool after exclusions contains 162,500 questions; six repeated question hashes are pruned label-blind, leaving 162,494 distinct contents before selecting the frozen 32,768. All 1,336 native feature manifests actually satisfy L26/Prefix20/Q273/D3584. The balanced bank has no independent `features.json`: it reuses the original bank's native Prefix20 manifest list, whose real SHA is bound in the receipt. Canonical and balanced banks' 169,028 own qid/hash/ordered-five-IDs/EM were verified equal.

Actual candidate K=0..5 histogram is `[15230, 3403, 2889, 2938, 3652, 4656]`. This makes the requested four-stratum quota 4,092 reachable (maximum common quota 4,656); no class-driven candidate resampling occurred. Frozen E4 GPU scoring and score-driven targeted/random branch selection remain pending. Compact acceptance and immutable worker/controller backups are in `actual-launches/hard-enrichment-FINAL-ACCEPTANCE.json` and adjacent code/observation files; no raw candidate or feature rows were downloaded.

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

Verification: 9 hard-enrichment tests plus 8 scorer tests, 17/17 passed on CPU. Covered label-blind candidate selection, scorer schema compatibility, exact coverage/own IDs/EM/q, source binding, deterministic strata, equal macro-class budgets, score-independent random control, and explicit insufficient-quota failure. Model/feature/scoring files remain remote. Real 32,768 scoring and actual class reachability are pending deployment by the parent task.

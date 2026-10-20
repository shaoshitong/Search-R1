# Fixed CPU paired-stability protocol for append512/768/938

Prepared after the G20 DP terminal audit and **before any append milestone comparison is run by this helper**. This is a descriptive analysis of the already-viewed, tuning-allowed recipe-validation pool; it cannot turn these data into an untouched final evaluation. The helper does not fit, rank candidate recipes, select an epoch or recommend a threshold. The three previously specified append milestones remain fixed.

1. Only use the exact same 2,048 questions and own fixed-G5 IDs/strict EM for the candidate and both same-HF E2 and E4 baselines. Bind prediction file bytes/SHA and existing independent identity/backend audit receipts. Before reading any metrics, freeze the ID-only partition: sort by `(SHA256('9137' + U+001F + str(qid)), str(qid))`, then distribute round-robin into four equal 512-question strata. No labels/scores influence strata. Store the membership SHA and preserve it across candidates and baselines.
2. Stored probabilities must satisfy `q=1-prod(p_j)-prod(1-p_j)` within2e-6. Use threshold0.5 for every model. Use `(-q,str(qid))` for all selection. Full pool selects512 (25%); each 512-question stratum selects128 (25%). Do not call the stratum Top128 results Top512.
3. Report each model's full and four-stratum Mixed BA, Mixed/NonMixed recall and true-Mixed count/fraction in the selected quarter, then candidate-minus-E2 and candidate-minus-E4 differences. The original question is the sampling unit; neither individual trajectories nor G5 members are independently resampled.
4. Use **2,000** paired nonparametric question bootstrap replicates, with replacement. For each scope, derive its RNG from seed9137 plus the first8 hexadecimal digits of SHA256(scope_name). The same resampled indices feed all three models. Rerank and reselect the same fixed quarter on every replicate, retaining repeated draws as repeated observations. Report 2.5/97.5 percentile CIs for BA difference, selected-Mixed count difference and selected-Mixed fraction difference. Skip and count rare single-class replicates where BA is undefined; fail if none are valid. Do not add pseudocounts.
5. CIs are pointwise and descriptive, without multiplicity correction. Four overlapping inferential views plus three milestones and two baselines are not independent confirmatory tests. Small positive differences or a CI crossing zero are not a reason to add a new checkpoint, change a threshold, rerun training or inspect untouched final4096.

Implementation: `reproduction/operations/qwen7b_warmup/paired_question_stability.py`; five small CPU fixtures check identity-only equal strata, deterministic paired resampling, zero deltas for identical models, correct direction for a known perfect-vs-inverted example, fixed threshold/quarter, q formula and own-ID/EM rejection. No real prediction data have been read by this preparation.

## Manifest for later remote execution

The parent operator supplies actual, complete receipts only; no placeholder SHA is accepted:

```json
{
  "role": "already_viewed_recipe_selection_validation",
  "questions": 2048,
  "bootstrap_replicates": 2000,
  "independent_identity_audit": {"path": "<actual accepted audit>", "bytes": 0, "sha256": "<actual SHA>"},
  "same_HF_backend_receipt": {"path": "<actual shared-backend receipt>", "bytes": 0, "sha256": "<actual SHA>"},
  "predictions": {
    "candidate": {"path": "<actual milestone predictions>", "bytes": 0, "sha256": "<actual SHA>"},
    "E2": {"path": "<actual same-HF E2>", "bytes": 0, "sha256": "<actual SHA>"},
    "E4": {"path": "<actual same-HF E4>", "bytes": 0, "sha256": "<actual SHA>"}
  }
}
```

The shown zero sizes are schema examples, not valid asset references. Every referenced path must be under the remote experiment directory; raw probability rows stay there. The CLI binds and returns audit receipts but relies on the already-accepted independent identity/backend audits for provenance; it does not claim to re-prove feature extraction or own-EM tokenizer execution. It separately checks exact paired trajectory IDs/EM and score formula from all three files.

Run on HPC3 CPU only, after actual milestone audit acceptance:

```sh
python3 -m reproduction.operations.qwen7b_warmup.paired_question_stability --manifest /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/.../manifest.json --output /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/.../paired-stability.json
```

The worker pins at most two available CPUs, caps address space3GiB and CPU time180s, imports no torch/model/CUDA runtime, requests no GPU, and refuses to overwrite an existing output. Only the aggregate JSON may be copied locally. No Slurm submission or data analysis is performed by preparing these source files.

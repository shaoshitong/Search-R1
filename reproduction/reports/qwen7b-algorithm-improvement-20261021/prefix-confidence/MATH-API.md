# Prefix-confidence v2 pure mathematics: remote qualification

Exact source commit **9932cee9e04eebb4e617ed5ae5f938452ff36264** is published and remote verified. Six synthetic tensor fixtures passed in the existing HPC3 CPU runtime: unittest3.852s,13.648s including imports, peakRSS508420096B, CPUaffinity[0,1], torchthreads2. No CUDA, actor/probe, real prediction rows, prefix/full rollout, model weights or original optimizer were loaded/updated. Toy scalar checkpoints were serialized only in remote RAM.

The two new source files are:

- `reproduction/operations/qwen7b_warmup/prefix_confidence_calibration.py` (15988B), SHA`3244d6a0fb68793fb23913650874d4b24ace4b3721df1c4d82446771ea22a761`.
- `reproduction/operations/qwen7b_warmup/test_prefix_confidence_calibration.py` (6640B), SHA`5c91cca80ac065775b81daef4b068a4d4e6db1437c89624c10d84cf2eec4fe84`.

[Actual small remote test receipt](math-qualification-result.json), [immutable source/dependency binding](math-qualification-source.json).

## Fixed interfaces

`FEATURE_NAMES = ('mean_logp','std_logp','min_logp','first5_mean_logp','last5_mean_logp','actual_len_over20')`.

`question_split(qids, holdout=2048)` returns an order-aligned bool NumPy mask. Ranking key is `(SHA256('prefix-confidence-qid-9137:'+str(qid)), str(qid))`; first2048unique qids are heldout, remaining6144fit. No labels or scores enter this function. The operator first freezes this only-ID split and then checks it against the module.

`fit(logits, labels, features, fit_qids, source_binding, progress=None, checkpoint_callback=None, *, use_prefix=True, deadline_monotonic=None)` returns `(state,report)`. Real inputs are6144×10 logits/own strict EM, full-mode features6144×10×6, and the same ordered6144qids. Stored probabilities use `probability_logits` clipping1e-6 identically for fit/inference. Defaults are fixed40LBFGSiterations/60optimization-closure evaluations; accepted final objective is separately verified once. `progress(calls,total_loss)` may write a small curve. Source data availability must already be accepted, and binding must include:

```text
availability_qualified = True
availability_receipt = {path, bytes, sha256}  # actual checked extract-complete receipt
calibration_mode = 'prefix_full8' or 'matched_platt2'
```

The caller owns actual file/provenance checks. Module binding is not a substitute for extraction qualification. Smaller budgets/cohorts exist only for explicit synthetic `qualification_only=True` with matching source marker; they cannot silently change a real recipe.

Full mode learns eight scalars. `use_prefix=False` requires source mode`matched_platt2`, takes`features=None`, has no normalizer and keeps sixw entries exactly zero. It is the same-E4, same-split/objective/regularizer/budget matched control, never an alternative external candidate. Both fits receive **one shared** `deadline_monotonic` set before the entire two-mode fit phase; no per-mode26minute reset. The caller also enforces the global wall/RSS guard.

`normalizer_fit` uses only passed fit rows, population std across all6144×10 trajectories, and scale floor1e-6. `calibrated(z,features,raw,normalizer,use_prefix=True)` computes the declared bounded scale/bias and linear normalized-prefix correction. `objective_parts`/`chunk_objective` use trajectoryBCE1 + three-eventNLL1, all252 G5 subsets with total per-question weight1; regularizer is fixed.01. CPU float64 math is bounded in128-question chunks.

## Checkpoint and inference order

At accepted fit completion, `checkpoint_callback(state)` runs **before holdout assessment**. It must atomically save the full state, SHA/load/reload/restore-verify it, then return a receipt. State includes raw8, fullLBFGS, CPUtorch/NumPy/Python RNG, normalizer, fit-qid digest, all source bindings, protocol, mode/use_prefix, phase`ALL_FIT_LBFGS_UPDATES_COMPLETE_BEFORE_HOLDOUT_ASSESSMENT`, and explicit holdout/external=false. This new scalar optimizer does not reset the original probeAdam.

`restore(state)` checks schema/phase/source-mode binding, fullLBFGS cursor, normalizer identity, and matched2zero-w/no-normalizer. It returns raw/optimizer/normalizer/source/phase/mode/use_prefix. A synthetic same-next-update test passed for full state before/after serialization. This verifies state recoverability; it does not authorize another actual optimizer step beyond the fixed registered budget.

`score(z5, features5_or_None, state)` returns label-blind G5q; `summary(z5, labels5, features5_or_None, qids, state)` returns the existing lightweight metric schema. For matched2, passNone features. For internal assessment, use indices5…9 (newseed5), not a new sampled subset. Raw/unclipped and clipped-identity E4 baselines remain separately reportable.

Only full8 is eligible for external scoring: it must strictly improve internal rawE4 BA and Top512; against matched2, neither may decrease and at least one must strictly improve. The operator owns this gate, freezes the exact accepted checkpoint, and only then may open fixed2048 same-HF external payload once. No threshold/grid/refit/final-pool operation exists in this module.

On deadline/budget failure, the callback receives a complete **auxiliary partial** scalar/optimizer/RNG/source state, with explicit accepted-boundary versus unaccepted-LBFGS-trial phase. It remains no-holdout/no-external and is not called complete or safe-to-resume; normal`restore`rejects partial schema. The operator must retain it without pretending the fit finished.

## Fixture coverage

All252 objective value/eight-scalar gradient versus independent32-pattern enumeration; zero-init identity/fit-only stdpop normalizer/ID split; real toy8-scalar optimization/fullstate serialization/same nextupdate andRNG; input/availability/budget/feature rejection; clipped G5 inference; matched2 feature independence/zero weights/mode binding and timed-out auxiliary state preservation. These fixtures qualify code only, not learnability or a real-data result.

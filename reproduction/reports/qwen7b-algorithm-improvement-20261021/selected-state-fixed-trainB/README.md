# Selected-state common fixed trainB2048 diagnostics

Source-stage only. This entry has no training action, new rollout generation, checkpoint selection, threshold fitting or final4096 access. Root executes once after the second Z seed genuinely completes CPU acceptance. The deliberately invalid future descriptor cannot run.

The exact training diagnostic pool is the existing2048 ownG5 metadata with SHA `e6e74d92160efb0740ba5c0316c4ee7686f06a499c90cafd5a545f325b62209b`:888 Mixed,950 all-wrong,210 all-correct. Old trainA is retained separately and is never substituted. Saved score reuse validates every question, ordered own trajectory ID and actual strict-EM label, not just the class counts.

Required models are the original E1 EMA(native L27/P20, exact published CP c0a1c7dc...), and both fullP20 Z training seeds fixed to E10/raw. The Z models already have trainB predictions: source evaluation, exact saved scores, CPU acceptance CP SHA, training publication, complete10epoch receipt and original source fixture trainB SHA must match before reuse. Therefore this job does not repeat Z inference. Optional selected fullP80 MSE scores may likewise be reused from their exact existing fixed trainB evaluation; no new P80 selection is performed here.

E1 needs genuine native causal-context prefill. Its published mature scorer preserves each trajectory's real retrieval/tool context; flattened P80 compact tokens are not fabricated into native contexts. The accepted full physical-v2 fixture points to the previously accepted8192 P80 pilot fixture. Its own P80 tokens and initial prompts are joined with original causal80 records per actual trajectory ID, EM and question, and must match. Full producer acceptance established zero changed P80 rows for all81920 pilot trajectories. This job reads only the needed subset of those accepted token banks, not new rollouts.

The E1 baseline adapter also replays exactly the first32 sorted QIDs from the existing viewed development ownG5 pool, comparing same-CP scores against the original saved E1 EMA prediction file SHA `1d821ef5059710213681b4eab3f0b55ad499485ae0b2a774a7684ffc6e9910ea`. Absolute probability tolerance0.001 is fixed before execution. This is an adapter consistency check, never a new checkpoint/threshold selection or a rerun of the complete development set. A failed replay produces an explicit alignment-failed status.

Outputs provide the same fixed trainB metrics for every model: diagnostic loss with correct units (direct Mixed BCE vs variance MSE), Mixed balanced accuracy, recall, all-wrong/all-correct false-positive decomposition, ROC-AUC and Top512 true-Mixed count/fraction. Thresholds stay0.5 for E1 and0.08 variance for Z/P80. No BCE is invented for variance regressors.

## Later root-owned execution

After publishing source, `deploy_selected_fixed_trainB.py --commit SOURCE_SHA --action stage` writes immutable source, `scoring.sbatch`, `future-unbound-trainB-template.json` and stage proofs. It performs only CPU import/metadata tests, never CP, prediction, GPU or Slurm operations.

After root binds the actual complete accepted references, the once-only invocation is:

```sh
sbatch REMOTE_TRAINB_ROOT/scoring.sbatch REMOTE_ROOT_BOUND_TRAINB_JSON CONFIG_SHA256
```

The source root is `A/selected-state-fixed-trainB2048-unified-20261023-v1`; normal/acd_u2H100,12CPU,128GiB,1hour. E1 scores2048 train questions plus32 adapter replay questions, five native trajectories each; packed model predictions are reused exactly. Existing training and old source roots are untouched.

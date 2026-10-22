# Packed variance final4096 scorer — prepare/stage only

No final journal, strict EM, aggregate final metric, checkpoint tensor or candidate freeze has been opened by this preparation. No Slurm job is submitted. Root remains the only submitter.

The new immutable output/locks are independent from native scorer v1/v2/v3 and from all active training. An explicit future freeze binds the exact candidate checkpoint SHA, actual `raw` or `EMA` field, viewed2048 validation evidence and scorer publication. The template is invalid for scoring. Baseline stays the original accepted native direct-Mixed E1 EMA checkpoint SHA `c0a1c7dc3663d4216966a3440c2103348bcc52db3cd04c2f5c59a784a2792917`.

Only after the freeze and an owned two-GPU allocation may the scorer open the original confirmation2048 and stability2048 materialization. It reuses v3's original label-blind G5 identity checks, own strict-EM/full-journal lineage and coverage checks. Question identities and each selected five-trajectory order remain unchanged. No G5 resampling or label-driven selection is allowed.

Candidate input is **one question prompt plus the same five genuine Prefix20 sequences**, literal training separator, short-prefix padding/mask and positions, in one actor prefill row per question. Frozen official blocks0..20 produce features; actual PackedVarianceProbe block21 and PackedVarianceHead (positions1024, output field) load the frozen state strictly. These are intentionally repacked group contexts as during packed training, not a claim of original causal rollout-context identity. The baseline separately retains its original causal per-trajectory prefill.

Candidate sigmoid predicts4v; divide by4 and sort **predicted population EM variance descending** to select512. Strict EM only supplies offline targets. Report original-v MSE/MAE/Pearson/Spearman, Top512 true Mixed and fixed diagnostic BA at **predicted v>=.08**. Candidate variance is not Mixed probability; there is **no candidate Mixed BCE**. Baseline has its own original Mixed probability threshold.5. Paired BA/Top512 differences use2000 question-bootstrap replicates, seed20261022, reselecting512 inside each replicate; they do not measure training-seed or cross-domain uncertainty.

Before candidate inference, any prior native scorer's `final-use-started.json` prevents silently unblinding through this new entry. Re-entry to this same scorer must use the identical freeze SHA. No further candidate/threshold tuning after final scores.

Three tiny pure synthetic CPU tests passed: explicit packed freeze contract/rejections, exact20/short true-policy extraction independent of EM, and zero paired differences for identical decisions/ranking. Static Python compilation passed. Actual GPU inference is pending the future Root-authorized allocation; no full-model test is claimed by preparation.

Future Root command after freeze and current job/lock check:

```bash
sbatch /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/packed-variance-frozen-final4096-20261023-v1/scoring.sbatch REMOTE_FREEZE_JSON EXACT_FREEZE_SHA256
```

# G20 exact uniform all-15,504 DP: completed negative result

Job **743485** completed all **136** updates from sealed E1 Adam **10464 → 10600**, and Slurm ended **COMPLETED / 0:0** after **45m18s** on ACD1-22. Final full checkpoint was durable before external evaluation. Source actor updates are zero; Adam/EMA/RNG were preserved, not reset.

## Matched comparison

Fixed validation: 2,048 questions, five authentic own trajectories per question, 663 true Mixed (32.3730%). The frozen G5 identities and native feature backend are identical for all three rows. Threshold is 0.5; Top512 ordering is (-q, string qid). All subsets retain the same q=1-prod(p)-prod(1-p) inference score, with no labels used to score or rank.

| Branch | Mixed BA | Mixed recall | NonMixed recall | AllWrong FPR | AllCorrect FPR | AUC | Top512 true Mixed |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| E4 (same native frozen G5) | 55.5291% | 66.3650% | 44.6931% | 49.8578% | 72.7273% | 0.576101 | 200/512 |
| G20 random4 (same E1 and budget) | 55.4422% | 91.1011% | 19.7834% | 75.5450% | 95.1515% | 0.575250 | 201/512 |
| G20 all15,504 DP | 55.2850% | 90.4977% | 20.0722% | 75.0711% | 95.4545% | 0.575195 | 197/512 |

**No improvement:** versus native E4, BA is lower by 0.2441 percentage points and Top512 contains three fewer Mixed questions. Versus the same-E1, same-budget random4 G20 run, BA is lower by 0.1573 points and Top512 contains four fewer. Both predeclared strict joint gates fail. The E4 row is an evaluation reference, not an equal-training-budget claim.

The all-15,504 variant still predicts many NonMixed groups as Mixed: AllCorrect FPR is **95.45%**, AllWrong FPR **75.07%**. Its selected512 K histogram (K=0…5) is **[187,39,35,50,73,128]**. Increasing the number of uniformly averaged real G5 combinations did not solve this false-positive problem in this matched run. This is a finite validation result, not evidence about an untouched final pool or an abstract impossibility claim.

## What was actually run

- Same 6,528 questions, exact B48×136 order, authentic G20 IDs/own strict EM, native features, E1 model/EMA/Adam/RNG and evaluation configuration as original G20 random4 job743370.
- Only the training grouping objective changed: random four G5 views → uniform average over all C(20,5)=15,504 subsets per question. Every original question has total weight1. No within/inter/stat or binary-Mixed auxiliary objective was added.
- Trajectory BCE + three-event NLL use the three-state log-space DP. Working chunk256 retains at most48×256 G5 rows; the whole 744,192-view objective graph is not retained. Native transformer forward is bounded to one micrograph, with two-pass same-RNG first-order VJP; no monolithic bitexact claim.
- One H100, normal/acd_u, 8 CPU, 64GiB, two-hour limit; no dependencies or other GPU requests. Measured peak CUDA allocated **6,655,523,328 B**, reserved **8,885,633,024 B** including evaluation. The allocation exited normally and is absent from squeue.
- First real GPU update was replayed twice with equality of raw model, EMA, Adam, raw buffers, EMA buffers and RNG. CP1 was independently CPU-loaded and all six component hashes plus full-file SHA matched. Final CP136 full SHA was independently streamed once; all producer checkpoint acceptance checks passed.
- Diagnostic512 BA52.6202%, AUC0.528879. This diagnostic is separate from fixed2048 recipe validation.

## Durable evidence

- Runtime source commit: `994185fad1c8d8ab9a9440a52fdda8d1a48c2d6a`.
- Config SHA: `af679a2a8ff93642a4573bb297f23ff954cca38b6749f0c7ff10c111e2cce4f4`.
- Original immutable plan SHA: `9d2ec808aa4365d80c7c7031487e0e746ad2165ceb2bfd59214aa631ab587e7a`.
- Final CP136: `checkpoints/cursor-000136.pt`, 3,745,866,714 B, SHA `fdd2ff1492a2abb418a1877c828b07444598b91f13e7bb910c43f87ed90dac9c`.
- Remote complete SHA: `f7b879b26347a6eee2d80128bb3cef9065cf4aaffc40a442afa8536df3d63e63`.
- Independent final audit SHA: `ba6cdf3e6b8c8dca8f0d786698e1882933492e5c360bb3ede6aaca57cff5a7df`.
- Final auditor source: `71156ee58cce4b1c5f094852d4628ddf561bf550`; standalone remote audit exit0.
- [Full small aggregate audit](final-independent-audit.json), [terminal Slurm receipt](terminal-slurm.json), [runtime/config backup manifest](actual-runtime-metadata/backup-manifest.json).

All large checkpoints/features/probability rows remain on HPC3 under `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g20-exact-uniform15504-dp-v1`. Mac retains source, configuration and these small aggregate receipts only. Untouched final4096 was not read or generated. No additional candidate or GPU job is launched by this report.

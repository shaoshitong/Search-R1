# Frozen final4096 paired scorer — staged only

This entry has not read final journals, strict EM, or final aggregate metrics. No GPU job is submitted by its deployer.

After an explicitly frozen candidate from the viewed validation2048, Root writes the exact candidate-freeze record using the remote `candidate-freeze-TEMPLATE.json` schema. The template is intentionally invalid for scoring. Bind the checkpoint path, bytes, SHA, actual state key (`raw`, `ema`, or `EMA`), layer, prefix, validation-evaluation proof, and scorer-publication proof. Baseline is exactly the original accepted E1 EMA checkpoint SHA c0a1c7dc3663d4216966a3440c2103348bcc52db3cd04c2f5c59a784a2792917.

All four final generation arrays must actually complete, then Root verifies no active same-name scoring job or occupied lock and submits one two-GPU job:

```bash
sbatch /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-scoring-20261022-v2/scoring.sbatch REMOTE_CANDIDATE_FREEZE_JSON EXACT_FREEZE_SHA256
```

The scorer uses one complete copy of the frozen actor per GPU, divides each independent2048 pool across the two ranks, and scores E1 EMA and the frozen candidate sequentially on the same original labelblind own G5 IDs. It supports L27/P20, L21/P20, L27/P40, L27/P80 and L21/P80. True retained policy tokens and preceding tool context are reconstructed from the existing real full journal; short responses are padded and masked. Actor and probe optimizer updates are zero. Hidden features remain in GPU RAM.

Each pool reports direct Mixed BCE, balanced accuracy at fixed 0.5, Mixed recall, nonMixed FPR, separate all-wrong/all-correct false positives, Top512 true Mixed, paired candidate-minus-E1 differences, Top512 overlap, and predeclared 2000 paired question-bootstrap percentile95 intervals. Intervals quantify within-pool question uncertainty, not training-seed or cross-dataset uncertainty. Raw predictions remain remote.

The first actual final use records the exact freeze SHA. Retries must use this same frozen candidate; final outcomes are not available for further candidate or threshold tuning. A completion receipt requires both intact2048 pools and all4096 own question identities.

Validation performed without final data: Python compilation; CPU-only tests for freeze rejection, original own-G5 binding and exactly zero paired differences for identical fake scores. The deployed CPU import check imports no torch and opens no checkpoint or final files. Actual GPU inference remains untested until the future authorized frozen confirmation job.

The v2 addition only admits L21/P80 in the candidate geometry contract and changes the unused output/staging root. Old v1 source and entries remain intact. The v2 template is still invalid and no candidate was frozen. A runtime guard prevents silently switching to v2 after v1 has already unblinded the final pool. Source commitbd8a37f569a1549ccf7bc87836cf06a74c4750ed was pushed and remote SHA verified. New source-publication1281B SHAa93c53f9fa180d69b256d780197a1065850d7c6ab39ce009d7ae13c28981df9b; new scoring.sbatch1325B SHA5db92ba6c70dc9c0191a3a6aea6e6dba4047252c1bc69b3d3fc88fbb6556fbfa. CPU-only import, bash syntax and three existing pure fixture tests passed. No final journal, EM aggregate, probe score, model/checkpoint tensor, or actual candidate-freeze receipt was opened or generated.

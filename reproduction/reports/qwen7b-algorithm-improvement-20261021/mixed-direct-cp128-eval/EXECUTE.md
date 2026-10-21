CP128 is hardlinked only after producer full-CPU reload and atomic publication. The pin controller reads metadata only; the GPU evaluator fresh-hashes the whole pinned checkpoint and official files before restoring all nine components. Main config bytes/base and original21 runtime file origins stay unchanged.

Root-only submission after actual READY:

```sh
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_cp128_eval.py --action stage
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_cp128_eval.py --action prepare
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_cp128_eval.py --action submit --submit-authorized
```

Single normal/acd_u one-H100,8-CPU,128-GiB,one-hour job. Immutable intent, same-name query, held-resource verification and explicit release protect submission. Unknown result preserves intent and requires reconciliation. No dependency, training update, main pause or final scoring.

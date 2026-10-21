Fast continuation sources are actually staged; no signal or GPU submission has been performed by this subagent. Controller source d89eb9015b3ef2b5643109b37a71464feae843bf, branch codex/mixed-direct-fast-continuation-20261022.

Stage/prepare accepted original config2a7f and original21 runtime files. New source-publication7750B/SHA dc34dcc787b3a52a5dbfe19d6d1fa39c9599ff158654bc4acdda56e71a8ace4a contains unchanged21 plus qualified reader c2fd, updater/worker24c68933 and stdlib handoff guard94bb18ef. Three handoff fixtures and five event CPU fixtures passed. Actual source writer746083 was RUNNING at cursor364 during prepare.

Root-only graceful handoff after the accepted actual full-step profile:

```sh
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_fast_continuation.py --action retire --handoff-authorized --profile-receipt reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-only-direct-group-20261021/actual-fullstep-profile-receipt.json
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_fast_continuation.py --action status
```

Retire verifies the live old job, exact source config/runtime and compute-node PID/UID/birth/cmdline/job environment before sending only SIGUSR1. A bounded metadata-only observer waits for exact main COMPLETED0:0, source training lock free and latest=progress=accepted intent=interrupted cursor. It refuses unresolved tails or any rollback, pins the highest complete source inode, copies original config bytes and creates a new config with explicit9state binding migration. It does not submit. Source checkpoint payload SHA and actual tensor restore happen on the new owned GPU allocation.

Only when status contains OFFICIAL_MIXED_FAST_EXACT_CONTINUATION_READY_NOT_SUBMITTED:

```sh
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_fast_continuation.py --action submit --submit-authorized
```

Single normal/acd_u one-H100,8CPU,128GiB48h, no dependency. Immutable unknown-submit intent and same-name/old-writer checks prevent duplicate submission. B32, real4G5 eachweight.25, original native/head/twoAdam/EMA/RNG/sampler/global21132 budget stay unchanged. GPU micro32, CPU producers8 and disposable lookahead8 batches are explicit implementation changes; no bitexact claim against scalar BF16 GEMMs. Original full feature payload seals are inherited; fresh manifest/index metadata and each used retained-row SHA are required, without falsely claiming a new full276GB scan.

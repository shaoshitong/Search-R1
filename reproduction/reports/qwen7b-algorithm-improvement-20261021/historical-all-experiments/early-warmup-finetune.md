# Early7B warmup and fine-tune history

Only existing local small reports were read. All values below are actual recorded EMA0.99 results; raw metrics were not reported. The independent2048 pool here contains641 Mixed questions (31.2988%), distinct from the later663-Mixed HF fixedG5 pool. Do not rank results across those pools. No new scoring or GPU evaluation was run.

| Experiment/checkpoint | Train Mixed BA | Fixed512 diagnostic BA | Independent2048 BA | Actual Mixed in top512 |
| --- | ---: | ---: | ---: | ---: |
| 7B native L26→block27 Prefix5 trajectoryBCE+MixedBCE warmup E1 (EMA0.99) | 59.6270% | 54.6659% | missing | 169/512 |
| 7B native L26→block27 Prefix5 trajectoryBCE+MixedBCE warmup E2 (EMA0.99) | 66.8419% | 53.2891% | missing | 181/512 |
| 7B native L26→block27 Prefix5 trajectoryBCE+MixedBCE warmup E3 (EMA0.99) | 73.7080% | 54.1050% | missing | 177/512 |
| 7B native L26→block27 Prefix5 trajectoryBCE+MixedBCE warmup E4 (EMA0.99) | 80.0430% | 54.5944% | missing | 178/512 |
| 7B native Prefix5 warmup E5 preserved tail ONLY E5 (EMA0.99) | missing | missing | missing | missing |
| Mixed-only continuation from warmupE2, Prefix20, rank weight0.1 E1 (EMA0.99) | missing | 51.3864% | 50.3138% | 182/512 |
| Prefix20 1:1 Mixed/nonMixed continuation, rank weight1.0 E2 (EMA0.99) | 73.8221% | 53.5235% | 56.3608% | 182/512 |
| Prefix20 1:1 Mixed/nonMixed continuation, rank weight1.0 E3 (EMA0.99) | 76.8178% | 54.4220% | 56.1364% | 190/512 |
| Prefix20 1:1 Mixed/nonMixed continuation, rank weight1.0 E4 (EMA0.99) | 79.6482% | 52.3079% | missing | 179/512 |
| Prefix20 1:1 Mixed/nonMixed continuation, rank weight1.0 E5 (EMA0.99) | 81.6780% | 54.8320% | missing | 188/512 |
| Uniform ownG5 class-weighted fork from direct E1 (later metadata only) (unknown) | missing | missing | missing | missing |

Mixed-only training has no negative class, so its train Mixed balanced accuracy is undefined. Its cohort differs from the later1:1 balanced67198-question training cohort. Warmup evaluated the original169028 questions, while fitting used a three-class-balanced233175-group plan with repeated wholeG5 groups.

WarmupE5 was not completed: the user changed the target to4 full epochs and a4-update E5 tail was preserved. The balanced rank-weight1 continuation does have recorded completed E2/E3/E4/E5 metrics; these are continuation epoch numbers, not five additional balanced epochs.

## Exact local sources

- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-four-epoch-unseen2048-20261020/delivery/metrics-and-acceptance.json` (72467 bytes, SHA256 `94a29b1f641fd89eb733f36eeaa26e5e981bd04a529fe3ea0959fc9984a513f2`).
- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-prefix20-balanced-rank1-20261020/mixed-error-diagnosis/mixed-ranking-diagnosis.json` (14872 bytes, SHA256 `30fb67e9b1fbee1592cb67f6a3dfaa356d91ab7fa18332af4cb39b3465fe7a4d`).
- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-prefix20-mixed-20261020/epoch1-complete/prefix20-epoch1-complete-audit.json` (5664 bytes, SHA256 `37829b6eb7431dbd12b8fac88b7233a01e910291e40f87902d76f360ab61d3de`).
- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-prefix20-mixed-20261020/prefix20-current-source-config.json` (9259 bytes, SHA256 `e9e6ff1ed44036b3070a21a9064b91a467f19d962f8af1de68a690cf1590aef9`).
- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/mixed-ft-status-1792510209.json` (17227 bytes, SHA256 `59dc98bec5ce1536fb9bd19b13d02ea1289fdabd414fbcaeb262ea09001aee7c`).
- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/balanced-rank1-from-epoch1-prefix20-20261020-epoch02-prediction-audit.json` (4301 bytes, SHA256 `1602133a0faa1dbac2600237075e20de58fe512ce6b85f60430e693a38f9ec52`).
- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/mixed-ft-status-1792514366.json` (17200 bytes, SHA256 `dea57f1a12abf9ffe67d52274eb283cc08c8e7e58b3750b6fad97886af1165eb`).
- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/balanced-rank1-from-epoch1-prefix20-20261020-epoch03-prediction-audit.json` (5888 bytes, SHA256 `70d87eb6ac9fc2fef06c31734ce656e3cf2ec766b1b99937758da71b99ea61f0`).
- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/mixed-ft-status-1792519011.json` (17216 bytes, SHA256 `b480e6f6042229cf717064fb764274e4d31342425a9f8002f084749a54a54a40`).
- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/mixed-ft-status-1792523075.json` (23948 bytes, SHA256 `58fb3f2cbc47eb1092a0825e5a8bd6bbd01311a0f1a62419ef561ccc0f19b4fb`).
- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-uniform-g5-weighted/actual-status.json` (65 bytes, SHA256 `d965ee59d617e3eaa0319aa7b7a43af3ffc889aff30bb55d57de75cbd9cd9289`).

Each JSON row also identifies the actual remote checkpoint path, pool identity, weight state, source and comparability limitations. Missing values remain null; none were inferred from confusion counts, other states or later pools.

# R2 multilayer replay bank accepted

Remote root: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-matched9088-bank-v2-20261021`.

Four original capture jobs 743087/743088/743089/743090 completed with exit 0:0. Each wrote 11,360 trajectories and 36 complete chunks; each was observed at least twice with positive trajectory/chunk/byte increments. Only original job IDs were escalated where the actual normal-partition forecast exceeded 30 minutes. No emergency partition, duplicate submission, student training, or active source edit was used.

The unchanged prepared/capture commit is `65f08f72b4e61cf72e3c43a202bda0508830a160`; Python3.6-compatible controller commit is `92bf7117404fb2b6c4dd23794a49773b777b67ed`. CPU full-SHA merge completed with exit 0 and status `ALL_FOUR_MATCHED9088Q_MULTILAYER_HF_REPLAY_BANKS_SHA_VERIFIED`.

Accepted source-of-truth: `prepared/bank-complete.json`, 2,756 bytes, SHA256 `c593363dfaca0edf3f142e1fa5f93bd7f461432ce7b0f1e186101039ac163dc4`.

| Capture index | Human block | Feature list SHA256 |
|---|---|---|
| 20 | 21 | `378053edf25605803b6752f633c985bbb8e2bbdd08845da42767a9d1e61edf09` |
| 23 | 24 | `242c788f66a8c28012c58e42f95e7b76a09cba8abadc8ff538e1fb43cb245c65` |
| 26 | 27 | `d248a82c87a1a59297829f74d4cfdf977368f9f16fec129873b7e7ea6684aced` |

Each `prepared/features-layer-<index>.json` is 38,306 bytes and binds 144 full-SHA-verified manifests, 45,440 distinct trajectories. Identity roles independently checked: fit 6,528, diagnostic 512, selection 2,048, totaling 9,088 distinct questions. `prepared/group-identities.json` SHA256 is `fd153748d2cddd8952267af9a5e3d18dbcf1cd191dea6c8277aeb4bbcec6553f`.

Use `ReplayFeatureReader` for uniformly HF-replayed fit and evaluation across all layer variants. This accepted bank is not claimed bit-exact to native capture or qualified to replace production-native inference. All model, feature, and rollout assets remained remote. The compact machine-readable acceptance and source bindings are in `replay-bank-v2-FINAL-ACCEPTANCE.json` beside this report; older RUN-STATUS snapshots describe earlier times only.

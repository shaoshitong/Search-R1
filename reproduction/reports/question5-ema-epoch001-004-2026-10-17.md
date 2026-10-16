# Question＋prefix5：epoch1–4逐轮EMA指标（2026-10-17）

上海时间00:51:12的最新实际回执确认，**4/200个完整epoch已提交，第5轮同进程完成700/2175次Adam／EMA，累计28975次**。四rank一致、身份存活，无记录错误、退出或nonfinite拒绝。以下全部来自各轮已闭合的原生metrics／checkpoint／complete记录；每轮训练与测试使用该轮同一EMA0.99权重。没有第5轮评分。

固定训练集每轮445440个G5／2227200条轨迹；固定测试512个G5／2560条。混合定义为`0<K<5`，只有1条正确也算混合。表中accuracy阈值为0.5；混合组和逐轨迹正确性分别计量，不能混用。

训练混合balanced accuracy从67.7370%持续升至84.7472%，联合loss持续下降；测试混合balanced accuracy依次为54.3517%→57.3314%→56.3273%→56.5715%，测试联合loss相较首轮没有改善。第4轮训练／测试混合balanced accuracy差28.1757个百分点，当前不能宣称泛化良好。

第4轮是首个完整全局1024轮：2175次Adam加既定全量EMA训练／测试评测记录耗时2014.141419秒，即33.5690分钟，不含checkpoint写入。第2、3轮计时分别只覆盖恢复后的1363/8700、8697/8700次更新及评测；这些计时不能直接作完整轮速度比较。即使与首轮41.6574分钟相比，也没有受控倍速结论。

完整数值与原生哈希见[统一JSON](question5-ema-epoch001-004-2026-10-17.json)。所有checkpoint状态SHA为当时保存的可变`latest.pt`记录，不证明每轮都有独立不可变权重副本；不可变保存仍为每10轮及200轮。本报告只汇总已有本地回执，不新增网络查询、模型读取、测试、review或评测。

## G5 mixed accuracy

| Epoch | Train balanced | Test balanced | Train ordinary | Test ordinary |
| --- | ---: | ---: | ---: | ---: |
| 1 | 67.7370% | 54.3517% | 70.5238% | 58.0078% |
| 2 | 76.1967% | 57.3314% | 78.5441% | 62.3047% |
| 3 | 80.1052% | 56.3273% | 81.7434% | 60.5469% |
| 4 | 84.7472% | 56.5715% | 85.8933% | 61.3281% |

## Trajectory correctness accuracy

| Epoch | Train ordinary | Test ordinary | Train balanced | Test balanced |
| --- | ---: | ---: | ---: | ---: |
| 1 | 79.6537% | 77.3047% | 56.5685% | 52.2220% |
| 2 | 80.8413% | 77.3438% | 60.8854% | 51.9247% |
| 3 | 81.8230% | 78.2422% | 66.2237% | 54.0464% |
| 4 | 83.4955% | 77.0312% | 69.5869% | 52.3693% |

Test all-wrong trajectory baseline:78.1641%; test all-unmixed ordinary group baseline:64.2578%; constant-prediction balanced baseline:50%.

## Same-EMA losses

| Epoch | Train trajectory BCE | Test trajectory BCE | Train mixed BCE | Test mixed BCE | Train total | Test total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.474310255 | 0.567329205 | 0.602849333 | 0.753820239 | 1.077159588 | 1.321149443 |
| 2 | 0.416559220 | 0.592635315 | 0.484882650 | 0.779791027 | 0.901441870 | 1.372426342 |
| 3 | 0.381938702 | 0.570880566 | 0.433038426 | 0.777595952 | 0.814977128 | 1.348476518 |
| 4 | 0.325693022 | 0.592608474 | 0.326148609 | 0.774522007 | 0.651841631 | 1.367130481 |

## Actual clocks and consumed counts

| Epoch | Returned Adam per epoch | Total Adam = EMA | Train G5 / trajectories | Test G5 / trajectories | Recorded seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8700 | 8700 | 445440 / 2227200 | 512 / 2560 | 2499.441084 |
| 2 | 8700 | 17400 | 445440 / 2227200 | 512 / 2560 | 985.782079 |
| 3 | 8700 | 26100 | 445440 / 2227200 | 512 / 2560 | 2293.261938 |
| 4 | 2175 | 28275 | 445440 / 2227200 | 512 / 2560 | 2014.141419 |

Four rank copies count once. Every recorded save has four equal Adam counters and zero nonfinite rejects. Epoch1-3 use mean256 trajectories/Adam; epoch4 uses mean1024. Epoch2 and3 durations are suffix timings as described above.

## Mixed G5 confusion matrices

| Epoch | Split | Positive | Negative | TP | TN | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train | 160986 | 284454 | 92862 | 221279 | 63175 | 68124 |
| 1 | test | 183 | 329 | 76 | 221 | 108 | 107 |
| 2 | train | 160986 | 284454 | 109032 | 240835 | 43619 | 51954 |
| 2 | test | 183 | 329 | 73 | 246 | 83 | 110 |
| 3 | train | 160986 | 284454 | 119443 | 244675 | 39779 | 41543 |
| 3 | test | 183 | 329 | 76 | 234 | 95 | 107 |
| 4 | train | 160986 | 284454 | 129775 | 252828 | 31626 | 31211 |
| 4 | test | 183 | 329 | 73 | 241 | 88 | 110 |

## Trajectory correctness confusion matrices

| Epoch | Split | Positive | Negative | TP | TN | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train | 474644 | 1752556 | 77531 | 1696516 | 56040 | 397113 |
| 1 | test | 559 | 2001 | 43 | 1936 | 65 | 516 |
| 2 | train | 474644 | 1752556 | 123907 | 1676591 | 75965 | 350737 |
| 2 | test | 559 | 2001 | 38 | 1942 | 59 | 521 |
| 3 | train | 474644 | 1752556 | 185285 | 1637077 | 115479 | 289359 |
| 3 | test | 559 | 2001 | 62 | 1941 | 60 | 497 |
| 4 | train | 474644 | 1752556 | 215234 | 1644377 | 108179 | 259410 |
| 4 | test | 559 | 2001 | 48 | 1924 | 77 | 511 |

## Native evidence SHA256

Epoch 1 (fully committed):

- Execution: `091f3979549347d190065387c2214af6`
- Runtime source: `98b5ec2af9a5b45cd535e67408980c72a75d7746`
- metrics_file_sha256: `667c446eb39c0dd8488778e967cd176d56a2d0537eae2312a58e39688cb819ec`
- epoch_metrics_canonical_sha256: `0a819c1d3c6f2537719817e674b63d1b2ee59ba549142d16b4466cc522e60621`
- checkpoint_metadata_sha256: `b04a7bfde161dea9351f36766ca1e7a6d407aafe3ddc1dc85b8506d091b99675`
- complete_metadata_sha256: `13fbf538256ef7dc748192764cbf925b13a3af3e6b5dfc6874bcd6a13d648c31`
- checkpoint_state_sha256_at_save: `a3a4959f14413948f88c9a7add54eb670746fa63005972390c0ab907d30a5f8c`
- binding_sha256: `80c0abea4462b1bfb6852b0f33a79258ad61f8a17a89fe47a8ff2e0e830580c6`
- State bytes at save: 1242422752; mutable latest, no independent immutable epoch copy asserted.

Epoch 2 (fully committed):

- Execution: `141a9ada1d464f0d9925d8397a194944`
- Runtime source: `ed2a997e5162cd3a6e1dff0bd1d91fcc659caf2a`
- metrics_file_sha256: `d1884da3e4535d35259d0790f12fc45b9993fa70a12d4105298e75d7084f6512`
- epoch_metrics_canonical_sha256: `e632754be438b847c37f068a86c8b6334cf1ff17fe7078bcb4009076a7156cf0`
- checkpoint_metadata_sha256: `8f7cc815005d00f0e2eb8a4c6a2340a4a2ff42fc65e8c1d3ea389e65d012b861`
- complete_metadata_sha256: `a686b8d62689123a6632bae2a079637b6d6c8e974e9aa8e69c4e745514f008b6`
- checkpoint_state_sha256_at_save: `b59d75f190e99a8f7d1f2a577b36d39260bf557cc3f99a3f32796aa7bde459aa`
- binding_sha256: `1c5d35aa00de7f2f0d226566122beeb4f81a3b9fa13893f3a26b2a33e6ef0045`
- State bytes at save: 1242423712; mutable latest, no independent immutable epoch copy asserted.

Epoch 3 (fully committed):

- Execution: `bcfa1de0a90a4fcda70263a75153fcee`
- Runtime source: `ddf36cca14ac7aafb0b5395ffa88d57ed8c35e74`
- metrics_file_sha256: `1e47331c2d0b4ef67c20b03319e526101515592ee4a79750092e00e3adbaa71e`
- epoch_metrics_canonical_sha256: `7e1ec9f0c20aa297244f2c878a11041cfb674a06d647ba22ea73266043ca1173`
- checkpoint_metadata_sha256: `9207011e7b70f143405991682f905bbafe3a4a5534e4f0dfff481167e827e152`
- complete_metadata_sha256: `6b61fdbb8d0dea7fea418dbc54bed205bc9df6fea4657c2a1b7dbff5a958221d`
- checkpoint_state_sha256_at_save: `4fccd80f9c0761e0b613f8fbf45f0b562c08ec9d007fa868aa746bc49b4f4437`
- binding_sha256: `f82e64dd66f442e3d6595694d26f735f3d7a6d3dc8996e1de435f8f7c38ace02`
- State bytes at save: 1242424288; mutable latest, no independent immutable epoch copy asserted.

Epoch 4 (fully committed):

- Execution: `bcfa1de0a90a4fcda70263a75153fcee`
- Runtime source: `ddf36cca14ac7aafb0b5395ffa88d57ed8c35e74`
- metrics_file_sha256: `bb615cea90fa041566cb24f2df1148867bacb0573f0a2c69f323a82f61a22f90`
- epoch_metrics_canonical_sha256: `d0e06ba0c5ce4db0a7edb385d5baf91225fdb7fbe3edbc7f322c858d7efa7f40`
- checkpoint_metadata_sha256: `3851f106b7c8b9f96958522ecb3125964b010baf41deabd62edb4f30ec7a2298`
- complete_metadata_sha256: `3a9b215c16170ffb14251e88943f8d82a727c1ec7fd9df4f0d7137b0682cc82a`
- checkpoint_state_sha256_at_save: `f5b7c900bca27a62c1215a482f1697ea8090c279d1b17fa093e06a5226d5231a`
- binding_sha256: `f82e64dd66f442e3d6595694d26f735f3d7a6d3dc8996e1de435f8f7c38ace02`
- State bytes at save: 1242424352; mutable latest, no independent immutable epoch copy asserted.


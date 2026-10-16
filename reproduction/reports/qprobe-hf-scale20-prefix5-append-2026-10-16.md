# HF追加完成 — A800 prefix5（2026-10-16）

已在 2026-10-16 17:15:01（Asia/Shanghai）完成HF实际上传及全部文件回读验收。

- 目标：[Sutoonq/Search-G1-prefix20-dataset-probe-epoch68-20261016-111a2d](https://huggingface.co/datasets/Sutoonq/Search-G1-prefix20-dataset-probe-epoch68-20261016-111a2d)。
- 新配置：`a800_frozen_g5_prefix5_scale20_20261016`；HF final commit：`d794759ab9f28cf184128b227fb38f40b59fcedc`。
- HF保持私有；原17个非README文件的Git/LFS内容身份与大小全部保持，epoch68归档、原teacher/test、probe及optimizer均未覆盖。根README保留原正文，额外注册新配置。

| split | 唯一题目 | 轨迹行 | 原完整回答strict EM=1 |
|---|---:|---:|---:|
| train | 102400 | 512000 | 111509 |
| fixed test | 512 | 2560 | 559 |
| 合计 | 102912 | 514560 | 112068 |

Train包括原5120题/25600条合格事件复用和新97280题/486400条事件；固定test是既有随机heldout512，非官方dev512。所有G5完整、顺序与题/事件身份绑定，train/test的题ID、hash和归一化文本隔离。原teacher和test与现有归档的SHA完全一致；在新Parquet配置中再次提供是为自包含加载，不是重复采样。

本配置共20个train+1个test Parquet，75,150,605 bytes；另加provenance和说明，新增23文件全部经HF实际完整回读，byte数/SHA256与源文件一致。导出时514560行原IDs、positions和policy mask也已独立逐行Parquet读回SHA核验。

**本次是prefix5，不能当作原prefix20或完整response。** 表中包含question/prompt/前5个policy token及其有效上下文文本、原IDs/positions/masks与原完整回答strict EM。新增完整response先前已在compact SHA验收后按原合同删除，不能由prefix还原。本次没有重生成、重评分、GPU计算或训练，也未上传feature bank、模型、Head、optimizer或环境。

CPU监督入口真实exit0，owned PID1201031/birth622069738和整个owned session均已实际消失。上传仅Mac有界内存中转，没有本地数据落盘或凭据持久化。旧正式actor goal及人停探针保持暂停。

```python
from datasets import load_dataset
data = load_dataset("Sutoonq/Search-G1-prefix20-dataset-probe-epoch68-20261016-111a2d", "a800_frozen_g5_prefix5_scale20_20261016", token=True)
```

对应JSON包含全部新文件实际回读SHA、旧文件内容身份及小型执行证据SHA；源码入口commit `10cb68440fb97dc5d672e60557c7ff7eb7f1ccdf`，helper SHA `940921c1b1769eb3fc2038c5326263046a72f5f76e1c5139d5603bf385fa9e5a`。

# A800 102400题 prefix5 数据追加到HF（2026-10-16）

用户授权目标：`Sutoonq/Search-G1-prefix20-dataset-probe-epoch68-20261016-111a2d`。保留目标私有状态、epoch68原归档、探针权重、optimizer和所有历史文件；仅追加独立配置 `a800_frozen_g5_prefix5_scale20_20261016`。

## 数据范围

| split | 唯一题目 | G5轨迹事件 | 来源 |
|---|---:|---:|---|
| train | 102400 | 512000 | 原5120题/25600轨迹复用 + 新97280题/486400轨迹 |
| test | 512 | 2560 | 固定随机heldout，同事件复用，非官方dev512 |

新配置自包含，可直接 `load_dataset(repository, config, token=True)`。旧teacher identity SHA `88998de7a30455b9135b02cc3c82251fbc1ba1765708244538b6a66c1371e871`、complete SHA `07f4e2e3937c814637b783468ff1062af7ba701e1ac1e136da2dd56f838b45b1`，以及test labels SHA `d7d056477666558a2dd0fbaeb0f7ee8aff679e80c5730fbb0a3a8c684395740b` 与目标原teacher归档manifest一致。这些事件在新Parquet格式中再次出现是为完整可加载配置，不是重新采样。

本次新增配置是 **prefix5**，与目标既有prefix20/epoch68实验不同。每行提供question原文、实际prompt、含上下文的有效前缀文本、前5个生成policy token文本、原始input IDs/position IDs/policy mask/attention mask、原完整回答的二元strict EM、G5正确数/混合状态及来源哈希。padding已由原生产者去除，attention mask均true；short1–4 token按真实长度保留。新增完整response在compact验收后按原合同删除，因此本配置不声称包含完整response，不重建/重采样它。

不上传300GB特征bank、冻结模型、Head、optimizer、环境或凭据。本次只CPU读取不可变已验收数据，不生成、重评分或训练。

## 固定来源

- 主run `62407ad0ed404586998439ca1f55b902`；execution `dd3032e313e14ce2b9c8fea996aaf721`。
- BASE `/data/shaoshitong/Search-G1-a800-fresh200-0850db2977bd4801b9ced300b0ab387f`。
- runtime commit `fcb77204f48d5cb8ec21c68df7d4a2c937b3557a`。
- dataset complete SHA `ea1aecb682d8faf184f7b4c32d52bb48b88aeb1bc483242993cf349c51a85fcf`。
- common manifest SHA `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`。
- official Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`。

## 执行入口

`reproduction/operations/qprobe_hf_scale20_prefix_dataset.py` 的 `build` 在新独立远端目录执行；入口源码先Git发布并核验实际远端blob，禁止修改活跃runtime。它全SHA检查200个chunk/commit/labels/prefix文件，校验四rank冻结记录、原EM、题序、G5身份、IDs/positions/mask逐行SHA、test隔离；写20个train Parquet及1个test Parquet，再独立逐行读回核对精确token载荷和行数。模型目录只读取tokenizer文件，不读取模型tensor或features。

`upload-memory` 沿现有control.py物理路由传输已封存Parquet，每批最多4文件/256MiB，仅Mac内存；大数据不落本地盘。HF token仅隐藏stdin进入内存，不login、不放参数/env/源码/cache，不输出。每批以当前parent commit追加，已存在同路径仅在SHA完全一致时跳过；不同则失败，绝不覆盖。所有数据上传后才追加根README的config入口。保留原README正文、其他metadata及所有旧文件内容SHA/size。

完成条件：514560行、102912题、G5完整且train/test隔离；全部新文件HF实际回读size/SHA一致，原归档Git/LFS内容身份保持，HF私有状态保持，保存小型verified回执。失败保留源文件及已上传文件，不能把部分上传说成完成。原200outer goal和人停的探针不因本次上传恢复。

# Question diversity probe 与 mixed-first 的正式iter延迟

> 对象更正：用户随后明确要同线程的H100八卡探针实验。本文件比较A800历史版本，不是该请求的目标；其亚秒/数秒探针成本不适用于H100 job3402。正确对照为 `outputs/h100-qprobe-vs-mixed-latency-20261016/REPORT.md`。

2026-10-16只读实际console标量核查。没有重跑模型、训练、生成或评估，没有停止/修改作业，也没有下载问题、轨迹、token、tensor或权重。

## 核心结果

**Question probe 将候选阶段的完整生成量从10240条减为2560条，在线训练选题head只需亚秒到数秒。但当前A800四卡的actor阶段需要9–13分钟，整个iter没有随生成量同比缩短。** 这些是不同硬件和运行实现下的实测，不能作为同条件算法加速归因。

| 每iter主流程 | mixed-first | 旧question-diversity小head | 新Qwen-block questionhead |
|---|---:|---:|---:|
| 硬件 | H100八卡 | A800四卡 | A800四卡 |
| 样本范围 | 187普通outer均值 | 32普通outer均值 | 仅1个完整outer实测 |
| 候选问题 | 2048题全部G5生成后筛选 | 5120题缓存特征打分后选512 | 5120题缓存特征打分后选512 |
| 完整生成轨迹 | 10240 | 2560 | 2560 |
| 评分/采样/筛选总计 | 820.504秒 | 214.523秒 | 268.562秒 |
| 其中agent交互计时 | 未单独记录 | 198.131秒 | 240.167秒 |
| Search-G1 reward探针＋优势 | 115.832秒 | 128.390秒 | 147.604秒 |
| actor更新 | 130.123秒 | 738.722秒 | 776.855秒 |
| 在线questionhead更新 | 不适用 | 0.682秒 | 6.542秒 |
| 其他训练内工作，含可用的ref | 122.351秒 | 86.665秒 | 90.372秒 |
| **iter主流程总计** | **1188.809秒＝19分49秒** | **1168.982秒＝19分29秒** | **1289.935秒＝21分30秒** |

agent交互是采样总计的子集，不可再加一次。其他训练内工作为timer残差；不将缺失的reference字段补造为独立0秒记录。

## 实验身份和数据去重

1. mixed-first：run `15bb15365346401da1d919e9cde8f813`，正式job3381/source `0553adec146537e07b84377ce31cba41a54f37cd`，h100-8。沿用上一报告的187普通outer：6–200中剔除1536题的27–29与有保存/验证的10/50/100/150/200。原来被废弃的job3368不混入。
2. 旧question-diversity：run `1f34290a70454871aea9d368dc8a0b36`，execution `59ef9209e5724cf6adc6b126d30071c8`，source `91f673c58ff415115c581b36568801b39ae3db86`。远端base `/data/shaoshitong/Search-G1-a800-fresh200-af113a29fa9a48a7b0c1658ce4c0396c`。此次实读6–38共33个完整outer，排除10轮checkpoint保存后32轮；39轮只有采样，没有完整更新，不计。旧小head预测K的六类分布，在CPU缓存特征上训练，和新Qwen-head不同。
3. 新Qwen-block：run `509c391b7e5c4174bfb55f724f41a3d6`，execution `60fca3cc2edd4dedb3773230b91f34ea`，source `fcb77204f48d5cb8ec21c68df7d4a2c937b3557a`。远端base `/data/shaoshitong/Search-G1-a800-fresh200-e9c4c584107b4624bcc9787aaf24b7a2`。只有outer1完整，outer2部分更新不计。固定layer27全token特征、GPU copied-Qwen block35、mixed二分类＋std。

两个A800实验均实际使用物理GPU0–3。每个run的 `trainer.log` 与 `search-g1-common8-training.log` 有重复console计时：先按logical step核全部metrics逐项一致，再去重。66条旧console记录是33个outer，而不是66；2条新console记录是1个outer，而不是2。

每个完整outer都与selection审计和actual-updates逐项核对。本次旧33轮、新1轮均实际20次Adam、无skip；这个结论来自实际记录，并非由旧版本“处理20个opportunities”的合同推断。每轮训练2560条，不再乘20，也不乘rank数。

## 探针开销具体在哪里

原生Qwen首轮的268.562秒由240.167秒agent交互和28.395秒其他collection工作组成。后者包括5120题cached feature/head评分、选题、RNG保存、EM标签和序列化，**没有独立选题评分timer，不能将28.395秒全部叫head推理延迟**。

正式在线questionhead update为6.542秒，其中内部更新elapsed为6.503秒；外层计时还包括状态备份和完成票据。首轮仅正式512题，2epoch、4次Head Adam，warmup200→204，没有5120预热replay。不得按5632条replay解释，也不能把后来的单条正确性prefix实验当这个历史run。

旧小head32普通轮平均外层update0.681594秒，内部update0.320秒，实际每轮64–100次Head Adam、replay8192–12800组。训练容量、目标和replay不同，不能用它的亚秒成本冒称Qwen-block成本。

Search-G1的b/d reward探针属于 `adv` 阶段，不是用于选题的questionhead。表中147.604秒不能被归为questionhead训练。

## 吞吐与时间份额

| 每阶段速率 | mixed-first均值口径 | 旧question小head | 新Qwen-block首轮 |
|---|---:|---:|---:|
| 完整生成轨迹 / 评分采样总秒 | 12.480条/秒 | 11.933条/秒 | 9.532条/秒 |
| 完整生成轨迹 / agent交互秒 | 未单列 | 12.921条/秒 | 10.659条/秒 |
| 训练轨迹 / actor秒 | 19.674条/秒 | 3.465条/秒 | 3.295条/秒 |
| actor摊销 / 真实Adam | 6.506秒/次 | 36.936秒/次 | 38.843秒/次 |

mixed-first时间主要在候选采样（69.02%）；旧questionhead实验主要在actor（63.19%），在线questionhead仅0.058%；新Qwen首轮actor占60.22%，questionhead更新占0.507%。所以降低head拟合成本本身对该轮总时间的改善空间很小。

旧question小head普通轮中位数：评分采样218.392秒、actor779.168秒、iter1239.771秒（20分40秒）。32–38后期7轮的描述性均值则为评分采样192.434秒、actor542.132秒、iter912.776秒（15分13秒）；只说明速度有阶段差异，不把后期快轮替代全32轮统计或认定因果提速。

## 计时范围与边界

两方案都按 `candidate_fill + timing_s/step` 计算记录内iter主流程。采样在step timer之外，actor和questionhead更新在step之内。一次性模型加载、全题库缓存构建、warmup、baseline、日志后milestone reward-probe refresh，不混入正式iter。

新Qwen启动→首次actor返回的1851.7546秒是包含启动/加载的约31分钟，不是actor本身，也不是稳态iter；本次恢复出实际console后得到了上表776.855秒actor和1289.935秒主流程。

旧observer的primary actual-update白名单未保存timer，之前摘要中的unknown只表示该入口缺字段。此次从原console恢复真实timer，未把unknown补0，也未重新执行模型计时。

A800四卡与H100八卡的GPU型号、并行方式、padding/actor实现、轨迹长度、运行时负载和选中题分布不同。只能报告实际阶段变化，不能把两列总时间之比称question-probe算法的同硬件加速倍数。新Qwen只有首轮，不能推断长期稳定均值。

## 证据

`qprobe-remote-scalars-first.json`保存原console的标量及文件/行SHA；`comparison-analysis.json`保存去重、分母、每轮phase与actual writes；`summary.json`为compact聚合；读取脚本 `read_qprobe_timing.py`，核对脚本 `analyze_comparison.py`。随后第二次实际读取15个关键文件的完整SHA，全部与首次一致，证明 `remote-second-readback.json`。原问题、轨迹、特征和权重继续只在远端。

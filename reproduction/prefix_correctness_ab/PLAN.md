# Search-G1：逐轨迹 5-token 前缀正确率探针，8H100 离线 A/B

日期：2026-10-16（北京时间）。这是本线程最新人工方案；取代旧混合题二分类＋方差辅助实验。旧作业3402已按人工要求发送停止，实际终态和资源释放另附证据。本文件是实施方案，不代表新探针已训练或已取得效果；正式actor/PPO 200轮保持停止。

## 1. 目标与唯一损失

每个样本对应一条真实完整轨迹：实际初始prompt、该条轨迹自己的前5个policy生成token、该条完整轨迹最终strict gold EM标签C∈{0,1}。

预测p=sigmoid(z)≈P(C=1 | prompt, prefix, actor版本)。损失只有全样本均值BCEWithLogits(z,C)，不含mixed标签、六分类、方差/标准差回归或类别重加权。保留同输入不同标签；不要求训练准确率100%。

5120题×G5提供25600条训练记录，但统计独立单位仍为5120个问题。不要给五条前缀复制K/5或group-mixed标签。A组同题五条输入相同时，平均BCE数学上等于对K/5拟合；B组逐轨迹输入不同，必须保留各自C。

## 2. 缓存与无泄漏特征

合格只读teacher缓存的既有来源：

`/mnt/data/shaoshitong/search-g1-qprobe3-h1008-003a0b0-20261015-v3/teacher-warmup`

其保存契约为10个chunk JSON（各512题×5）和对应rollout PT；chunk的trajectory_ids[j]/strict_em[j]与原DataProto的behavior_trajectory_id、behavior_turns相互绑定。代码契约及既有小型依赖证据已核对；本方案尚未把全部远端原件逐项读回验证。

- 精确使用turn0的actual_request_token_ids和该trajectory自己的raw_generated_token_ids。不能decode后重新tokenize、替换模板或仅按qid猜兄弟轨迹对应关系。
- 前缀取首次实际解码流最前min(5,真实已生成长度)个policy token。包含前缀内真实发生的EOS；提前终止保留实际短前缀。不得跨工具调用、拼检索文本、后续turn或用未来token补齐5个。
- 同时记录raw-prefix span与原policy/sampling mask映射；不能把kept_token_count、prefix_cut等整轮后处理结果倒用为第5token时可见状态。≥5token时不向head提供后续finish_reason/final_turn/stop_reason。
- 输入、mask、positions来自原请求记录及真实有效token位置。若原始记录缺mask/positions，先证明原接口如何去padding，按原规则重建并明确记录reconstructed，不能冒称原件已保存。位置仅随有效token递增，prefix紧接prompt。
- 所有未来回答、最终答案、gold、EM、后续检索内容仅供标签/审计，绝不进入encoder/head输入。
- 对prompt＋短prefix做一次因果前向，冻结官方Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`，eval/no_grad/use_cache=False。仅输入短前缀，不重算完整轨迹、不重新生成teacher或检索。
- 沿参考实现明确采用**decoder index27（zero-based，第28个block）输出**，即消费完prefix后所有有效token的hidden；缓存spec写明`decoder_index=27, boundary=post_block`。不能误用预测第5token、只看过前4token的hidden。参考代码layer27就是这个编号约定。
- 保留全部有效token（包含新增prefix位置）的2048维表示并detach；旧question-only feature缓存不可重标为新prefix特征。

先只读扫描训练及诊断两组实际原prompt＋prefix长度，确定统一Lmax；不使用标签确定长度，不直接沿用244/249，不静默截断。两组统一Lmax和flatten顺序，保存实际长度分布、最大值、短EOS比例、原ID/encoder/spec哈希。raw首5含EOS与旧sampling_mask有效policy口径不同，不能用旧mask过滤掉真实raw前缀token；两个映射都保留。

## 3. 公平A/B与模型

| 项目 | A：question-only正确率 | B：question＋5-token正确率 |
| --- | --- | --- |
| 输入 | 同一实际初始prompt，同题5条相同 | 同prompt＋各自真实短prefix |
| 标签 | 各条完整轨迹自己的strict EM | 完全相同的逐轨迹标签 |
| 可训练模型 | 官方最后完整block复制＋flatten Linear(1) | 相同结构与初始权重 |
| padding | 同Lmax，prefix空槽mask=0 | 同Lmax，仅实际prefix有效 |
| 优化 | 同seed/顺序/LR/优化器/1000次同步Adam | 相同 |

使用**一个**完整原生Qwen最后decoder block（index35），完整复制attention、MLP、RMSNorm所有参数；不沿用旧三block混合头/Adam。全部有效hidden→复制block→再次将padding输出置0→补到同Lmax→flatten→单logit Linear。保留因果attention、RoPE position和每样本隔离；不能把不同样本拼成可互相attention的一条序列。

冻结encoder与actor无梯度；训练只更新复制block及Linear。A/B相同head容量、更新预算和样本顺序；有效token不同导致实际特征计算量不同，时间分别实报。优先顺序执行A、B，同8卡避免竞争；参考线程的20epoch决定不自动覆盖本线程10epoch方案。

## 4. 探针也用8卡：256/8=32，无累积

- 8个DDP rank各持有相同head；全局batch256，每rank32轨迹，gradient_accumulation_steps=1。每rank本地mean BCE，DDP平均梯度等于全局256样本mean BCE；不再额外除以8或重复聚合loss梯度。
- 每epoch全局打乱25600条并恰好用一次，各rank3200条→100次同步optimizer.step；10epoch→**1000次Head Adam/rank**，不是旧200，也不能8rank相加冒称8000次训练更新。
- A、B使用同全局permutation、同分片规则/初始化/学习率计划；保留实际见样次数、每rankstep、梯度同步及样本覆盖计数。指标按样本数sum/count聚合，accuracy阈值0.5固定。
- 特征在远端分片/mmap按batch读取，不整份放入每卡。先离线提取特征、卸载冻结encoder，再做DDP head训练，避免无用常驻显存。大特征不经Mac。
- BF16模型计算、FP32 master/Adam和BCE归约，FlashAttention2 varlen去padding并保持样本隔离。保留当前实际无效数据/非有限loss或grad检查，不增加猜测阈值停训、重复baseline或额外GPU门禁。

静态显存依据：hidden2048/attentionheads16/KVheads2/FFN11008，单复制block约7708万参数；FP32参数＋grad＋两个Adam moment约1.15GiB，另有BF16副本、DDP通信桶和激活。若**仅举例**L=249，local32的BF16输入约31MiB，单MLP中间约167MiB，flatten Linear约51万参数。单block头训练有充分理由预期能放入80GB H100；实际Lmax、后端和峰值尚未实测，不能把旧244或此例当新长度/显存验收。以后真实训练首步记录峰值即可，不为此增加独立压测。

## 5. 数据与报告

训练保留全部既有5120题×5=25600条。诊断用**新随机512题×5=2560条**，与train按question及规范化文本隔离，A/B共享同一固定题目、轨迹、标签和actor版本；不是把同题轨迹随机拆分，也不复用canonical固定512作为训练诊断。若已有合格新512缓存，优先共用只读缓存；尚未具备时只为这些新题生成一次完整G5，两臂不重复生成，不触其他线程任务。

每个epoch后同一诊断集评估；训练前epoch0也记录。诊断不反传、不纳入replay、不以测试结果挑选最优checkpoint，主比较固定epoch10。数据按qid完全隔离，不从5120训练题中扣出验证题，因此100step/epoch口径保持成立。

主要交付轨迹二分类accuracy，并同时报告：

1. train/diagnostic正确与错误比例、多数类accuracy基线、BCE、Brier；固定0.5阈值precision/recall/F1作为解释。
2. 概率均值/分位数、接近0/1比例及饱和定义；A/B配对差值，按question聚类统计，不当25600条独立题估置信区间。
3. 同题5条前缀不同ID序列数、全同题比例、成对相同比例、逐位置差异、短prefix/EOS比例。前5token多为格式时可能无额外收益，如实报告。
4. 抽特征、head训练和评估各阶段wall time/GPU峰值/缓存字节数，1000次实际Adam/rank。训练集拟合与新题泛化分别报告。

这些指标与旧“mixed题分类99.88%”不是同一目标；旧在线Top混合率不直接比较为新轨迹accuracy。若prefix未改善诊断结果，也交付真实负结果，不宣称方案已有效。

## 6. 后续选题接口（本阶段不启动PPO）

候选5120题各独立生成同actor版本下G5短prefix，得到p1…p5。给定这些prefix且后续条件独立的近似下：

`q = 1 - product(p_j) - product(1 - p_j)`

按q取Top512，随机探索0。不能先平均p再代入同概率公式。选中后继续**这同5条prefix**；actor快照、RNG/解码状态、首段行为logprob、检索/turn状态及剩余token预算必须明确保持，前缀纳入正常PPO轨迹及概率记录。不丢掉被评分的prefix重新采另一组；不把有prefix筛选后的采样分布称作原先question-only分布。

最多128000个新prefix token只是生成token上限；prefill、冻结encoder特征前向、KV保存/重算和调度成本分别记录。现有服务缺“5token暂停→评分→同prefix续写”完整合同，需独立实现；不能仅改max_tokens=5就声称已有正确续写。PPO目标/选择分布处理作为接入前的明确实现项，不在离线A/B中悄改actor/PPO配方。

本阶段完成离线A/B后交付结果，正式200轮由主线依据后续人工决定统一启动。旧3402不自动恢复，不新启正式actor训练。

## 7. 代码、资产和边界

本线程独立worktree：`/Users/shaoshitong/Documents/Search-G1/qprobe-prefix5-ab-h1008-20261016`，分支`codex/qprobe-prefix5-ab-h1008-20261016`，基础源码c5fc0960保留。参考线程：`codex://threads/01a0fb1b-d3ff-74b3-aa13-49ca57871f92`，参考本地`qprobe_token_features.py`的layer27约定、原生block full-copy和DataProto行为记录；新原IDprefix抽取、DDP及单BCE接口须独立实现，不能直接复用裸question tokenizer入口。

模型、head/optimizer、特征、原始轨迹和全量预测留远端；Git/Mac只代码、配置、本文和小型指标/哈希摘要。保留旧run及完整只读teacher缓存，不因本次停训删除它们、不占用或停止其他人任务。

当前交付：方案与3402限定停止控制。未实现/未运行新A/B，不冒称显存实测或正确率收益。已发布的旧restore-only38及archive fba36501保留历史，本方案取消旧实验继续200的安排，不自动执行该验收或旧canonical后续作业。

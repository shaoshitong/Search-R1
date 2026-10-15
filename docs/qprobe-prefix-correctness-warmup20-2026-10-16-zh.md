# 轨迹前缀正确性探针：20 epoch 预热与固定新512题测试

日期：2026-10-16（Asia/Shanghai）。这次只训练独立探针；原正式 actor 作业继续人工停止。

## 输入和监督

每个样本使用已保存轨迹的实际 prompt，以及该轨迹最前面至多5个 policy 生成 token。按原 `sampling_mask` 定位第5个 policy token，在该位置截断完整实际上下文；此前已经出现的检索观察保留，之后的 token、观察和最终答案全部排除。不 decode 再 tokenize，不重新生成原 teacher 轨迹。实际不足5个时保留真实短前缀，禁止补造 token。

冻结官方 Qwen/Qwen2.5-3B-Instruct（revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`），提取原生 layer27 的全部有效 token 表示，detach 后送入复制官方 layer35 参数的原生 `Qwen2DecoderLayer`，再补零、展平、经过全新单输出 Linear。block、RoPE、因果和 padding mask 使用 Transformers 原生实现，保留实际 position IDs。

唯一目标是该条轨迹最终 strict EM 的正确1/错误0。使用逐轨迹 `BCEWithLogits` 均值，无标准差输出、方差损失、六分类、类别重权或排序损失。前缀在 encoder forward 之前截断，最终答案只作为监督标签，不能进入特征。

全局补零长度从训练和固定测试输入形状扫描确定；这是不使用标签的形状检查，不能沿用原 question-only 的244截断假设。所有有效 token 都保留，超过已绑定长度则报错。

## 数据和计数

- 训练：原合格 teacher 的5,120题，每题5条，共25,600条轨迹。原缓存逐块 SHA 和模型、协议身份核验后使用。
- 测试：新随机512题，每题5条，共2,560条；与训练题及此前正式候选题按问题身份和内容排除重叠。随机种子2026101601，固定本次已生成的G5，不在每个epoch重新生成。
- 原 teacher 和新测试标签均来自同一官方冻结模型、原生成与检索协议。测试题只评估，不参与梯度、采样训练、阈值调节或早停。
- 最新人工要求动态均衡采样：每个batch精确128条正确、128条错误，从训练两类池内分别均匀有放回抽样，再打乱批内顺序。测试题不参与采样。每epoch100个batch，即25,600次样本抽取；有重复，不冒称每epoch逐一遍历全部训练轨迹。
- 20 epoch，batch256、micro32精确累积8次、Adam LR3e-4、weight decay0、seed1729；每epoch100次，合计2,000次探针Adam。单BCE不另加类别损失权重，actor更新始终为0。记录每epoch两类各12,800次抽取、实际唯一轨迹数、重复数、抽样身份SHA和私有RNG状态。
- 原 mixed-question Head200、其 optimizer/RNG 和原正式 actor checkpoint均不继承。新 block 从官方复制，单输出 Linear、Adam和私有RNG全新。

## 每轮记录

在第0轮（训练前）和每个完整epoch1..20后，分别评估全训练集25,600条和全测试集2,560条。保留原分布下BCE、二分类准确率、正确/总数、预测正确比例、真实正确比例、混淆矩阵和多数类常数基线，另列平衡准确率 `0.5*(正确类判对率+错误类判对率)`；两类各50%权重，全判一类的平衡准确率为50%。原始输出阈值固定0.5。

均衡采样的sigmoid输出表示均衡先验下的正确概率。设完整原训练集正确比例为 `pi`，另报告 `corrected_logit=raw_logit+log(pi/(1-pi))` 得到的概率与完整BCE/准确率，用于区分自然分布正确概率。该解析校正依赖类别条件分布稳定与模型拟合假设，不保证自动校准；不使用测试集正确比例来拟合校正。

保留所有20轮结果，最终模型固定epoch20；最佳测试epoch仅作为曲线描述，不据此选参或提前停止。

新测试集单轨迹正确559/2560=21.8359375%，所以全判错误的准确率为78.1640625%。新单轨迹正确性准确率不能与原题级“是否混合”的准确率直接横比。

## 原探针对照实测

原 Head200 在原5,120题固定G5标签上复现5114/5120=99.8828125%；在本次新512题上题级混合二分类准确率301/512=58.7890625%，BCE1.0598413683474064。新512题实际混合183/512，全判不混合的基线329/512=64.2578125%。相同新输入下原预热权重与正式初始权重预测最大差异为0。

这证明原训练准确率不能代表新题泛化；这份对照并未证明前5个token必然提供足够的正确性信息。新版是否有效，以逐epoch固定测试实测为准。

## 执行入口

新增外部入口均发布后再部署，原 `fcb77204f48d5cb8ec21c68df7d4a2c937b3557a` 科学runtime不热改：

- `reproduction/operations/qprobe_prefix_correctness_features.py`：核验原rollout和标签，按实际前缀重新前向编码并生成远端packed BF16缓存。
- `reproduction/operations/qprobe_prefix_correctness_fit.py`：原生block单BCE预热和epoch0..20固定测试。
- `reproduction/operations/a800_qprobe_prefix_warmup20.py`：均衡版沿既有物理SSH路由一次stage、一次run-once和有界observe。`--reuse-features-from`绑定已完成的同一前缀编码缓存，禁止重新生成teacher或重复编码。仅实际空闲GPU0通过UUID准入后运行；原编码进程退出释放，再启动均衡fit进程。

原编码产物保留远端 `/data/shaoshitong/Search-G1-qprobe-prefix-correctness-warmup20-<id>`，均衡版新Head/Adam/私有RNG及结果保留独立 `/data/shaoshitong/Search-G1-qprobe-prefix-correctness-balanced20-<id>`。旧未均衡fit未启动；按最新指令只冻结其supervisor，保留编码继续，完成后仅退出该旧supervisor，不热改活跃源码。Mac和公开仓库只保存代码、配置、说明及小型scalar/hash记录。未知SSH结果通过同一handle观察，禁止盲目重复launch。

本次不启动正式200 outer，也不恢复原30分钟自动训练监控。若后续要用前缀选择正式轨迹，必须继续同一已采样前缀，并明确选择改变采样分布的科学合同；这不属于当前离线预热范围。

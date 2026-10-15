# Question 加生成前缀的成功概率探针

日期：2026-10-16，Asia/Shanghai。用户已认可用单条成功概率p替代固定G5混合类别；本次进一步分析question加每条轨迹前5个真实生成token的输入。已完成数学与静态接口设计，没有修改或重启另一线程的实验。

## Target

用每条已有轨迹的短生成前缀预测该轨迹未来最终严格EM；说明它与问题级成功概率的区别、训练数据构造、短前向需求及选题部署方式。避免把条件概率错误地用于另一条重新采样的轨迹。

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION。**

监督目标可成立。增加前缀后预测的是条件成功概率，不再只是question-only p。它可能提供信息，不能保证5个token带来显著提升。利用它计算G5混合概率需独立续写假设；现有系统未实现暂停5token后筛选再续写的接口。

## Invariant Object

固定actor、采样/检索/评分协议，看到真实前缀后，单条轨迹最终答对的条件概率。选题的混合概率是其下游用途，不再直接作为唯一训练标签。

## Assumptions

- 标签C是该条完整真实轨迹的严格EM0/1，输入为该条轨迹自己实际生成的前5个token；禁止错配兄弟轨迹或给每个前缀复制group mixed标签。
- 原始prefix ID和完整第一轮prompt可从已有保存契约获得；本次未读取远端原轨迹逐项证明完整资产可用，实施前仍需远端清单核验。
- 表征抽取模型、层、prompt、tokenizer、position/mask、dtype、actor版本明确绑定；固定encoder表征和实时actor表征是不同实验，不混称。
- 输入没有gold、后续检索结果、未来回答；抽取因果hidden，不让未来token进入表示。
- 推导条件G5 mixed概率时，给定题目和五个前缀，各条未来续写独立；这不是仅凭代码结构就已验收的事实。

## Notation

- x：问题及完整初始prompt上下文；θ：产生前缀和续写的actor版本。
- U_j=(y_{j1},…,y_{j5})：第j条轨迹的真实前缀；不足5且已终止时保留实际长度与终态。
- C_j∈{0,1}：该条完整轨迹的最终严格EM。
- h(x,U)：明确版本encoder对prefix的表示；φ：可训练的预测头参数。
- fθ(x,U)=Pθ(C=1|x,U)：真实条件成功概率；\(\hat f_\phi=\sigma(z_\phi(h))\)：预测值。
- pθ(x)=Pθ(C=1|x)：尚未观察前缀的问题级边缘概率。
- q：未来G5出现正确与错误两种结果的概率。

## Derivation Strategy

定义条件概率 → 用每轨迹EM做BCE → 由条件期望连接问题级p → 区分已见五前缀的条件G5与未知前缀的新G5 → 构造短前向数据 → 先离线检验，再选择部署采样目标。

## Derivation Map

1. 前缀提供了比只看问题更细的状态信息，训练目标成为f(x,U)。
2. 25,600条轨迹各有一个C标签；同题五条不再共用同一个输入。
3. 在总体上BCE最优预测为条件成功概率；不要求每题重复采样混合频率。
4. p(x)=E_U f(x,U)，已见五前缀则分别有五个p_j。
5. 利用独立续写的补事件公式计算混合概率，不能随意用平均概率的五次方代替乘积。
6. 当前只有token ID、无prefix hidden缓存；短teacher-forced前向可产生新表征。
7. 两种上线用法对应不同采样目标，必须明确是否继续原前缀。

## Main Derivation

### 1 每条轨迹训练一个正确性概率

每题五条完整轨迹形成五个监督样本：

\[
\big(h(x,U_1),C_1\big),\ldots,\big(h(x,U_5),C_5\big).
\]

主loss只有稳定二分类正确性BCE：

\[
\boxed{L(\phi)=-\frac1{5N}\sum_{i=1}^N\sum_{j=1}^{5}
\left[C_{ij}\log\hat f_{ij}+(1-C_{ij})\log(1-\hat f_{ij})\right].}
\]

实现为BCEWithLogits(logit, strict_em)，不在外面先sigmoid再对数，不再用group mixed的0/1替代单条C。5120题有25600条轨迹标签，但仍只有5120个unique qid，兄弟样本共享题目；不是25600个独立新问题。

给定(x,U)，BCE的期望为 \(-f\log\hat f-(1-f)\log(1-\hat f)\)，最小值在\(\hat f=f\)。一次观测可用作训练，不要求先构造同题多次G5频率。给定前5token后未来仍然随机，标签不变成确定答案，模型也仍可能记忆训练样本。

**与上一版count loss的关系。** question-only时五条输入和logit相同，题内平均BCE精确等于BCEWithLogits(z(x),K/5)。加不同U_j后logit不同，不能再把K/5复制到每个前缀；需要保留逐轨迹C_j。取消重复G5软频率不是前缀才带来的数学资格，上一版count likelihood本来也不要求重复G5。

### 2 能改善多少是待验问题

理想可使用完整信息(X,U)时，Bayes Brier风险满足精确恒等式：

\[
E[(C-p(X))^2]-E[(C-f(X,U))^2]
=E[(f(X,U)-p(X))^2]\ge0.
\]

这说明前缀有额外判别信息时可以降低不可约误差；不能推出有限数据、压缩表征或当前78M参数模型一定更准。若前5token主要是固定<think>标签/换行，新增信息可能很少。需比较question-only、+5、+16，而且三个版本都预测同一个单条C；不能拿旧mixed类别准确率作直接对照。

终奖励仅为EM0/1、无折扣时，f(x,U)数学上接近critic对这个前缀状态的value预测；此处只训练选题探针，不自动引入GAE或通过该预测头反传actor。

### 3 已看到五个前缀时怎样计算混合概率

令 \(p_j=f(x,U_j)\)。给定这五个前缀并独立继续，各条不必同成功率：

\[
\boxed{q(x,U_{1:5})=1-\prod_{j=1}^5p_j-\prod_{j=1}^5(1-p_j).}
\]

不是 \(1-\bar p^5-(1-\bar p)^5\)。例如p=[.9,.9,.1,.1,.5]时，乘积公式给.9919，而平均概率.5的公式给.9375；两者预测的条件对象不同。独立假设失效时，仅这五个边缘概率也不足以确定联合mixed概率。

### 4 尚未看到前缀时仍要边缘化

由全期望公式：

\[
p_\theta(x)=E_{U\sim\mu_\theta(\cdot|x)}f_\theta(x,U).
\]

若未来完整轨迹独立同分布，则未知前缀的新G5仍有 \(q(x)=1-p(x)^5-(1-p(x))^5\)。一次已观测前缀的f(x,U)不等于p(x)。只采一个共同前缀再分叉五次的G5，也不同于独立五个完整rollout；该设计属于共享前缀采样变更。

此外，已见前缀后未来C的条件方差是p_j(1-p_j)。它不能直接冒称无条件G5的原样本std期望。单p方案可删除旧独立std回归支路；若仍报告group variance，应明确新的统计对象和假设。

### 5 已有数据怎样变成representation

已有teacher raw output的保存契约含behavior_turns、每轮actual_request_token_ids、raw_generated_token_ids、逐轨迹身份及strict_em向量；不必重新生成完整回答或重新搜索。不能从postprocessed文本decode再encode冒充原ID。

当前question cache只有冻结encoder对question的layer27输出，没有新前缀hidden；生成捕获也未返回hidden。因而当前实现下需额外一次**短前缀特征前向**：

```text
原第一轮完整prompt ID + 该条轨迹真实y1..y5
    → 因果encoder forward
    → 取包含生成token位置的hidden
    → 缓存特征及其身份
    → 后续多个head epoch复用
```

不需要整条trajectory再次forward，不需要重新采样这五个token。已有同版本、同输入和同层hidden时可直接复用，但当前契约没有这项保存。

两个关键因果对齐点：

- 必须让head读取新增token的位置或处理完前缀后的状态。因果模型中原question位置看不到后面的5token；只追加文字却仍取旧question位置表示不会获得新增信息。
- 生成y5时用于预测它的hidden只见过y1..y4；要得到处理完y5的状态，需要将y5作为输入消费。短前缀forward直接做到；缓存增量路径也可做到，不要求再生成y6。[HF生成输出文档](https://huggingface.co/docs/transformers/v4.45.2/en/internal/generation_utils)、[Qwen接口](https://huggingface.co/docs/transformers/v4.45.2/en/model_doc/qwen2)。

若未来直接从actor解码服务捕获hidden，可省去额外的完整prefix重算，但需要新接口及精确位置/版本绑定；不能把actor hidden当作固定encoder hidden的等价替代。当前系统没有对应合同。

缓存key至少绑定actor snapshot、trajectory ID、prompt/prefix token SHA、prefix实际长度/终态、encoder权重/层/dtype/模板和label scoring contract。现有仅qid的question cache不能直接混用；原244的max length也不覆盖完整chat prompt+5，不能悄悄截断。

### 6 选题时两种不同用法

**A 根据已经生成的五个前缀筛选，再继续同一前缀。**每题先生成五个独立短前缀，预测五个p_j并用乘积q排序；选中的题沿这五个原前缀继续，不重置actor、检索/turn状态或token预算。首5token和后续token都保存真实行为记录。early EOS/完整行动终态按原协议处理，不用工具文本补足5token。

继续原前缀是条件预测与执行匹配的必要条件，但不消除选择偏差。选择事件S依赖实际生成U，故选择后前缀分布变为 \(\mu(U|x,S)\)，一般不等于原 \(\mu(U|x)\)。不能称只是question-only课程变更、或原PPO逐token行为概率已经表示完整选择后分布。这是新的prefix-aware采样方案，需要单独验收其目标和偏差。

当前异步服务只处理finished输出，忽略unfinished，没有5token pause→读取hidden→筛选→保KV继续的合同。直接把max_tokens改5后重启生成，也需核终止/forcedEOS/剩余预算，不是现成的等价恢复入口。

**B 前缀仅作问题级p的Monte Carlo估计，入训完整轨迹另采。**用独立pilot前缀估 \(\hat p(x)=M^{-1}\sum_m\hat f(x,U_m)\)，按g(\hat p)选择题，再用新的独立随机流生成原协议五条完整训练轨迹。给定所选x，训练轨迹可保持原μ分布；但有限M的g(mean f)存在非线性估计偏差，原条件乘积q不能直接当新组的概率，pilot也有额外成本。

不能混用A的条件q与B的执行方式，声称丢弃前缀重新采样仍由原条件预测保证。第一阶段仅检验f，不自动选择A或B，也不修改PPO。

### 7 成本与验证

按5120候选、每题5前缀、每前缀5token计，筛选阶段会生成128000个prefix token并处理25600条短请求；最终只完整生成512×5=2560条。生成token数只是成本的一部分，prefill、固定encoder再编码、KV保留/重算、调度均需实测，不能据“只有5token”声称免费或固定倍数加速。

最低可信比较：

1. 远端核已有25600轨迹的prefix/EM身份是否齐全；只生成小型审计到Mac，不搬特征或原rollout。
2. 按qid先划train/dev/test，同题五条全部同一split；去除重复/近重复题。原新随机512诊断保持独立，不因本方案自动加入训练。
3. 固定actor/encoder，同一单条C目标比较question-only、+5、+16；按同预算head训练和dev proper loss选checkpoint，报告BCE/Brier、分类accuracy和常数正确率基线。25,600是轨迹数，不把兄弟样本当25,600独立题估CI。
4. 检查新BCE头的raw logits、概率饱和、梯度和实际参数变化；旧mixed输出语义不同，不能将原99.88%的分类头及Adam直接当新p头继续。
5. 固定actor验证通过后再比较A或B的在线收益、时间/显存与同预算随机选题。在线标签绑定采样前actor版本，先预测后发现C，再拟合，避免future-label泄漏。

## Remarks and Interpretation

方向合理，新增信息是早期轨迹状态；不需要同题多次G5的混合频率作为训练目标。5token后的未来仍有不确定性，增加输入也不会自动修好过拟合。当前标准差支路全零是另一个已证训练失败；删除它改用单条EM-BCE是明确配方变更，不是声称原回归已修复。

建议先完成已有轨迹上的离线单p对照，验证prefix确实提供增益，再决定是否承担在线短前缀筛选的接口和采样变更。此次方案不会重复启动另一线程的新512诊断或正式训练。

## Boundaries and Non Claims

未运行特征encoder或probe，未触训练节点，未下载模型/缓存/原rollout，未热改源码或启动新训练。代码保存契约不等于远端原始资产已逐项验收；准确率提升、成本和pause/resume正确性均未实测。本文条件概率与独立性推论不是端到端EM提升保证。

## Open Risks

前5token信息量、模型容量与泛化、同题样本依赖、actor和encoder漂移、真实token/position对齐、解码停止与RNG续接、prefix筛选改变行为分布、KV和prefill开销。

分项静态来源见 `source-notes.md`、独立数学见 `math-notes.md`、接口与评估审查见 `review-notes.md`。

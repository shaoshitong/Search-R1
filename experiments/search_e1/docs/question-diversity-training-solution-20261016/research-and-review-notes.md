# Question-diversity probe：原始来源与反向审查

2026-10-16。范围是概率目标、随机标签、泛化划分和选择后的评估；只读4份原始学术来源并做本文数学展开，没有访问训练节点、下载模型/轨迹/环境或修改训练。使用 deep-research 的 quick/fact-check 范围及 paper-glance 方法阅读框架。下述99.88%/50.98%、模型结构和actor状态来自本轮任务说明，本笔记未独立重跑其审计，不将其升级为新增实测。

## 先纠正比较对象

“固定一次G5标签训练分类准确率99.88%”和“后来选出Top512题，fresh G5实际mixed比例50.98%”是**不同估计量**：前者衡量对一份已见标签的拟合，后者是一个选择策略的新采样产出率，通常不是独立分类测试accuracy。不能相减称48.90个百分点泛化gap，也不能仅凭50.98%说排名失效；需要当前actor、同候选池下随机512的产出率，以及选中题预测概率的均值/校准误差。

若Top512全部被分类为mixed，则其产出率碰巧等于该子集precision，但仍不等于全候选池accuracy。应保存逐qid的预测时点、actor/权重身份、选题前分数和fresh结果，并区分训练集、选择集和独立审计集。

## 仅保留4份直接相关的原始来源

1. **Gneiting & Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation，JASA。**作者大学主页的[原始PDF](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)，§3及Examples1/3给出quadratic/Brier与logarithmic scores；properness的对象是预测分布的期望评分，非某一次标签的100%分类准确率。用于选择概率训练/评价目标，不保证有限高容量模型在固定数据上学到真实条件概率。
2. **Guo et al. (2017), On Calibration of Modern Neural Networks，ICML。**[官方PMLR页面](https://proceedings.mlr.press/v70/guo17a.html)、[原始PDF](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)，§2、§3及Figure3区分accuracy、NLL与calibration，观察到高容量模型可继续降低训练NLL并增大置信度，而概率质量不随accuracy同步改善。校准需独立holdout；该论文常规校准假定train/validation/test同分布，不能证明旧actor上的temperature scaling可修复新actor漂移。
3. **Li, Chu, Langford & Wang (2011), Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms，WSDM。**[作者原始arXiv/PDF](https://arxiv.org/pdf/1003.5956)，§3及Theorems1/2以随机logging和明确支持条件构造可靠policy评价。它揭示只观察已选行动反馈的难点，支持保留随机探索/审计数据；不是我们TopK系统已经满足其定理，也不是deep encoder可自动消除选择偏差的证明。
4. **Kendall & Gal (2017), What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?，NeurIPS。**[官方会议页](https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html)、[原始v2全文](https://arxiv.org/html/1703.04977v2)，§§2–3区分观测随机性aleatoric与模型参数未知epistemic。本文只借该区分，不把视觉任务的Bayesian/dropout效果当成当前probe效果；一个sigmoid值或一次label翻转都不能单独识别这两种不确定性。

以上是定向4-source核查，不冒称系统综述；没有用二手博客或仓库star替代证据。后续推导和实验建议是本任务的独立数学分析，不是这些论文已经验证的question-probe方案。

## 应学习哪个概率

令 (C_{q,a,r,j}\in\{0,1\}\) 为题q、actor快照a、fresh第r组的第j条原协议严格正确性；令

\[
Y_{q,a,r}=\mathbf1\{0<\sum_{j=1}^5 C_{q,a,r,j}<5\},
\qquad d_a(q)=P(Y_{q,a,r}=1\mid q,a,\mathcal P).
\]

\(\mathcal P\) 绑定temperature、检索、预算、评分和独立随机源。**mixed是题×actor×协议的随机事件，不是qid永久标签。**固定teacher特征 (x_q\) 只保证输入表示稳定，不能保证 (d_a(q)\) 随formal actor训练不变。

对概率预测s，单个fresh标签的BCE为

\[
L(s,Y)=-Y\log s-(1-Y)\log(1-s),\quad
E[L|q,a]=-d\log s-(1-d)\log(1-s).
\]

求导得 \(\partial E[L]/\partial s=(s-d)/(s(1-s))\)，内部唯一最优为s=d；端点按极限理解。由BCE一致性不能推出有限单次标签集的训练accuracy应该接近100%。当每qid只给一次G5，容量足够的模型仍可把这些实现值记成0/1，从而训练99.88%，同时对新实现值明显过度自信。这种随机标签记忆本身就是需要排除的过拟合机制，不能用“本来有噪声”替它免责。

若同题同actor有m个独立fresh G5，k组mixed，Binomial条件似然等价于

\[
-k\log s-(m-k)\log(1-s)
=m\,BCE(s,k/m).
\]

是否让多次采样题具有m倍权重必须预先明确：目标是等qid总体时可题内先平均再等题平均，不能让TopK重复曝光悄悄改变目标数据分布。重复采样能减少估计d的误差，不会让下一次Y变确定。

同题两次独立G5在固定d下，标签翻转概率为 \(2d(1-d)\)。d=.5时即50%翻转，所以翻转不是模型必然错的证据；相反，一个预测.5的校准模型对随机硬标签只能约50%分类正确。仅有两组结果又不足以精确估出d，不能据此断言全部gap均由不可约噪声解释。

### 0.9375不是普适上限

只有同组5条**条件IID、同成功率p**时

\[
d(p)=1-p^5-(1-p)^5\le1-2(1/2)^5=15/16=.9375.
\]

独立但不同 (p_j\) 时应为 \(1-\prod p_j-\prod(1-p_j)\)，可以达到1，例如固定一条正确、四条错误；存在相关时也可超出.9375，或完全同步而mixed概率为0。因此不得用.9375统一裁剪经验概率、宣称模型已达理论最优，或把50.98%除以.9375当“已完成54%能力”。应先实查随机种子/检索共享/生成相关性，再决定是否值得使用p→d的结构化模型。

## 不确定性、过拟合和actor漂移必须分开

在一个假定有参数后验w的Bernoulli概率模型中，总预测方差可展开为

\[
Var(Y\mid x,\mathcal D)=E_w[d_w(x)(1-d_w(x))]+Var_w(d_w(x)).
\]

第一项是固定模型下的新G5随机性，第二项是对条件概率估计的模型不确定性。确定性lastblock+flatten244+linear只输出一个概率，并未自然给出第二项。不同训练seed/ensemble disagreement可以作模型敏感性诊断，不能不经校准就叫真实posterior epistemic variance。

actor变化属于目标分布非平稳，不应一并塞进“aleatoric噪声”。同题在teacher/旧actor下的label和新actor下的label不一致，可能是d真的变了；更强的拟合器未必能从固定teacher特征推断这个变化。应先量化跨actor可迁移性，再考虑current-actor侧特征、显式actor状态或在线刷新；这些均是新的模型/数据变量，不能与降容量合并后归因。

## 最小可信验证协议

| 检验轴 | 数据怎样切 | 回答的问题 |
| --- | --- | --- |
| 固定实现拟合 | 旧的一次G5原样保留 | 能否拟合已见标签；只作优化/容量sanity check |
| 同题新实现 | train qid可重合，**fresh G5及seed不重合**；固定同actor | 学到d还是记住那次Y；用此验证early stopping |
| 新题泛化 | 以qid分割train/dev/test，重复组全部跟随qid；规范化/近重复检查 | 能否推广到未见题，而非问题指纹记忆 |
| 时间外/actor外 | 在较晚当前actor上复采，测试label在预测时不可见；明确零shot或允许多少refresh | 固定teacher特征对current actor的迁移与漂移 |
| 选题收益 | 同actor/同候选池/同5条预算下，事先冻结Top512与uniform-random512比较 | ranking是否比随机多产mixed，不等同分类accuracy |

不能只随机拆所有重复行：同qid/相同teacher表示出现在两边会高估新题泛化。反之，只做qid holdout不能回答是否预测对已见qid的新G5，必须把两个轴分别报告。所有校准、模型容量/weight decay/early stop选择只用dev，最终random test不再反馈给probe或E0筛选。

还应核对特征只含预测时真实可得的题目与允许的actor状态，不含这次待预测G5的答案、correctness、参考解或未来检索结果；否则会把结果信息泄漏误当question diversity预测。若任务确实允许这些信息，须重新定义预测时点和baseline，不能沿用采样前选题的结论。

为了打断Top512 E0-selected反馈，应保留按qid总体随机抽取、在结果产生前独立指定的审计slice。训练可混合exploit和random探索，但记录每题被选概率及策略版本；deterministic TopK对未选题概率为0，不能靠事后importance weighting恢复全总体风险。选中集的fresh产出率仍是该选题策略的有效直接结果，只是不能推广为全体校准。没有已知正支持/概率时优先明确随机audit，不强造IPS修正。

## 报哪些数，而不是继续追训练99%

1. 独立fresh测试主报等qid **Brier** \(mean(s-Y)^2\) 与 **log loss**，辅助报hard accuracy、AUROC/AUPRC（单类时明示undefined）、预测概率分箱与实际mixed率；每箱同时给n和区间。严格使用mixed事件概率校准，而非把分类置信度和成功率混在一起。ECE受分箱和小样本影响，不独占决策。
2. 预先固定baseline：train-only总体mixed率常数、必要的dataset分层常数、较小/正则probe；Top512必须与uniform-random512及预先指定排序规则比较。只与旧Top512自己比较会混入actor变动。
3. 选题结果主报 \(\hat U_S=512^{-1}\sum_{q\in S}Y_q^{fresh}\)、与random的差值/比值和95%CI，重复q时按qid聚类/题内聚合；整个G5是一个label单位，不能拿2560条当2560个独立mixed标签。并列报告总生成token/搜索/walltime成本，避免只优化mixed数却使单位GPU成本下降。
4. 如果50.98%准确来自261/512，在独立Bernoulli近似下Wilson95%约[46.66%,55.28%]；这不是本次新实测，也不是考虑qid重复/环境相关后的最终CI。缺少同条件random收益及其方差，当前不能宣布显著优于随机或不如随机。
5. 当前actor snapshot冻结后可先评分同一候选池，再各选Top/random并在相同预算下fresh采样；交集qid可共用同一次fresh结果并用配对/设计相符的差值估计，重复的多组按qid cluster。提前固定K、时间点和主指标，不能看结果后挑K或最有利seed再给未经校正的CI。

## Devil's advocate：必须挡住的结论跳跃

- **“能拟合99.88%，说明representation已经足够预测未来。”**只能说明这一组标签被表达；高容量固定特征亦可记忆。新qid、新实现和时间外必须分别通过。
- **“同题会翻转，所以低fresh结果不是过拟合。”**随机性与memorization可同时存在；heldout proper scores才区分概率学习与极端置信度记忆。
- **“Top512 50.98%已经很高/很低。”**没有random同actor基线、预测概率均值和CI，无法判相对排名价值。
- **“冻结teacher就没有distribution shift。”**输入特征固定，formal actor的 (d_a(q)\) 仍可变；训练时E0筛选还改变观测qid分布。
- **“加ensemble/UCB或soft labels就修复了。”**这些是待验证设计；未经探索支持、actor外校准和samebudget对照，不能保证有效。先用新的同题实现与随机audit验证，不能只在现有标签上调参直到又99%。

建议判据：固定实现99.88%保留为sanity proof；成功标准改为独立fresh/actor外Brier或logloss改善，并在同预算current-actor Top512-vs-random中得到预先定义、带CI的mixed-yield增益。当前证据支持“原验证目标不足以判断下一批排名”，尚不足以把退化唯一归于容量、噪声或actor漂移。

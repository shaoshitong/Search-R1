# Question + rollout前5token：单答案概率探针核对

日期：2026-10-16。仅数学与本地标量synthetic；未访问远端、未模型前向、未启动训练，也未编辑Search-G1原package。

## Target / Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION。** 新对象是“给定实际前缀后，这一条续写最终严格答对的概率”，不是问题本身的确定正确性，也不是直接拟合五条count。需要把已看到prefix后的条件概率，与尚未采样prefix时的问题级概率分开。

## Invariant Object / Notation

- X：问题及固定prompt；θ、c：冻结actor与完整生成/检索/评分协议。
- U：某条真实轨迹前5个实际生成policy token，以及恢复续写所需的全部已发生状态。若5个token跨工具/turn边界，不能只保存字符串而丢失真实observation、budget和processor状态。
- C∈{0,1}：该条完整轨迹的strict terminal EM，模型协议非法沿原规则为0；infra缺失不补0。
- \(f_{\theta,c}(x,u)=P(C=1\mid X=x,U=u,\theta,c)\)：prefix-conditioned single-answer概率。
- \(p_{\theta,c}(x)=P(C=1\mid X=x,\theta,c)\)：question-only边缘正确率。
- \(q\)：五条中至少一对一错的mixed概率。

概率中的随机量包含未来独立采样流及协议定义的外部随机性。若固定同种子/确定性执行原样重放，不应当作新独立续写。

## Assumptions / Derivation Map

1. 条件BCE与Brier投影恒等式不要求五条iid。
2. 五个prefix条件下的乘积公式，需要五条未来续写给定各自完整state后条件独立。
3. question-only的单一p五次方公式，进一步需要五个prefix独立同分布及相应续写独立、同actor/协议。
4. finite probe/feature可识别性、优化是否健康及策略漂移均另验；下述Bayes结论不保证实际神经网络改善。

## 1. 单答案监督与question-only概率

每条轨迹形成一条监督：

\[
\ell_j=\operatorname{BCEWithLogits}(z(x,u_j),C_j),\qquad
\widehat f=\sigma(z).
\]

固定x,u时，条件总体BCE唯一最优输出是f(x,u)。对prefix分布取期望，**恒等式**为

\[
\boxed{p(x)=E_{U\mid x,\theta,c}[f(x,U)].}
\]

仅有question输入时，不能直接查询f(x,u)；必须积分/采样prefix，或者另训练question-only p头。若只给有限representation h(x,u)，最优输出为P(C=1|h)，只有h足够表达时才等于完整状态条件概率。

5120题×5条=25600个trajectory标签，unique question仍5120。五条原EM本来就存在，新方案是利用不同prefix特征展开监督，没有凭空新增五倍独立问题。所有qid的五条完整时，先题内平均BCE再题均值与25600行均值相同；缺条/重复次数不同则须明确权重。评估应按qid拆分，不能把同qid不同prefix随机分到train/test后称为跨题泛化；统计误差按qid聚类，不能默认25600行完全独立。

## 2. 五个已实现prefix的mixed概率

给定 \(u_1,\ldots,u_5\)，令 \(p_j=f(x,u_j)\)。在未来续写条件独立的假设下：

\[
\boxed{q(x;u_{1:5})=1-\prod_{j=1}^5p_j-\prod_{j=1}^5(1-p_j).}
\]

这对应Poisson-binomial count，不应以 \(\bar p=\sum_jp_j/5\) 替换为 \(1-\bar p^5-(1-\bar p)^5\)。AM–GM反而给出条件乘积公式≥均值代入式；异质性可能使同一组更容易同时有对有错。

**本地toy：** \(p_{1:5}=(.9,.9,.1,.1,.5)\)，均值.5，两个全同结果的概率均.00405，所以mixed概率 **.9919**；均值代入得到 **.9375**，差.0544。

若还需预期组内sample std，应先用

\[
P(K=k)=[z^k]\prod_{j=1}^5[(1-p_j)+p_jz],\quad
E[s]=\sum_{k=0}^5P(K=k)\sqrt{k(5-k)/20}.
\]

不能把 \(\sqrt{\bar p(1-\bar p)}\) 当E[s]。上例E[s]=.5276367594、E[s²]=.282，二者平方不相等。

**重要边界：** 若未来5个prefix本身独立同分布，且各自续写也独立，那么在看到prefix以前，边缘 \(C_j\) 仍iid Bernoulli(p(x))；对乘积式积分后，question-only q依然精确为

\[
q(x)=1-p(x)^5-[1-p(x)]^5.
\]

所以“不能用均值五次方”针对**已经看到的异质prefix组**，不是否定上述明确iid设计下的question-only边缘公式。

## 3. 同一prefix续写5次 ≠ 5个独立prefix各续写一次

同一prefix u复制后，若未来五次续写独立，条件mixed为 \(g(f(x,u))\)，其中 \(g(p)=1-p^5-(1-p)^5\)。若每题只抽一次U再共享给五条，无条件mixed为E[g(f)]。

五个prefix独立采样则为g(E[f])。由于g严格凹，

\[
E[g(f)]\le g(E[f]),
\]

并非同一课程/rollout设计。例：prefix各有一半概率使f=.9或.1，边缘p=.5；共享一个prefix再续写5次的mixed概率为.4095，而独立五prefix为.9375。共享prefix还会改变GRPO组的共同token结构，不可在声称其余不变时静默采用。

## 4. prefix信息为何可能帮助，但不保证有限模型更准

令question-only的Bayes预测为p(X)、prefix Bayes预测为f(X,U)。**不需iid的精确Brier分解**：

\[
E[(C-p(X))^2]-E[(C-f(X,U))^2]
=E[(f(X,U)-p(X))^2]\ge0.
\]

这表明可用prefix信息在Bayes最优层面不会增加Brier风险；它没有保证有限probe、有限训练集或错误feature extraction的实际提升。若前5token几乎都是相同格式前缀，右边可能接近0；若高容量模型继续记住qid，更多行也不会自动获得泛化。

toy f=.9/.1等概率时，question风险.25、prefix风险.09，减少.16。C在给定prefix后仍有f(1−f)随机性，不是看5token就知道未来答案。Bayes分类accuracy也不会因额外信息变差，但可能完全不变，例如f=.6/.8时两个都在同一分类阈值一侧；有限模型accuracy没有该保证。

当前std全零的训练健康问题不能被这个Bayes信息论解释替代。改为单答案BCE可不再依赖独立sigmoid-std MSE支路，但仍需验真实参数/梯度更新与fresh proper score，不能称已修好原优化问题。

## 5. 是否需要重新forward：特征与采样分开

- 数学不要求“重新生成轨迹”。已有完整轨迹提供C及前5个token IDs，标签可直接复用。
- 但token IDs/count/logprob并不包含隐藏representation。若生成时未保存同encoder、同层、同pooling、同精度/协议的可用hidden，需要对 **question＋所需前缀** 做一次短teacher-forced/prefill前向提取特征；不必重新跑完整续写或全部长轨迹。
- 若原生成过程已能安全取得该prefix位置的匹配hidden，可以复用，不再额外encoder前向；这取决于真实采样服务接口，不能先声称零额外forward。
- 提取必须因果：只读prefix末位置/已发生context，不从完整轨迹末位置或全响应pooling泄漏未来。历史actor、feature encoder与目标actor分别绑定；后期actor label不冒充初始actor真值。
- 固定prefix特征encoder与当前actor生成prefix是可定义的设计，但f的策略条件仍会漂移。即使question相同，tokenizer、template、层或encoder变动也需独立feature命名空间。

## 6. prefix筛选必须兑现同一个条件事件

若把f(x,u)或五prefix的条件q作为实际续写评分，后续必须在**同一已评分prefix**上续写。不能保留这个条件概率的解释，却丢掉prefix再从question重新生成：那样下一条的条件分布变回p(x)，所选评分不再对应实际续写状态。第7节另定义允许丢弃pilot prefix、但预测对象已改成边缘p的方案B。

可以保留KV并继续，也可以从完整相同prefix重新prefill后按相同生成/processor/budget契约续写；是否能重建相同条件分布或精确复现需实现验证。原prefix随机消耗、检索observation、turn/search预算、repetition规则和终止状态均要记录；不能额外赠送5token预算或重新算满turn。已终止/不足5token的prefix须按预先规则记录，不能静默丢掉容易/难的样本来美化概率。

若候选仍为5120道question且每题先生成5个prefix，将产生25600个prefix、最多128000个初始policy token（均达5token时），外加真实prompt prefill/潜在工具成本；只选512组续写时，未选prefix仍是已消费成本。它比question-only缓存探针多了一段真实actor采样，不能叫免费或只有一个小MLP。

若改成每题只生成一个prefix，然后复制为五续写，是第3节的另一种group协议。若按单条prefix在全池选择2560条，还必须处理每题5条/512唯一题的结构约束；不能把自由选轨迹假称保留原group设计。

## 7. sameprefix只修条件匹配，不消除action-selection偏差

**方案A：prefix-aware筛选再继续同prefix。** 令S为入选事件，U可表示一个prefix或一组五prefix，V为其后续生成。在冻结选择头、S只依赖已发生prefix而不看未来C、未来随机流独立的条件下，记e(x,u)=P(S|x,U=u)，有

\[
\nu(U,V\mid x,S)=
\mu(U\mid x)\frac{e(x,U)}{P(S\mid x)}\mu(V\mid x,U).
\]

所以同prefix的未来续写条件概率仍匹配f（给定所需完整状态），但 **selected prefix边缘分布已不是原μ(U|x)**。原始μ逐token old logprob记录的是筛选前policy likelihood，不是选择后ν的完整behavior likelihood。除非S给定x后与U独立，不能称这只改了question curriculum、原on-policy PPO完全不变；共享参数和prefix的PG如何解释需另列。仅把prefix从loss mask移除也会变成给定所选context的续写RL，不自动恢复原完整轨迹目标。本笔记不声称已经给出无偏校正或授权更改mask。

**方案B：prefix只作question-level Monte Carlo pilot。** 为候选题独立抽L个pilot prefix，估计

\[
\widehat p_L(x)=L^{-1}\sum_{j=1}^L\widehat f(x,U'_j),
\qquad a(x)=g(\widehat p_L(x)).
\]

据a选question后，另用与pilot独立的新随机流从question生成五条**完整fresh训练轨迹**。给定chosen x，训练轨迹分布可保持原μ；改变的是题目课程，不能据此声称仍无偏估计原未选择题库的总体目标。pilot prefix允许丢弃，因为这里明确预测边缘p，而非把旧条件q冒充新组q。

方案B也有代价：若f正确且pilot独立，则p估计无偏，但g严格凹，\(E[g(\widehat p_L)]\le g(p)\)，finite-MC的mixed score向下偏；有限probe误差另计，不能保证相对真q偏差只有这个方向。L小还会产生排序噪声和Top选择的乐观反馈。它没有复用pilot token作最终训练轨迹，生成成本需单列；不能把同prefix续写的条件利用收益移植到这种独立fresh方案上。

两方案必须明确选择，不能一边用方案A的高条件q/续写结果，一边声称拥有方案B的given-question原rollout分布。

## Boundaries / Minimal Acceptance

1. 保存question ID、五条strict EM与count、每条前5实际generated IDs、prefix/full-state SHA、actor/protocol/feature spec、采样/续写随机流和选择前分数。
2. 先核同一prefix的因果特征与完整轨迹标签严格对齐、没有gold/未来token泄漏；训练/fresh按qid分开并另报同qid新prefix泛化。
3. 单答案Brier/logloss/校准与组mixed产出分别验收；概率相乘依赖的条件独立性不可从单个group证明。
4. 不新采样、不部署、不训练；本文件供后续独立实现审查。所有模型、prefix/rollout、tensor与全量标签留远端，Mac只留本文小笔记。

## Local Synthetic Checks

Python标准库枚举32种G5正确向量，核对上述异质prefix count分布：(.00405,.07785,.4181,.4181,.07785,.00405)，mixed=.9919；Bayes Brier分解.25−.09=.16；共享prefix/独立prefix toy=.4095/.9375。无模型推理。

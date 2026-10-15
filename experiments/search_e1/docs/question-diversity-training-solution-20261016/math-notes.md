# Question-diversity probe：随机标签、训练目标与最小修正

## Target

推导问题级探针应学习的对象、一次G5标签与重复软标签的关系，以及不依赖iid假设的二输出一致性修正。只做数学和本地小型合成计算，不访问训练节点、不运行模型、不启动训练。

当前讨论给定的实际摘要：warmup训练拟合指标99.88%；fresh Top-512得到261/512 mixed；53个原mixed题以新G5重测，16个转为nonmixed。这些来自root已读证据，本笔记没有独立远端复读。训练拟合指标与Top-512 mixed yield不是同一统计量，不能直接相减当泛化准确率下降。

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION。** 主对象必须是未来一组的条件概率，而非“一题永久是否mixed”的确定标签。iid闭式解是受限基线；主训练目标和最小二输出修正不需要iid。

## Invariant Object

固定当前actor参数 \(\theta\)、完整生成/检索/评分协议 \(c\) 及未来随机种子/批次设计分布 \(\mathcal R\)，对问题 \(x\) 定义

\[
C_j\in\{0,1\},\quad K=\sum_{j=1}^{5}C_j,\quad
M=\mathbf1(0<K<5),\quad
q_{\theta,c,\mathcal R}(x)=P(M=1\mid x,\theta,c,\mathcal R).
\]

这是下一组五条完整轨迹的mixed概率，既不是单轨迹答对率，也不是模型参数/预测头的认识不确定性。严格合法答案评分仍沿原契约；infra缺轨迹不补0，模型协议非法按原严格规则计0。

## Assumptions

1. 每个监督group恰好五条完整真实轨迹；保留五个EM、轨迹身份、count和协议版本。不同group不能共享轨迹、拼接响应或把同一次确定性执行冒充重复实验。
2. 所有关于未来随机标签的结论必须固定actor与协议，并明确随机量是什么。若固定完全相同种子、上下文和确定性执行，复算可能根本不是新独立group；温度0且工具确定时可全部确定性输出。
3. iid Bernoulli仅在明确标注的段落采用：给定 \(x,\theta,c\)，五次 \(C_j\) 条件独立且有相同成功概率 \(p\)。其他段落允许相关、slot差异和非Binomial count。
4. BCE/CE总体最优结论要求目标分布固定、容量能表示该条件函数、达到总体全局最优。它们不保证有限训练集、有限特征、小/大MLP和在线优化的实际结果。
5. question-only预测只能使用问题及已知历史策略context；不得用本轮gold、rollout或实际count作预测输入。评分后的标签可以用于下一次拟合。

## Notation

- \(p=P(C_j=1\mid x,\theta,c)\)：iid切片中的单轨迹严格正确率。
- \(q=P(M=1\mid x,\theta,c,\mathcal R)\)：真实group mixed概率；\(\widehat q\) 为探针输出。
- \(s(K)=\sqrt{K(5-K)/20}\)：G5的EM样本标准差，ddof=1。
- \(\bar s=E[s(K)\mid x,\theta,c,\mathcal R]\)：未来group的预期样本std，不是 \(\sqrt{\operatorname{Var}(C)}\)。
- \(a=\sqrt{0.2}=0.4472135955\)，\(b=\sqrt{0.3}=0.5477225575\)。
- \(R\)：同一题、同一actor/协议下相互独立的新G5 group数，不是每group响应数。
- \(\pi_k=P(K=k\mid x,\theta,c,\mathcal R)\)，\(k=0,\ldots,5\)：真实六类count分布。

## Derivation Strategy

先从G5的count定义得到不依赖iid的概率/std约束，再在iid切片求闭式概率与重测噪声。用proper scoring rule区分总体正确目标与有限样本记忆，最后比较direct BCE、binomial NLL和六类CE，并给出版本/探索反馈及小预算重复anchor设计。

## Derivation Map

1. \(K\to M,s(K)\)是恒等式；得到 \(a q\le\bar s\le bq\)。
2. 加iid Bernoulli假设后，\(K\sim\mathrm{Binomial}(5,p)\)，得到 \(q(p)\)。
3. 加跨group独立假设后，得到标签flip与重复软标签方差。
4. 对BCE/CE取条件期望并求导，区分proper总体解与经验过拟合。
5. 两输出重参数化给出合法 \((\widehat q,\widehat s)\)；IID模型、相关反例和Beta解释单列。
6. 数据版本与选择反馈决定实际训练分布，不能被一个概率公式忽略。

## Main Derivation

### 1. 无iid假设的精确约束与最小二输出修正

**恒等式：** \(s(K)\) 仅取 \(0,a,b\)：K为0/5时0，1/4时a，2/3时b。因此逐group成立

\[
aM\le s(K)\le bM.
\]

取任何真实联合生成分布的条件期望，得到

\[
\boxed{a q\le\bar s\le bq.}
\]

所以若两个输出分别声称预测 \(q\) 和 \(E[s]\)，\(q>0\)却std输出为0并不联合一致。该结论不依赖五轨迹独立或Binomial模型。

**最小建议参数化：** 保留两个logit而不强制增加六类头，令

\[
\widehat q=\sigma(z_1),\quad \widehat u=\sigma(z_2),\quad
\boxed{\widehat s=\widehat q[a+(b-a)\widehat u].}
\]

\(u\)可解释为 \(P(K\in\{2,3\}\mid M=1)\)；不是额外的iid假设。q=0时u不可识别，但std必须为0。闭概率域中的真值对总可表示；sigmoid有限logit只能逼近q/u等于0或1的边界，不能声称有限参数精确达到它们。

采用

\[
\ell=\operatorname{BCE}(M,\widehat q)
+0.3\left(\frac{\widehat s-s(K)}b\right)^2.
\]

**总体命题：** BCE的唯一最优边缘概率是q，MSE的最优std是 \(\bar s\)；该真值对处于上述合法域，两项可以同时最小化，所以总体全局最优仍保持q正确。在q>0时，\(u=(\bar s/q-a)/(b-a)\in[0,1]\)；q=0时u任意。有限共享模型、正则化和局部优化下，辅助项仍可能改变实际拟合结果。

**不必为该总体论证detach q。** 默认不detach更直接对应所写联合标量loss。实现需记录BCE与std分支对q-logit的梯度、各输出饱和和校准；若另做detach，只是明确的训练消融，不能继续原样引用“对联合标量loss求梯度”的论证。误预测q很小时，std支路对u的梯度也小；这种耦合及sigmoid饱和不会因合法域自动消失。

最小验收：每行finite且 \(0\le\widehat q\le1\)、\(a\widehat q\le\widehat s\le b\widehat q\)（数值容差明确）；新定义std为预期样本std，不得同名混用方差或单轨迹Bernoulli std。排序可直接用z1避免sigmoid数值精度导致大量精确1的tie；它是同一数学排序的数值修复，**不是校准或过拟合修复**。

### 2. iid Bernoulli切片：G5混合概率与极限

**受限命题：** 条件iid时 \(K\sim\mathrm{Binomial}(5,p)\)，所以

\[
q(p)=1-p^5-(1-p)^5.
\]

\[
q'(p)=5[(1-p)^4-p^4],\qquad
q''(p)=-20[(1-p)^3+p^3]<0.
\]

q关于0.5对称，端点0，唯一最大值

\[
\max_p q(p)=q(1/2)=15/16=0.9375.
\]

因此iid模型下最理想选题也只保证高期望mixed率，不能保证下一组mixed。512题各p=.5时预期mixed数480；这个结论不是一般相关生成模型的上界。

### 3. 标签重测、分类准确率与当前观察的可解释范围

若两组 \(M_1,M_2\) 给定同一题和actor/协议后独立，均为Bernoulli(q)，则

\[
P(M_1\ne M_2)=2q(1-q),\qquad
P(M_2=0\mid M_1=1,x)=1-q.
\]

后式不是前式；不能把仅原mixed题的反转比例当无条件flip率反解p。53个原mixed中16个变nonmixed，描述为fresh仍mixed37/53=69.8113%、条件反转30.1887%；它描述已筛选的53题，不代表随机题库q或每题q。actor/协议改变、非独立重测会进一步破坏上述同分布解释。

强制给未来标签作0/1分类时，已知真实q的最优准确率为 \(\max(q,1-q)\)。记住一次历史标签再复用其0/1预测，跨独立重测的平均准确率为 \(q^2+(1-q)^2\)。这并不等于学会q。

| iid单轨迹p | mixed概率q | 两组无条件flip | 每题Bayes标签准确率 |
| ---: | ---: | ---: | ---: |
| .01 | .049010 | .093216 | .950990 |
| .05 | .226219 | .350088 | .773781 |
| .10 | .409500 | .483620 | .590500 |
| .20 | .672000 | .440832 | .672000 |
| .30 | .829500 | .282860 | .829500 |
| .50 | .937500 | .117188 | .937500 |

q=.5时无条件flip最大.5，对应iid p约.129462或.870538。不能把“.9375上界”用作全题库分类准确率的上界：极端p的nonmixed标签几乎确定，全题库准确率可接近1。

fresh Top-512的261/512=50.9766%是已选择集合的mixed产出率/precision，不是总体分类accuracy。若head对这些题预测接近1，而当前actor、协议与评分已确实匹配，则支持该选择区间过度自信；不能仅凭训练拟合99.88%确定是何种特征、损失或实现问题。强随机标签能被高容量模型记忆，训练accuracy接近100%并不违反任何上述概率界。

### 4. 一次group/题的BCE为何仍成立，重复软标签为何更稳

固定题x，模型输出z∈(0,1)。条件总体BCE为

\[
E[\ell_{\rm BCE}(M,z)\mid x]
=-q\log z-(1-q)\log(1-z).
\]

求导 \(-q/z+(1-q)/(1-z)=0\) 得唯一最优z=q。**不要求每题都重复才能成为proper目标**；一次group/题仍可通过跨题共享结构学习条件概率。若固定表征h不能区分某些题，实际可识别对象为 \(E[q(x)\mid h(x)]\)，不是每个隐藏不同题的精确q。

一次标签的经验loss会被高容量模型逐题记忆；把随机标签0/1当确定目标无限优化，可能出现近乎0/1的过度自信。class-balanced BCE也会改变概率目标：正负样本权重分别w1/w0时，最优输出为 \(w_1q/[w_1q+w_0(1-q)]\)，不能未经校正称原始q。

同一题R个独立G5 group，定义 \(\bar M=R^{-1}\sum_rM_r\)、\(\bar s_R=R^{-1}\sum_rs(K_r)\)。恒有

\[
E[\bar M]=q,\qquad \operatorname{Var}(\bar M)=q(1-q)/R,
\qquad a\bar M\le\bar s_R\le b\bar M.
\]

平均group BCE精确等于以 \(\bar M\) 为soft target的BCE。平均std MSE等于对 \(\bar s_R\) 的MSE加一个与预测无关的样本常数。因此重复数据可以聚合成同一合法soft pair；R不等时，给每题等权还是按R加权须明确，二者不是同一经验目标。

小R只能诊断，不能精确知道每题q：q=.7时R=4/8/16的 \(\bar M\) 标准误约.2291/.1620/.1146；q=.9375时约.1210/.0856/.0605。重复不会自动修复特征不能泛化、当前actor漂移或过大模型容量。

### 5. 三种统计训练模型的比较

| 训练模型 | loss/输出 | 优势 | 必须承认的限制 |
| --- | --- | --- | --- |
| 直接mixed BCE | BCE(M,q)，可用上文一致std头 | 直接对应选择目标；不要求iid | 丢失K的方向/幅度信息；有限样本仍会记忆 |
| Binomial p头 | \(-K\log p-(5-K)\log(1-p)-\log\binom5K\)，q由闭式导出 | iid成立时使用完整count，参数紧凑 | q映射强制iid；相关生成时即使p估对，q也可错 |
| 六类count CE | \(-\log\widehat\pi_K\)，q=1−π0−π5，std=Σπk s(k) | 不要求iid；保留全部count分布/mean/std | 更多输出且罕见类别样本有限，不能自动阻止训练集记忆 |

Binomial NLL即使在非iid情况下，其期望最优p仍是E[K]/5（相当于边缘Bernoulli准似然）；但这**不能**证明analytic q正确。六类头也可增加非负权重的mixed BCE或std MSE：真count分布可同时最小化各项总体proper目标，实际有限容量权衡另验。

**主建议：** 当前只需要binary mixed和std时，先实施上文二输出一致参数化＋稳定BCEWithLogits＋明确辅助std项；这是最小改法，不引入iid假设。保留六类count CE作为信息更完整的对照，Binomial p作为需检验的受限基线。是否采用线性头、小MLP、正则和early stopping，应按同一actor的freshgroup/新qid误差选择，不能以warmup fit最高决定。

### 6. 单组p plug-in偏差与Beta posterior predictive

若直接用一次G5的 \(\widehat p=K/5\) 再计算q，得到K=0/5时0，K=1/4时.672，K=2/3时.912。不能将其当无噪声的每题mixed概率标签。

由于q(p)严格凹，iid下 \(E[q(\widehat p)]\le q(p)\)，有向下偏差。p=.5时真实q=.9375，但一次G5 plug-in的期望只有.78。重复R组后合并5R条估p，R=2/4/8/16时其期望为.86625/.903984375/.9212988281/.9295422363，仍不等于真实q。观察M本身始终是q的无偏一次样本。

若明确采用p未知的Bayesian iid模型，先验Beta(α,β)，n=5R条共S正确，则对未来G5的posterior predictive为

\[
q_{\rm pred}=1-
\frac{(\alpha+S)_5}{(\alpha+\beta+n)_5}
-\frac{(\beta+n-S)_5}{(\alpha+\beta+n)_5},
\]

其中 \((u)_5=u(u+1)\cdots(u+4)\)。它积分了对固定p的认识不确定性，不等于把posterior均值代入q。

例如Beta(1,1)先验、一次G5：K=0/5时posterior predictive约.452381，K=1/4时.714286，K=2/3时.833333。这不是数据证实“全错题实际有45% mixed概率”；它是强烈受小样本先验影响的预测。不同先验、真实p=0或1、非iid都会改变解释。

### 7. iid失败的反例、过度离散与认识不确定性

**反例一：** 每个group先抛公平硬币，再令五条全部同对/同错。每条边缘p=.5，但q=0，不是.9375。

**反例二：** 每组随机产生恰好2或3正确（各半，成功slot均匀），每条边缘p=.5，但q=1。说明无iid时q上界可以为1。

若slot条件独立但成功率不等，q=1−∏j pj−∏j(1−pj)，不能用统一平均p闭式代替。固定seed设计、共享工具状态、批次布局或不同概率后处理可能改变条件分布；它们不自动证明存在正相关，应实测/记录协议并避免将不同版本混合。

一个正相关的解释模型是：每group latent P_r~Beta(α,β)，给定P_r后五条独立。则K服从Beta-binomial，边缘p=α/(α+β)、题内相关 \(\rho=1/(α+β+1)\)，

\[
\operatorname{Var}(K)=5p(1-p)[1+4\rho],\quad
q=1-\frac{(α)_5+(β)_5}{(α+β)_5}.
\]

p=.5时ρ=.1/.25/.5，q分别.8671875/.7421875/.5078125；同样边缘正确率，group更容易全对/全错。它只覆盖一类正交换相关，不覆盖所有依赖；单group/题无法独立识别每题p与ρ两个自由参数。

Beta posterior对固定p的不确定性属于**epistemic**；真实每group latent P波动的Beta-binomial是生成过程的过度离散。虽然公式形式相似，物理含义不同。q(1−q)则是mixed标签自身的未来随机性，不是模型误差条；ensemble disagreement和训练集不足应另测。

### 8. 随机anchor、选择反馈及actor版本的最小实验设计

**第一阶段固定actor/协议，不先改actor训练。**

1. 在训练题库中预先hash抽32或64道随机anchor，冻结当前head的预测；每题R=4个独立fresh G5，记录所有K/五EM。32×4×5=640条、64×4×5=1280条；若已有同actor/协议的第一group，则增量分别480/960条。额外生成成本必须单列，不冒充原512×5预算内免费完成。
2. 分开两种泛化：历史见过qid的新seedgroup测试，检验是否记住一次标签；未进入head拟合的qid测试，检验表征/函数跨题泛化。anchor先用于predict-before-fit评估，不能拟合后又当独立holdout报告。
3. 对随机anchor与focus Top-512分别报告mixed logloss/Brier、分箱预测概率对真实频率、K分布、std MAE与输出合法域。少量每题soft target方差大，主要用于聚合校准和noise/依赖诊断，不能宣称精确每题真值。
4. 比较最小一致二输出、Binomial p、六类CE时，共同qid/重复group/actor/协议、特征和训练/评估切分保持一致；选择模型看fresh proper score与mixed yield，不看训练fit最高。

**在线阶段条件对象必须绑定版本。** 每条label至少绑定actor-before SHA、生成协议/检索器/scorer SHA、slot seed/stream、group及trajectory身份、candidate/selection账本、采样时head SHA。replay中旧标签仍是qθu的观测，不是当前qθt真值。

若只以question表征h(x)并对近期标签年龄加权，BCE总体解实际是各历史策略q的加权混合，除非假定近期漂移足够小。每轮更新head本身不证明已跟上actor；新head用θt采样标签却用于θt+1，仍有一轮测量滞后。策略hash用于防混淆，单独把从未见过的hash当one-hot输入并不能学会新策略表现。

最小可行路线是先明确近期窗口/年龄权重并报告其近似，保持现有随机explore cohort为持续校准/漂移审计；必要时另对照历史随机cohort统计组成的policy context或相邻快照重复anchor，不能声称固定h(x)已包含当前actor的全部能力变化。

**selection/bandit反馈边界：** 只对预先选中的题观察K。按预测Top选择本身不看未来标签；给定完整x/actor/协议，选择不会改变该题的真实q，但会改变观测问题分布和有限模型拟合权重，并可能持续利用模型的过度乐观误差。无随机覆盖时，未选区域无法从日志识别真值。

先从N=5120候选均匀抽E题、再其余Top的设计中，每题至少有E/N进入随机cohort的概率。随机cohort可独立估候选分布上的prequential误差；focus不能冒充随机测试。若拟合总体候选风险确需倾向校正，应记录实际入选propensity，避免把简单E/N误当最终总入选概率，并单列方差/截断权衡；**不自动将该权重乘到actor GRPO优势**。

在可信校准前，Top-512分数不是可验证oracle。最多可说给定预测，Top最大化预测mixed数量；未选题没有rollout真值，不能报告全5120的真实recall/regret。探索率、policy漂移跟踪、正则/early stopping都是独立设计选择，不能用“标签随机”解释完所有失败。

## Remarks and Interpretation

- 上述joint-coherence修正解决“q高而预期std为0”的代数矛盾；不解决高容量训练记忆或fresh过度自信。
- 一次group BCE数学正确；少量重复提供方差诊断和更稳定soft监督，但不构成必须全题库重采的理由。
- iid解析q、六类count分布、二输出一致约束各有范围：首个强假设，后两者不要求iid。不要把Binomial模型的.9375上界套在所有真实生成系统上。
- 当前261/512是新采样已选集合的一次结果；53题重复只来自原mixed子集。不同cohort和指标需分列，不假设它们足以定位唯一根因。

## Boundaries and Non-Claims

- 无新GPU/模型/远端执行，没有性能收益或修复验收声明。
- 本地合成计算只有标量组合数/概率，不含任何真实问题、rollout、tensor或权重。
- 不将训练fit高称为校准成功，不将随机标签称为“不可学习”，不保证512题全mixed。
- 所有训练/部署需另经完整源码、标签/版本、资源与端到端进展验收；本文仅供设计和审查。

## Open Risks

1. 需要实际区分label/parser错位、独立输出不一致、有限样本过拟合、actor漂移和selection反馈；目前摘要不能唯一归因。
2. repeated groups必须新随机流且相同actor/协议；否则噪声/相关估计不对应本文对象。
3. 32/64×R4 anchor是小预算诊断建议，不是已执行参数或足够精确的每题真值。
4. 若最终目标仍为预期std而非mixed概率，二者排名可能不同；主ranking应与人工目标明确对齐，不能因为修复q头静默换目标。

## Local Synthetic Verification

2026-10-16以Python标准库对G5闭式、Beta rising-factorial、Binomial精确求和核对表内数字。关键精确值：q(.5)=.9375；flip=.1171875；p=.5单G5 plug-in期望=.78；R4 plug-in期望=.903984375；Beta(1,1)单组K0 posterior-predictive=19/42≈.4523809524。仅本地标量计算，未产生模型推理。

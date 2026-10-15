# Question diversity probe 训练诊断与修复方案

日期：2026-10-16。本文以用户最新质疑为优先：先解释原训练集标准差全0和正式replay组成，再讨论随机G5下的概率预测。只做本地静态核对、现有小型审计读回和标量推导；未操作另一线程的训练，也未实施任何改动。

## Target

判断当前question diversity probe是否实际拟合既定训练目标，给出能够分别验收分类、标准差、在线replay和新采样泛化的修复路线。不能用随机标签解释已见训练集的标准差支路全零。

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION。**

已证标准差训练拟合失败，已证正式首轮并非5120+512；具体造成回归失败的raw logit、梯度、参数更新原因尚未定位。概率建模仍是之后的目标设计，不替代当前训练健康检查。

## Invariant Object

既定联合训练目标及其真实数据分布：

\[
L=\operatorname{BCEWithLogits}(z_m,M)
+0.3\left(\frac{\hat s-s(K)}{\sqrt{0.3}}\right)^2,
\quad M=\mathbf1\{0<K<5\},\quad s(K)=\sqrt{K(5-K)/20}.
\]

其中K是该题五条完整轨迹的严格EM正确条数；s为样本标准差，不是二分类概率。评价对象包括原训练标签的拟合和独立fresh-G5的概率质量，二者分别验收。

## Assumptions

- 所引用现有审计绑定原warmup Head200及正式初始状态；本次仅读取其小型metadata，没有重新读取模型或原rollout。
- 原5120题与其五条标签固定，旧权重控制实测复现5114/5120分类准确率。
- 将未来概率定义为题目、actor快照和完整采样/检索/评分协议的条件量；只有讨论Binomial对照时才额外假设同组五条条件IID。
- 本文不改已有网络、损失、训练或停止状态。方案需在独立版本实施，不热改现作业，不自动重训actor。

## Notation

- K∈{0,…,5}：一次G5中正确轨迹数；M：该组是否混合。
- q：未来一组G5混合的条件概率；\(\hat q=\sigma(z_m)\)：分类输出。
- s(K)：观测组标准差；\(\bar s=E[s(K)]\)：未来组标准差期望；当前\(\hat s=\sqrt{.3}\sigma(z_s)\)。
- \(a=\sqrt{.2},b=\sqrt{.3}\)：非零sample std的最小值、最大值。
- actor version与协议绑定每条标签，不将旧标签改名为当前策略真值。

## Derivation Strategy

实际训练记录 → 分清两个输出 → 核正式数据组成 → 推导饱和风险与联合输出约束 → 先验收训练健康 → 再检验fresh-G5概率及选题收益。

## Derivation Map

1. 用原训练集控制记录证明分类与回归表现不同。
2. 用adoption和首轮账本证明正式数据组成，不从原初始化权重推断仍保存warmup replay。
3. 对既定sigmoid-MSE求导，定位需要验证的梯度问题；不把机制推论冒充实际根因。
4. 从G5标签的有限取值推导无IID假设的联合输出边界。
5. 在训练健康通过后，学习随机事件的条件概率，以独立fresh标签验收。

## Main Derivation

### 1 已证事实及时间线

原5120中真mixed题数为1845（TP1843+FN2）。原Head200分类预测复现5114/5120=99.8828125%，但是标准差预测5120个全部为0，回归MSE=.08843749978，MAE=.17760460079。**这是分类分支拟合成功、标准差分支拟合失败，不是所有题都不混合，也不是整个fit没有执行。**

[原题控制记录](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-head200-control-read-1792083940623678000.json:31)，[标准差全零审计](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-same-question-G5-label-and-std-audit-20261016.json:25)。

即使不看题目，常数预测训练集平均std约.1776046，也能将MSE从约.0884375降到 \(.0884375-.1776046^2\approx.0568941\)。因此全零甚至没有达到这个无特征常数回归基线；这项比较固定同一训练标签，不依赖新G5是否随机。

现有warmup epoch小记录进一步表明：

| epoch | Head Adam | 分类accuracy | std MSE | std MAE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | .633984 | .158257106 | .336120605 |
| 1 | 20 | .375195 | .088437500 | .177604601 |
| 5 | 100 | .819336 | .088437500 | .177604601 |
| 10 | 200 | .998828 | .088437500 | .177604601 |

epoch1–10的std MSE、MAE和辅助loss贡献均完全不变，停在与最终零预测一致的误差平台。没有逐epoch预测数组，故不声称已逐题证明epoch1全零；最终Head200全零有独立审计。

来源JSON路径为 `observation.files['warmup/warmup-metrics.json'].metadata.epoch_history`：[小型记录](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-json-binding-repair-completion-observation.json:1)，本地SHA `8b2acf4c75e6fa14cb48b949f4d6e9db0a58c34e823e2b534ca13e0b8b5832cf`。

### 2 正式首轮实际没有保留5120题

`adopt_warmup`加载model、Adam和私有RNG，随后明确执行：

```python
instance.replay_labels, instance.replay_features, instance.protocol = [], [], None
```

注释包括“outer1's first fit”；正式首轮replay为512题，两epoch、每256题一次Adam，共4次Head Adam（200→204）。因此用户现在期待的“5120+512=5632题”与当前实际实现不同。保留预热权重不等于保留预热监督数据。

[清空位置](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_qwen_block_online.py:229)，[fit读取replay](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_qwen_block_online.py:139)。

如果第一轮真用5632条、batch256、两epoch，应有44次Head Adam机会，而不是4次。重复qid怎样处理、旧新G5怎样赋权必须另行明确，不能只改一个计数让它看上去等于5632。

**时间关系：标准差全零在原warmup200权重上已经存在。正式清空旧replay可能影响后续保持，但不能解释更早的回归失败。**

### 3 sigmoid-MSE为何可能有正loss却学不动

现输出为 \(\hat s=b\sigma(z_s)\)，辅助项为 \(L_s=0.3(\sigma(z_s)-s/b)^2\)。精确求导：

\[
\frac{\partial L_s}{\partial z_s}
=0.6\left(\sigma(z_s)-s/b\right)\sigma(z_s)(1-\sigma(z_s)).
\]

即使s>0，只要sigmoid已极接近0，梯度也可极小；若实际sigmoid激活精确0，标准sigmoid反向中该因子为0。但保存输出为0本身不足以证明实际激活或梯度精确0，还须排除dtype转换和记录精度。对照分类的稳定BCEWithLogits梯度为 \(\hat q-M\)，错误的极端预测仍可获得明显纠正信号。两支路因此可能出现“分类继续改善、回归长期不动”。

这是**可检验机制，不是已定位根因**。当前记录缺std raw logit及逐支路梯度，尚不能区分早期饱和、无梯度连接、优化器参数遗漏、更新后覆盖或其他实现问题。代码把回归项加入loss，有限性检查不能证明梯度非零或训练目标有效改善。

[输出参数化](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_qwen_block_model.py:178)，[loss构造](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_token_model.py:106)。

### 4 两输出应该满足的精确一致性

G5的s只有三个值：不混合时0，K=1/4时a，K=2/3时b。无需IID假设：

\[
\boxed{a q\le E[s]\le b q.}
\]

证明：逐样本恒有aM≤s≤bM，取同条件期望即可。当前独立双头可输出较大的混合概率与零标准差，违反其所声称的联合统计意义。

保持二分类主输出和既定BCE+.3std-MSE时，一个待验的最小一致参数化为：

\[
\hat q=\sigma(z_m),\quad u=\sigma(z_s),\quad
\boxed{\hat s=\hat q[a+(b-a)u].}
\]

u可解释为混合条件下K=2/3的概率，q=0时该条件量不可识别。此结构不增加输出数量、候选生成或encoder前向；当\(\hat q\)较高时\(\hat s\)不能再单独坍缩为0。在无限容量、固定目标分布和全局最优条件下，真实q与E[s]可同时表示（sigmoid的0/1端点在闭包极限中理解），从而同时最小化BCE和MSE；有限模型的共享梯度仍会耦合，u仍可能饱和，不能承诺自动治好优化。

这属于明确的输出语义/参数化变更，必须独立对照，不热改当前模型。若后续改为两级BCE监督M及混合条件下K=2/3，会改变既定loss，数学上等价三种结果的因子化likelihood；不能只称“还是两个输出”就规避先前不加CE3/6的约束。

### 5 训练健康通过之后才处理随机标签

定义目标 \(q_{\theta,c}(x)=P(0<K<5\mid x,\theta,c)\)。固定一次G5的M是该事件的一次观测；BCE本来就是合适的概率loss，单次标签并不令BCE在总体上失效。高容量模型记住固定M与对fresh-G5过度自信可以同时发生。

同题、同actor、同协议重复R组的soft target：

\[
\bar M=R^{-1}\sum_rM_r,\quad \bar s=R^{-1}\sum_rs(K_r).
\]

题内平均BCE精确等于对\(\bar M\)的BCE；题内平均MSE等于对\(\bar s\)的MSE再加与参数无关的常数，这些代数恒等式不要求观测独立。减少测量误差的重复采样方案则需fresh独立随机流；相关观测不能保证方差按1/R下降。反复epoch读取同一固定组不能替代新观测，也不把下一组结果变确定。

只有五条条件IID、同成功率p时，\(q=1-p^5-(1-p)^5\le.9375\)。这是可作对照的Binomial模型，不能当任意G5的普适上限；也不能将原53个已选mixed题的16次翻转当总体噪声率。BCE/概率验证的依据见[proper scoring原始研究](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)。

## 修复与验收顺序

### A 先定位已见训练集回归失败

只用已有原训练缓存、标签和原Head200，在独立诊断副本取包含K=0、1、2、3、4、5的固定小批，记录：std raw logits、sigmoid输出；分类/回归各自loss；两项分别反传时readout两行和block的梯度范数；一小步前后参数及预测变化。检查requires_grad、optimizer参数覆盖、detach/更新后覆盖、实际输出dtype。不得把finite且Adam计数增长作为学习有效验收。

针对首次坍缩的位置恢复最早可用状态作诊断；没有保存该状态就明确不可精确重放，不能把新训练伪称历史复现。先验收该固定小批的回归loss实际下降、nonzero目标得到nonzero可区分预测，再扩大到原5120。

### B 按最新数据要求制定显式replay合同

若采用用户最新期望，首轮监督保留5120预热观测并加入512条fresh观测，manifest逐项可核，总计5632条group记录；同qid的不同G5不可覆盖成一条伪确定标签。模型、Adam、replay、RNG分别绑定。

旧标签来自旧actor，必须保留actor/protocol身份；随后actor漂移时是否降权/退出warmup replay由明确配置决定。不能无期限让5120旧标签以十倍条数淹没当前512，也不能无说明清空。原label随机性影响长期概率学习，不是当前replay实现差异的理由。

### C 恢复两个目标的训练健康门禁

同时报告原训练分类BCE/accuracy、std MSE/MAE、两支路raw logit/输出范围、饱和比例、梯度和参数增量。warmup checkpoint不得仅凭99.88%分类准确率验收联合目标。epoch1–10回归停在零预测平台应判该项不通过。

若改参数化、优化器LR/weight decay/grad clip或冻结block，逐项明确、独立对照；现有LR3e−4、约7808万参数是优化风险，不等于已经证实学习率或容量是根因。现缓存layer27输出接copied layer35也是需单独验证的初始化分布风险。

### D 最后再做概率泛化和选题收益

固定actor下区分同题fresh-G5、新qid、较晚actor三个轴。以fresh Brier/logloss及概率分箱为概率门禁，以同预算Top512相对随机512的mixed产出提升为排序门禁；原题99.88%与新Top512的50.98%不是同一个估计量，不直接相减为泛化gap。

沿用已有随机新512诊断的独立性，不因本分析自动重跑或将其用于训练。若需额外重复采样，可先只在小规模训练/校准anchor补新G5，不在全部5120上重复生成；预算和actor冻结先明确。训练健康未通过前，不增加rollout来掩盖优化问题。

在线选题需保留独立随机审计/探索slice，否则纯Top512没有未选题反馈。raw logit排序可避免sigmoid精确1造成的数值平局，但不修复错误排序或概率失准。[校准原始研究](https://proceedings.mlr.press/v70/guo17a.html)、[随机日志评价原始研究](https://arxiv.org/abs/1003.5956)。

## Remarks and Interpretation

- 本次优先结论是训练健康和数据组成问题；随机标签只用于之后解释预测目标，不替训练失败免责。
- 预测标准差0，在真实标签层面确实表示K=0或5；模型输出0则可能是预测错误。当前分类概率与标准差独立，不能把二者混成一个字段。
- 正式Top512按分类概率排序，std不参与。因此std失败不能单独被认定为首轮50.98%的直接原因；它证明联合目标未通过验收。
- “99.88%拟合成功”只能用于分类分支和固定标签；全任务成功还要回归、fresh概率及选题收益分别通过。

## Boundaries and Non Claims

未执行本方案、未训练或恢复任何actor/head、未更改另一线程作业。没有新512结果，不报其准确率；没有std raw logits/grad，不确认下溢根因。不以数学约束证明某个参数化必然提升最终EM，不强行认定过拟合或噪声为唯一根因。

## Open Risks

sigmoid-MSE饱和的实际起因；分类/回归共享梯度是否失衡；replay规模与年龄权是否合约一致；固定问题特征对当前actor变化的可预测性；纯TopK的反馈覆盖；独立概率校准与最终actor收益的差别。

分项证据与数学核对见同目录 `code-and-evidence-notes.md`、`math-notes.md`、`research-and-review-notes.md`、`math-and-source-checks.json`。本文由AI辅助静态核验与推导，实验结论只引用已保存实测回执。

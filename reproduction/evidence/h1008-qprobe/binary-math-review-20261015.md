# Binary mixed probe：独立数学与算法审核

审核时间：2026-10-15。范围：仅本地只读数学/算法设计审核；未修改实现、训练状态，未远端操作或启动任务。

## 结论

**所提目标与损失在数学上成立，没有发现方案本身必须阻止实现的矛盾。建议采用该方案；但尚未通过新实现代码审核，不能用此文档授权启动。** 旧实现目前仍是六分类与 sigma 排序，必须完成下列全链路变更并经独立代码核对后再启动。不能承诺 10 epoch 训练就会分类完美、首轮 Top448 混合率必然提高或 actor 最终质量提高。

审核的明确方案：对五条完整轨迹的严格终答案 EM，K=sum(EM)，y_mix=1[1<=K<=4]；双标量输出 z_mix、z_v；p_mix=sigmoid(z_mix)，v_hat=.3*sigmoid(z_v)；逐组损失

`BCEWithLogits(z_mix,y_mix) + 0.1*((v_hat - K*(5-K)/20)/.3)^2`。

同一 5120 训练题 warmup 10 epoch，不另切 heldout；之后按 p_mix 选题、64 个均匀随机探索、官方 actor 从零 fresh200。

## 数学判断

1. **分类目标对应实际诉求。** y_mix 恰好表示同一 G5 中既有正确也有错误；K=0、5 均为负类。分类 BCE 的条件风险最小解是 `P(1<=K<=4 | question features, actor policy, sampling/retrieval protocol)`，不是单条答案正确率，也不是模型的永久题目属性。该结论不要求五条生成独立。
2. **方差公式与尺度正确。** 对五个 0/1，样本方差 `sum((EM-K/5)^2)/(5-1)=K*(5-K)/20`；K=0..5 的值为 `[0,.2,.3,.3,.2,0]`。这是 G5 内严格 EM 的样本方差，不是 K 的条件方差或二分类概率。除以 .3 后平方把辅助误差归一化到 [0,1]；其加权贡献至多 .1，主 BCE 无此上界。
3. **两个头必须分开。** BCE 优化 p_mix；辅助 MSE 优化条件平均样本方差。两者相关但不相同：K=1/4 与 K=2/3 均是 mixed，而方差分别 .2 和 .3。若只用方差排序，会重新引入用户正在纠正的目标错配；排名必须实际用 p_mix。
4. **有界回归合理。** `.3*sigmoid(z_v)` 严格处于 (0,.3)，有限 logit 无法精确取端点，但可以任意逼近；这与 sigmoid 分类的端点情形相同，不构成数学错误。无需为达到精确端点另加 clamp 或非标准梯度。
5. **辅助标签名称不能混淆。** 该头预测的是 E[v|x]；`sqrt(v_hat)` 通常不等于 E[sqrt(v)|x]。若保留诊断 sigma 字段，不能把 sqrt(v_hat) 命名为旧 `expected_std` 并宣称概率矩一致。辅助损失和曲线应直接报告 variance MSE/normalized variance MSE。
6. **样本权重应明示。** warmup 的 5120 组均来自 actor outer0，age weights 归一化后应全为1；必须实际 fit 全5120，不能继承普通 bootstrap 每 outer 上限512而只训练十分之一。online 若沿用 age weights，每组先计算上述原式，再按已绑定权重聚合，不改变分类标签或用 class weight 悄然替代现目标。

## 选题算法必须写清楚

建议保留现有 `uniform_first`：私有 RNG 从完整5120候选均匀无放回抽64，再在余下5056按 p_mix 降序取448，使用与标签无关的固定身份 hash 打破并列；最后恢复原 ordinal 顺序以保持 G5 绑定。它满足512个不同题组、Top448+64random预算，并保留来自全候选的随机监测样本。

这里 Top448 指**去掉探索64之后的余池 Top448**，不是完整5120全池 Top448。若改为先全池 Top448 再余池随机64，也是一种合法但不同的算法；其 random64 仅代表余池，不能再称全候选的无偏随机基线。不得在实现或报告中混用二者。

选题发生在当前轮 rollout/strict EM 之前；不能使用本轮 gold K、reward、轨迹或终答案特征参与排序，也不能 rollout 后根据 EM 丢弃/回填全错或全对组。真实 Top448 可能仍有 K=0/5，它们是预测错误或生成随机性，应保留并如实报告。

## 可学习性、噪声与评价边界

- 每个 q 的一个 G5 标签是随机 realization。teacher warmup 标签只对应冻结官方 actor 与其实际采样/检索协议；学习同一5120标签可能产生很高的训练准确率，但不能据此声称对未知题或更新后的 actor 校准成功。10epoch 没有数学上的完美分类保证。
- 用户明确不用 heldout，接受此约束；每epoch准确率、precision/recall/F1/confusion、Brier/BCE 和排序命中均标记 **同训练集 in-sample**。0.5 是报告分类阈值，不是 Top448 的排序截断条件。
- 类别不均衡时必须同时报告 majority accuracy baseline；全部预测 mixed 可有高 accuracy。报告 p_mix 的 min/max/mean/std 与 Brier/BCE，避免仅靠训练 accuracy 断言“学会”。不得为提升表面 accuracy 默改阈值、采样或加入 class weight。
- 真实首个 actor outer 的预测要在生成前保存版本；生成结束、head fit 之前按 Top448/random64 分组汇总实际 `1<=K<=4` 比例、mixed题数、分母、allwrong/allcorrect。随机64的估计有明显抽样方差；一轮差值不是严格随机对照因果收益。
- 首轮新生成 mixed 比例是该轮真实效果；warmup 标签上的回顾 Top448 mixed率是拟合诊断，二者不能互相代替。预测分数、标签与 teacher/actor snapshot 必须精确匹配。
- teacher5120允许属于 train，但必须与固定 validation512/final51713 的排除与身份约束一致；特征只能来自题目，不可带 gold答案、teacher轨迹、EM或K。不能把同训练标签拟合称 heldout 验证。
- actor 迭代后 policy 会漂移；online 每完整 outer 用真实采样结果更新 probe 是合理的。高 p_mix 不保证 PPO/GiGPO 梯度价值或最终问答分数提升，评测质量仍须实际测量。

## 实现代码审核的启动前阻断项

以下是待实现核对要求，不是要求新增 GPU 校准或基线：

- 删除新架构的6类 logits/softmax/countCE/std3CE，输出与检查确实为 binary logit 和独立 variance logit；不伪造六类分布以满足旧接口。
- model、sequence fit、rank7 RPC、CPU prediction DTO、selector、runtime admission/config、审计与 prequential metrics、warmup/checkpoint load/save 全链路接受新 schema，实际 ranking=p_mix。
- 核对完整 G5 strict EM，排除 incomplete/missing labels；对 K=0..5 的目标、损失及梯度做有意义 CPU 夹具；检查 normalized auxiliary coefficient=.1。
- 固定全5120、10epoch与 online epoch预算独立，epoch0..10及实际Adam计数记录真实；只保存标量，不新增全量statehash/模型数值诊断阻塞训练。
- 新 binary probe checkpoint schema/architecture 与旧 count6 probe 明确拒绝混读，不能继承旧训练 probe optimizer；官方 actor fresh0，旧训练 actor 的参数/Adam/scheduler/轨迹不得拼接。
- private selection/tie/shuffle RNG 不污染 actor RNG；64random+448focus无重复且是预测先于标签。首轮实际效果必须从完整真实标签提取，并在当前轮 probe fit 之前记录。
- warmup缓存如复用，只能复用采样/数据/检索/官方权重/strict scorer 完全绑定的完整 teacher原始标签；新 loss 改动本身不需要重采 teacher，但不能通过宽松source覆盖接纳不同标签协议。

**审核状态：数学方案 APPROVED_WITH_EXPLICIT_SCOPE；新实现 CODE_REVIEW_PENDING。当前没有新方案的实际训练/评测结果；本审核不声称训练或质量成功。**

# Prepared only：训练先验logit调整的粗事件目标

这是一组新的训练目标准备，不是给旧预测换阈值，也没有性能改善证据。当前Count head训练、正式旧结果与12节点screen均不变；没有runner接线、GPU提交或新预测。

六个count logits记为z0..z5。真实G5 own strictEM给出K=sum EM，Mixed为0<K<5。保留原始六个logits及unadjusted softmax6，保留原p_j用于trajectory监督与追踪。

- binary：e=[logsumexp(z0,z5), logsumexp(z1..z4)]，标签nonMixed/Mixed，τ=[0.5,0.5]。
- three-event：e=[z0, logsumexp(z1..z4), z5]，标签AllWrong/Mixed/AllCorrect，τ=[0.25,0.5,0.25]。

训练loss为 `CE(e + log(π_train / τ), target)`。π_train只从原30016题、完整938步、每题四个真实G5视图的120064个组的K直方图求得；每个view权重同为0.25，因此与加权计数相同。binary把K0/K5相加为nonMixed；three-event保留K0、K1..4和K5。固定private receipt仅在远端读取，绑定2155B、SHA `2f76659c49fa6e280b6ef3a3ef528128d9ab29dd41c4e671951e8df5563eae5e`。不使用validation先验，不重新估计π，不静默平滑零类，不导出私有数值。

原始问题归约保持 `mean_q(sum_v 0.25 * loss[q,v])`。API强制[Q,4,6]与[Q,4,5]，四个权重全部0.25；不class resample、不另乘inverse-class weights、不改变每题总权重1。未来若做对照，仍需明确保留trajectory BCE及旧block/Adam；本模块本身没有改动这些组件。

调整方向的检查：如果无信息的常数event logits为e=logτ，调整后e+log(π/τ)=logπ。真实训练标签按π分布，则期望梯度 `E[softmax(logπ) - onehot(target)] = π - π = 0`。反号会失去这一性质。这里用π对标签求期望只是数学验证，生产loss不额外按class加权。

推理使用raw event-softmax，不加训练offset；Mixed分数仍是raw softmax6的K1..4之和。它是**τ参考分布下的balanced decision score**，不能称原始题目分布上已校准的真实Mixed概率。固定z时raw输出没有被事后改写；未来训练会因为新目标而改变z，这才是该实验的区别。

粗目标只识别事件概率：两个模式都不能识别K1..4内部的拆分，binary也不能识别K0与K5内部拆分。保留softmax6用于追踪不代表它学会了真实六类count概率；6-way argmax准确率或countNLL不能用来证明这个新粗目标训练正确。主要应报告对应粗event loss、Mixed BA/TPR/TNR、全错/全对FPR、AUC与Top512，原始p/K输出另存且带清楚语义。

必要CPU资格限定为：K0..5映射、raw概率聚合一致、极端logits有限梯度、e=logτ下期望梯度为零与调整符号、四view/原问题归约的值和梯度一致、拒绝破坏每题预算。真实远端prior也只验证恒等式，不训练或评价模型。等待现有20epoch结果后才决定是否运行后续对照。

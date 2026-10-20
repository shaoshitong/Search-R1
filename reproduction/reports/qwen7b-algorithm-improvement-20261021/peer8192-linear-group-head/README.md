# Prefix20 / L26 直接 Mixed 线性头诊断：未改善

本次预先冻结的有限诊断未显示可替代现有探针的提升。新固定2048题上，内部选定的直接线性头 AUC52.63%、Top512真实Mixed158/512（30.86%），弱于同HF E4的57.65%、196/512（38.28%）。**不批准生产替换，不改原block/Adam，不继续用此验证集选择其它模型。**

## 实际实现与隔离

只读取已验收的 peer8192 NQ authentic old5＋new5 G10 的 L26 Prefix20 表征。先完整核对320个L26 manifest及payload SHA，再按索引读取 `policy.bin` 的实际span；不重构Q273、不读取L20/L23、不启动actor/原探针更新。

两种预注册轨迹池化：20个实际有效prefix token的masked mean，或最后一个真实有效token。零长度明确用零向量和长度标记，不拿padding当真实token。每个G5组的输入是五轨迹向量逐维均值＋逐维方差，再加3个真实长度统计，共7171维；直接 sigmoid(w·x+b) 判断 `0<sum(ownEM)<5`，不经过逐轨迹p的乘积公式。

采用真实B32×256、seed1729、epoch0的四个唯一随机G5视图，每题四×1/4总权重1。按qid hash整题分割：6603题（26412 views）内训、1589题（6356 views）内部验证，同题视图绝不跨界。标准化只拟合内训数据；普通BCE＋λ||w||²/2，未加类别权重。两池化×λ={.001,.01,.1}，最多80次LBFGS迭代，按内部BA、AUC及固定tie规则选择唯一winner和阈值。**保留内训80%拟合得到的winner，不用全8192重新拟合。**

winner为prefix_mean、λ=.1；权重、标准化和内部阈值于Unix1792529749.596冻结，之后才打开新V固定G5的表征、ownEM和E4已有预测。外部只对这个winner评分一次，未读取原diag512或final4096。该2048是已查看的recipe-selection validation，不能叫untouched最终测试。

## 同一份外部logits的两套预先指定阈值

| 模型／阈值 | Mixed均衡准确率BA | Mixed recall | nonMixed recall | 全错→Mixed FPR | 全对→Mixed FPR | AUC | Top512真实Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| **线性头，固定概率0.5** |**49.95%**|17.80%|82.09%|17.35%|19.70%|52.63%|158/512=30.86%|
| 线性头，内部冻结阈值（概率约0.321） |51.82%|46.46%|57.18%|40.28%|50.91%|52.63%|158/512=30.86%|
| 同HF E4，固定概率0.5 |55.77%|67.57%|43.97%|51.18%|71.52%|57.65%|196/512=38.28%|

公平jointgate使用**固定0.5 BA**＋阈值无关的AUC/Top512。内部阈值学习是另一条明确标注的校准协议，不用它掩盖固定0.5的结果；没有在外部重选阈值。两行线性头来自同一份既有logits，不重读表征或运行另一模型。总nonMixed→Mixed FPR依次17.91%、42.82%、56.03%。

固定0.5的线性头主要偏向nonMixed；移动到内部阈值虽提高Mixed recall，AUC和Top512不变，整体BA仍弱。不能把这一校准移动解释成排序能力提高。外部原始Mixed基率663/2048=32.37%，此线性头Top512甚至略低于基率。

## 已有的六个内部候选（未额外评外部）

| 池化 | λ | 内训BCE | 内部BA（内部选阈值） | 内部AUC | 最终梯度max | LBFGS迭代/上限 |
|---|---:|---:|---:|---:|---:|---:|
|prefix_mean|0.001|0.23063|53.61%|53.99%|0.00369421|80/80|
|prefix_mean|0.01|0.28587|54.65%|54.96%|0.00181415|80/80|
|prefix_mean|0.1|0.42106|56.12%|57.24%|0.000167794|80/80|
|prefix_last|0.001|0.27527|52.32%|52.81%|0.00246537|80/80|
|prefix_last|0.01|0.31948|52.91%|53.39%|0.0018106|80/80|
|prefix_last|0.1|0.43203|54.17%|55.05%|2.96196e-05|76/80|

训练初始BCE约0.62992。弱正则mean λ=.001把内训BCE降至0.23063，内部BA却只有53.61%；λ=.1的内训BCE0.42106、内部BA56.12%，最后外部BA51.82%（CV阈值）。这与高维有限样本的拟合／泛化不足相容，不能把低训练loss当成功。本有限网格、两种池化和80步预算不能证明所有线性头或L26本身没有Mixed信息；5/6候选到达迭代预算，完整梯度残差已报告，不宣称达到全局数值最优。未为追求更好数字扩网格、追加外部候选或调参。

## 真实执行与资源

原debug纯CPU12核/32G/30min、GPU0的 `sbatch --test-only` 失败，实际分区唯一节点ACD1-63为drained，未提交作业，也未申请GPU绕过。父任务明确授权后，使用新独立login CPU2分支：单进程亲和性限制为2核，3GiB RSS watchdog、26min预算、flock/intent和实际PID身份。未安装包、改环境或影响已有GPU作业。

8项synthetic测试实际在HPC3通过。第一次CPU2资格测试暴露NFS下临时mmap未先关闭导致清理失败，修成finally关闭，旧attempt保留；之后新immutable v2实际运行。观测由输入验收→2305/8192题真实span reduction→完成6模型拟合与唯一外部评分；完成162.07秒，峰值RSS1813012480B≈1.69GiB，CPU2/GPU0，无failure，原PID1399210已经退出。

源代码已发布：初版964baac、CPU2/内存边界0013b62、NFS测试修复及实际stage a8bdba2e3d25558d73b6c55b6de15b8616806b3e。活动v2代码未热改。OS默认Python3.6没有numpy，因此最终只读logits汇总使用已经安装的同Python3.10 runtime，没有安装依赖。

## 结果与资产SHA

- 根目录：`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-linear-group-head-cpu2-v2-20261021`。
- `output/complete.json`：7030B，SHA `f87ed1a0b64a7c4bf0dfd0ac6dc7e127a1c38d43e5f8fe86e7c960d4cba4d002`。
- 内部冻结文件：3543B，SHA `0b7d9a2b85bcccf40db94cebe1bb87d493921b33146ba4611096a5b9aeda1e45`。
- 只读双阈值审计：8945B，SHA `53f5dd595ce00f51dc199ff7ae08b70c875c73433bff4b74541377615d8fe00e`；复核了原CV指标完全一致、两阈值AUC和Top512一致及所有9源码SHA。
- **全部系数/scaler仍在远端** `output/winner-weights.npz`：87048B，SHA `f50a4863a9b90073e65e09da3e68391c21476f597407b9b4a45d9e1e323db603`。本地仅保存形状/路径/SHA摘要，无参数数组。
- 逐题预测仍在远端 `output/external/predictions.jsonl.gz`：90492B，SHA `bc43be676e3e7d1f89f8eb4713486ac60329c85e2be4531185013219ab909fda`，未复制到Mac。
- 实际config的1773B已与远端SHA `592fea700ad940b45db2fe146c53c727adb5d68e8c81a34348bfb1acff654464`核对，本地 `actual-config.json` 只含模式与路径/SHA，不含凭据、系数或tensor。
- 全量policy及source/row-token绑定证明保留远端 `output/train/input-audit.json`（571777B）和 `output/external/input-audit.json`，本地仅其ref。

本次诊断结束，不将该头接入生产，不改变当前actor/探针/Adam，不新增实验作业。

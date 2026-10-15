# H100八卡探针实验：正确对照与慢更新原因

2026-10-16更正：用户指定线程01a0fcbb-863f-7450-926d-db537f648265中的H100实验。此前A800比较选错对象；“探针仅几秒”的结论不能用于本实验。本次只读原日志与冻结源码，未启动、停止、恢复或热改任何作业；没有导出问题、轨迹、tensor或模型。

## 正确身份

- h100-8八卡H100，job3402，run `f77c3245dcfe49a88a5a99b1dec161e6`。
- source `c5fc09603a10244655b98ec0ce8eada32668f83b`。
- 远端base `/mnt/data/shaoshitong/search-g1-qprobe3-binary10-c5fc096-20261015`。
- 本地对应冻结源码 `/Users/shaoshitong/Documents/Search-G1/qprobe-h1008-transformer3`。
- 每轮5120候选打分，Top448＋随机64，选512题生成2560轨迹。
- 八卡训练actor；**探针只在GPU7单卡RPC执行，三个完整Qwen decoder blocks全部训练，强制FP32，train_micro=4，逻辑batch256，每轮全replay训练2epoch**。

## 同为八卡H100的一次普通iter

| 平均秒/iter | mixed-first，187轮 | 正确Q-probe，18轮 |
|---|---:|---:|
| 完整生成量 | 10240条 | 2560条 |
| 候选评分/采样/筛选 | 820.504 | 271.385 |
| Reward探针＋优势 | 115.832 | 76.748 |
| Actor，20次实际Adam | 130.123 | 151.854 |
| 在线question-probe完整更新 | 不适用 | **389.469** |
| 其他训练内工作，含可用的ref | 122.351 | 86.059 |
| **记录内iter主流程** | **1188.809＝19分49秒** | **975.514＝16分16秒** |

正确Q-probe已实读1–19共19个完整outer；第5轮含保存/验证，排除后18普通轮。第20轮只有采样无完整提交，排除。两份console重复38行逐项一致，去重为19轮；实际更新/selection计数核对通过。之前“18轮”只是更早快照。Mixed-first沿用已核验run15bb/job3381的187普通轮，不混入废弃job3368。

Q-probe各阶段份额：采样27.82%、reward7.87%、actor15.57%、**question-probe39.92%**、其他8.82%。生成量减少75%，collection省9分09秒，但在线探针又用6分29秒；观察到主流程只少3分33秒，约17.94%。双方采样结构、轨迹与运行负载仍不同，不将此差值归因为单一算法机制。

## 为什么第14轮更新要8分44秒

这是具体实现工作量，并非小线性头更新：

1. 只有GPU7参与head，其他七个actor rank不分担其反向计算。`qprobe_sequence_rpc.worker_transport`只取`group.workers[7]`。
2. 三个完整Qwen blocks可训练，`OnlineSequenceQProbe`显式将模型转FP32，forward没有autocast。
3. 第14轮replay12288题，训2遍，共24576题次；逻辑batch256，实际96次Adam。
4. 物理micro4，所以每次Adam串行累积64个micro；共 **96×64＝6144次micro前向＋反向**。
5. 训练结束还对全部replay做一次postfit预测；predict_micro32，对第14轮另有384次前向。
6. 还有逐micro的CPU特征→GPU FP32搬运、padding分配，有限性扫描，完整模型/Adam/replay状态回滚备份与RPC、权重hash和票据发布。

第14轮内部更新483.909秒，外层完整更新523.527秒＝8分43.5秒，差39.618秒。外层含状态/RPC/提交，内部也含postfit预测，因此不能把8分44秒全部叫纯GPU backward。上述各子项没有独立profiler，不伪造计算/拷贝/扫描各占几分钟。

| outer | replay题数 | Head Adam | 内部更新秒 | 外层完整更新秒 | iter主流程秒 |
|---|---:|---:|---:|---:|---:|
| 1 | 5632 | 44 | 172.335 | 202.906 | 763.128 |
| 14 | 12288 | 96 | 483.909 | 523.527 | 1087.638 |
| 15 | 12800 | 100 | 515.901 | 550.072 | 1116.934 |
| 16 | 8192 | 64 | 332.232 | 367.825 | 915.794 |
| 19 | 8192 | 64 | 317.837 | 343.756 | 879.355 |

正式首轮保留5120预热样本再加入512题，replay5632；不是A800后来的清空teacher方案。第16轮预热source_outer=0超出16轮age窗口，replay12800→8192，Head Adam100→64，直接解释15→16更新耗时降低。平均389.469秒是外层计时，内部356.340秒是子集，不再相加。

## 优化方向与执行状态

首要方向是修正**单卡FP32、micro4的串行训练路径**：采用八卡DDP，保持global batch256、每卡32，减少小micro累积；BF16/autocast作为单独数值验证项。还可缓存/批量搬运特征，减少重复完整state/RPC/hash，以及将无必要的全replaypostfit诊断改成明确频率。实际能快多少需要同一数据、同一更新预算的测量，不能先承诺八倍；减少replay或epoch会改学习配方，不作为透明提速。

本次仅定位原因与对比，没有实施这些优化，也没有干预另一线程正在运行的新prefix实验。

## 计时与证据

`iter主流程=candidate_fill+timing_s/step`，不是把step误当整轮。collection含agent生成/检索/评分/包装，mean271.385秒；其中agent交互213.036秒，余58.349秒不能全称head评分。一次性warmup/cache/模型加载，以及未计时边界操作不冒充普通iter成本。

完整19轮原console/selection/update标量在 `remote-scalars-first.json`，逐轮复算 `comparison-analysis.json`，compact `summary.json`。第二次实读11个关键文件完整SHA全部一致，另核本地相关源码与c5fc Git版本逐byte相同。读取脚本 `read_h100_qprobe_timing.py`；分析脚本 `analyze_comparison.py`。

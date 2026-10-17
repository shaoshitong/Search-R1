# I1–I6修复后的五轮恢复跟进（2026-10-17）

实际metadata时间：2026-10-17 16:19:01 Shanghai。当前同run `54ac33dcebe64e62a11fe4177f41f116`／execution `a4a0c2a18cca40bc8c3850d40eed8554`，runtime `75d536575593891876df339467da73b3c4cbdc7c`，只读operator `e2c1353c7b6021eb121056ec98489efe07b8feeb`。

只修正式训练I1–I6，保留现有question＋Prefix5探针初始化和trajectory BCE＋G5 mixed BCE的1:1损失。未重新开启独立warmup、未重置actor/probe Adam、scheduler或RNG，也未调用子智能体。

从完整4／80次Adam状态恢复后，第5、6、7、8、9轮已连续完整提交。每轮四rank各处理20机会、返回20次Adam、0skip，四rank副本只计一次；新增100次，累计180次Adam，scheduler9。各轮原始complete JSON SHA、实际Adam和学习率位置保留在同名JSON中。

|完整outer|实际候选题|选中真实Mixed／512|比例|新增成功actor Adam|
|---|---:|---:|---:|---:|
|5|2560（恢复前已预留）|412|80.46875%|20|
|6|5120（新规则生效）|396|77.34375%|20|
|7|5120|388|75.78125%|20|
|8|5120|364|71.09375%|20|
|9|5120|366|71.484375%|20|

第6轮普通规则的全链已实际核实：5120题／25600条Prefix5→Top512题→2560条完整actor训练轨迹→20次成功Adam→joint。第5轮旧候选预算与第6轮起新规则分段，不把候选变化当成单因素效果结论。

同一pipeline已自然进入第10轮，当前实际2560题／12800条前缀已完整生成并评分，正在完整生成全部未筛选候选。后续才进行strict EM Mixed优先的512题actor选择和全部12800条标签的probe训练。当前尚无完整full-pool或probe50更新完成回执，因此50次更新仍待真实验证；没有为验证额外插入周期。

现成QA5：strict EM136/512=26.5625%，F1=36.06585167224031%，固定512身份SHA `3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb`，新共享BF16交互协议、当前actor0005。未新增评测或放宽评分规则。

I1/I5/I6共用交互、BF16及流式action终止；I2保留原micro2等权归约；I3已接通真实grounding的50/100/150拟合与adopt生命周期，但这些实际刷新点尚未到达；I4保留采样策略／ratio-tail／finite等防护及真实更新计数。代码修复与新结果不是全部实验成功或因果收益证明。

总200目标继续，正常30分钟只读监控；当前尚未完成final51713、最终checkpoint finite/SHA及only-owned收尾。所有模型／Adam／原QA／token／完整rollout留远端，Git仅代码、标量和哈希。

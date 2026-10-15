# 递归消融已闭合评测总表

本快照合并已接受的原生汇总报告：计划标签节点21/44，独立新增训练节点5/28；另外列出A的4个参考节点。复用A的budget10、position-uniform、generation4，以及复用C1的generation1，不代表新增实验。

- [中文汇总](findings-zh.md)
- [曲线数据](curve.csv)
- [原始字段与逐点来源](aggregate.json)

截至本快照，C1完成150轮；C2只有第5轮正式评测，其余条件待继续。C1的150轮与A均为246/512；C2的5轮为100/512，A同轮为96/512。单seed的汇总差值不能证明机制收益。仅使用已有结果，没有追加推理、逐题配对或bootstrap。

合并器源码commit `b565fbc89493aefa6896fb180be145cde1e840a8`，输入为已接受的小型export capsule（SHA见aggregate.json）。所有成员大小/SHA、正式512计数、参考EM和复用关系已核对。原逐点报告的成本覆盖标记保持原样，A完整前缀及C1故障前复用采样的补充成本分别见[参考成本报告](../recursive-ablation-A-retained-cost-first150-20261015/findings-zh.md)与[C1完整报告](../recursive-ablation-C1-outer150-20261016/findings-zh.md)。不可将复用标签成本重复相加。

这是完整已闭合节点的阶段总表，整个150轮消融目标仍在执行。训练终态、剩余23个标签节点、剩余独立条件及归档需分别完成。

# A800 Prefix5/P10 监控输出修复 — 2026-10-18

00:26:46的唯一一次正常只读观察，在远端`emit`处触发128 KiB输出断言，stdout为空。该回执不证明训练退出，也没有新的进度、评分或故障终态。保留原失败回执，不重复SSH、不启动或停止trainer、不修改SSH/knownhosts/auth。

最近一次成功观察仍为2026-10-17 23:56:30：完整50/200、actor1000次Adam、probe250次Adam、QA50 EM226/512、F1=0.5276351354935914；当时51轮actor已开始。不能把这些历史值称为本次当前状态。

原输出重复包含最新actor reward中完整grounding状态及四个reward rank副本，也在jobs/live_jobs重复输出完整存活子进程身份数组。修复只改变本地发布的只读入口`reproduction/operations/a800_prefix5_variable_pool_control.py`在内存中投影返回值的方式：

- 保留reward均值、advantage统计、真实G5分类、grounding版本/来源actor/计数/manifest、head和adoption SHA。完整grounding状态以规范化JSON SHA绑定，rank内重复状态仅保留其SHA；其他rank字段不改。
- 所有原进程身份/出生时间存活检查仍在远端执行。返回job leader身份、状态、退出码、完整birth计数、存活birth精确数量与规范化SHA；省略重复的子进程身份数组，并显式标记投影语义。
- source/run/execution、四rank计数和恢复、完整outer判断、小时窗口、原生QA与probe指标仍按原入口检查和投影。没有将缺失值改为0，也不靠截断掩盖未知状态。
- 保持同一control.py物理路由、单次65秒SSH、64 MiB metadata读取和128 KiB输出上限。超限错误现在附带实际字节数，不放宽上限。

正式runtime仍为`75d536575593891876df339467da73b3c4cbdc7c`，trainer源码、优化器、scheduler/RNG/pool和算法均不修改。本入口发布后的第一次实际读取，留给下一次正常30分钟检查；发布与本地格式核对不等于新的远端进展验收。失败观察不推进真正已送达的houranchor49。

同一调用同时出现SSH的hostkeys RSA签名警告，但远端Python确实运行至输出断言；没有据此修改SSH设置或归因于训练故障。

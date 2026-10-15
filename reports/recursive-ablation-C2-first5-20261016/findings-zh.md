# 递归 Tree-GRPO 消融结果

只使用完整原生评测与已有日志，按最新授权仅汇总EM，不追加逐题捕获或评测。

| 条件 | outer | dev512 | NQ | Hotpot | 真实Adam/rank | 跳过/rank |
| --- | --- | --- | --- | --- | --- | --- |
| A-original | 5 | 96/512（18.7500%） | 54 | 42 | 未知 | 未知 |
| C2-random-final | 5 | 100/512（19.5312%） | 59 | 41 | 100 | 0 |

与A同轮次的总体EM差值：
- C2-random-final，5轮：0.7812pp。

耗时分量可能重叠，不能将它们相加当总耗时。GPU小时只按记录的outer wall×卡数计算，不是GPU实际忙碌时间；生成token遵循原生记录，包括padding与丢弃尾部。
实际记录范围、缺失覆盖、树统计与采样峰值见aggregate.json。单seed不证明跨seed稳定因果。

# B128训练曲线：固定100个更新窗口

这是主作业746266现有 `curve.jsonl` 的只读训练抽样汇总，快照截至cursor2048。前三个完整窗口的Mixed BA为54.2344%、55.0293%、55.0898%；loss先从0.68715966降到0.68519669，随后微升至0.68536795。这里只说明训练抽样上的拟合变化，不能当作独立测试或与CP512固定测试混用。没有模型、阈值、评分或训练操作。

| 更新窗口 | 完整更新数 | 问题访问次数 | 真实G5 view次数 | 加权Mixed BCE | Mixed BA % | Mixed recall % | nonMixed recall % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1676–1775 | 100 | 12800 | 51200 | 0.68715966 | 54.2344 | 58.0078 | 50.4609 |
| 1776–1875 | 100 | 12800 | 51200 | 0.68519669 | 55.0293 | 58.5977 | 51.4609 |
| 1876–1975 | 100 | 12800 | 51200 | 0.68536795 | 55.0898 | 57.7266 | 52.4531 |
| 1976–2048（不足100步） | 73 | 9344 | 37376 | 0.68426233 | 55.3430 | 58.5135 | 52.1725 |

| 更新窗口 | 聚合TP | 聚合TN | 聚合FP | 聚合FN | Mixed / nonMixed views |
|---|---:|---:|---:|---:|---:|
| 1676–1775 | 14850 | 12918 | 12682 | 10750 | 25600 / 25600 |
| 1776–1875 | 15001 | 13174 | 12426 | 10599 | 25600 / 25600 |
| 1876–1975 | 14778 | 13428 | 12172 | 10822 | 25600 / 25600 |
| 1976–2048 | 10935 | 9750 | 8938 | 7753 | 18688 / 18688 |

口径：loss按每步真实G5 view数加权；每行512 views，因此完整100步窗口的权重为51200。先聚合TP/TN/FP/FN，再计算两类recall和BA=(TPR+TNR)/2。每题四views均权.25，访问次数允许跨步重复，不能当作unique问题数。表中末窗口1976–2048只有73步，未与完整窗口等量。

逐行检查373/373通过：cursor从1676连续到2048，native/head两Adam时钟与cursor相同，source_cursor始终1675，时间不逆序；logged questions/effective_batch均128，512 views、Mixed/nonMixed各256，混淆总数及两类分母完全匹配。所有行实际micro64，但effective batch保持128；无缺失字段、无坏时钟或混淆行。未报告trajectory accuracy。

输入是活动文件的一次已完成行前缀快照，不代表之后增长的整个文件SHA。读取640085B，无尾部不完整行；前缀SHA256 `80e06637a78c99de823ef48b2626e9f3fbac5818b244a13605bc9c9fc54cef79`，inode11129927147，uid21930，快照epoch1792609272.613778。路径：`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1/curve.jsonl`。

通过既有 `outputs/qwen7b-warmup-20261019/balanced-control.py` 在远端执行纯stdlib只读代码，未读取逐题payload/模型/CP/评测结果。源码 `reproduction/operations/qwen7b_warmup/summarize_mixed_direct_b128_curve.py` 已先行发布 `7ac739c7231c65dfd58f79f05f7b7af249f94da8`；邻接 `actual-summary.json` 保留所有小聚合及字段检查计数。当前HEAD/index、主作业和运行源码未变。

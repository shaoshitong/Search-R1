# Question + prefix5：首个完整epoch（2026-10-16）

2026-10-16 22:39:08（上海）实际只读回执确认：**已完整提交1/200 epoch**，同一进程已进入epoch2，完成150次新更新，累计**8850次Adam／EMA**。四rank一致，副本只计一次；无记录错误、退出或nonfinite拒绝。

首轮完整遍历445440个同事件G5／2227200条训练轨迹，完成8700次Adam与EMA；固定测试为512个G5／2560条。下表均使用首轮结束时的**同一个EMA0.99权重**。

| 指标 | 完整训练集 | 固定测试集 |
| --- | ---: | ---: |
| G5混合balanced accuracy | 67.7370% | 54.3517% |
| G5混合普通accuracy | 70.5238% | 58.0078% |
| 逐轨迹正确性accuracy | 79.6537% | 77.3047% |
| 逐轨迹BCE | 0.474310 | 0.567329 |
| G5混合BCE | 0.602849 | 0.753820 |
| 两项总loss | 1.077160 | 1.321149 |

当前测试混合balanced accuracy仅略高于50%基线，训练与测试相差13.3854个百分点，尚不能宣称泛化良好。测试逐轨迹accuracy为77.3047%，低于全部预测错误的78.1641%基线；测试G5普通accuracy也低于全部预测非混合的64.2578%基线。这里的balanced accuracy、普通accuracy和两类基线必须分别理解。首轮结果不足以单独证明严重过拟合；按既定授权继续200个完整epoch，记录后续曲线。

原生首轮耗时2499.441秒，约41.66分钟，包含训练与既定全量EMA训练／测试评测，计时在checkpoint保存前结束，不能当作孤立optimizer训练耗时。raw在线窗口随权重更新：逐轨迹BCE为0.616493、G5 BCE为0.820846；它们不是训练后同一EMA权重的指标，不能据此作受控前后loss比较。

运行与证据：

- run：`8af926a49630428097b94027864da68b`；execution：`091f3979549347d190065387c2214af6`。
- 活跃runtime源码：`98b5ec2af9a5b45cd535e67408980c72a75d7746`，四rank均实际核验925个源码文件；此次仅发布标量与哈希，不改活跃源码。
- 首轮原生metrics／checkpoint／complete记录一致，`fully_committed=true`；四rank首轮Adam均为8700，EMA为8700。
- 首轮完整state为远端`train/latest.pt`，1242422752字节，SHA256：`a3a4959f14413948f88c9a7add54eb670746fa63005972390c0ab907d30a5f8c`。它是可被下一完整epoch替换的latest状态，**不是不可变epoch1权重**；不可变状态仍按每10epoch及最终200保存。
- 首轮绑定SHA256：`80c0abea4462b1bfb6852b0f33a79258ad61f8a17a89fe47a8ff2e0e830580c6`。
- 原始只读回执：`preparation/question5-slim-progress-observation-1792161548860011000.json`；完整标量、混淆计数及证据哈希见同目录[JSON报告](question5-ema-joint200-epoch001-2026-10-16.json)。

本次未新增rollout、EM评分、模型／tensor读取、额外测试或review。历史1500更新启动报告保留；本报告不完成或恢复原暂停actor任务。

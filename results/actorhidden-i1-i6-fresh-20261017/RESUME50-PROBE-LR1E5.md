# Complete50恢复并将选题探针LR改为1e-5

最新人工指令：`需要变成1e-5跑，探针部分`。范围为选题正确性探针，actor与grounding奖励探针配方保持不变。

## 实际执行

- 旧3431在完整58后受控停止。原inode日志SHA和自有进程/cgroup/GPU释放已读回；这是人工LR调整，不是CUDA/OOM失败。
- 按人工批准从独立完整SHA和CPU状态读回通过的50恢复。51–58未拼入新路径。
- 新3434/h100-8逻辑run保持6cd6f043ef1641ae91ac177b4c27fe4a；冻结runtime59dcbf0cc1da68e40974b652f92978ce6ded3766；1372目标代码逐文件SHA通过。
- 保留actor1000/rank、scheduler50、全部actor状态/RNG、题池cycle1/cursor87490/累计256000，以及真实拟合并采用的grounding50。
- CPU实际重载证明14份探针模型tensor、Adam动量/step200、8rank RNG均与原40更新后的已训练探针相等；仅优化器paramgroup、LR binding及隔离恢复路径改变。实际序列化LR=1e-5，没有reset或额外probe50更新。
- 原QA50完整512题、8worker exit0、EM190/512=37.109375%、F1=45.4629451%复用。未重新生成或将其归因为本次LR调整。

## 实际恢复后进展

原始8rank日志已逐一核对：51–55连续五轮，每轮每rank20次真实actor Adam写入，累计1100/rank；56真实前缀生成完成并进入score。个人只读观察器77639已自然exit0，初期五轮验收完成。同3434继续200，不执行201。

详见 `resume50-probe-lr1e5-five-complete.json`。该小证据核对更新账本和原生前缀捕获，不冒充最新55权重的独立完整文件SHA/CPU重载，也不构成质量收益证明。

## 后续探针更新

每20轮更新的协议保持不变。下一次在60，用同一actor快照补齐本轮全部5120题G5共25600轨迹，自然遍历一个epoch，100次同步Adam/rank，LR1e-5。其后61使用新head。60的实际1e-5写入、同一rollout更新前后train/heldout混合普通/平衡accuracy与两项loss尚pending，不能用配置/提交冒充完成。

同一正常间隔的单次只读等待已安排；不是重启App暂停的自动监督，也不改变训练。所有模型、Adam、hidden、完整轨迹和评测输出均在远端。

## 代码范围与恢复

`resume_probe_lr50.py` 保留已训练状态并仅覆盖LR；初次50人为回退和后续故障恢复分开。后续故障从当时实际最新完整点恢复，不能因旧helper退50。`resume_probe_lr50_chain.py` 在一个实际连接内依次核对终态、隔离代码部署、CPU导入和once提交，所有已有intent/receipt均先核对，禁止盲重放。

聚焦CPU测试10项通过。正常commit/push及独立全Git文件SHA已执行；后续结果commit不热改冻结训练源码。

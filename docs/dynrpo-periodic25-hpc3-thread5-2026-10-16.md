# 本线程 DynRPO 每25轮探针实验

独立实验以发布基线 `1af0d14937a3da6666eec63e8caa50c192d481e6` 为来源；代码、输出、锁与其他线程分离。实际运行版本以每次命名空间的 source-publication、run-manifest 和 Slurm 回执为准。本文说明执行契约，不代表正式训练已通过验收。

## 用户确认的最终配置

- Student 从官方 Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` 开始，四卡，完成200个完整 outer。
- 探针更新点为0/25/50/75/100/125/150/175。用户明确确认第200轮只保存和评测，不再训练探针。
- 始终继承同一个原生最后 decoder block35、readout、Adam 和 private RNG；不得在阶段切换时复制 student 参数或重新初始化探针。已有合法 warmup10继续到20 epoch，保留总计400次 Adam。
- 每个后续探针阶段使用当前 student，对5120个不同问题各生成5条真实 rollout，按 strict EM 构造标签，再拟合20 epoch。目标为原有 BCE + 0.3 normalized sample-std MSE，无跨阶段 replay。
- outer0可以使用已验收缓存；之后选题与探针训练使用当前 student 的 layer27 全 token BF16 特征。特征 detach，远端 RAM 内使用，不持久化，不传回本地。
- 25/50/75/100/125/150/175/200首先保存完整恢复状态；探针训练前后阶段分别验收。每rank模型、Adam、scheduler、RNG、driver游标、探针Adam/RNG及阶段均须实际可加载且SHA通过。
- 固定512题验证节点5/50/100/150/200；完成200后执行完整51713题最终评测。保持共同数据 manifest `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24` 和原有 student 奖励、优化器预算与超参数。
- 初始300题校准用于沿用的 Search-G1 双奖励 readout；它不同于5120题的选题探针 warmup，不得用它替代选题探针或重置后者。

## 环境、执行与修复

`reproduction/hpc3_dynrpo_periodic` 保存独立部署、连续 warmup、绑定、提交及恢复入口。`bootstrap.py` 引用既有只读基础环境，同时将本实验自己的 `env/overlay` 放到依赖搜索路径前。缺失校准依赖只安装到本实验overlay，不改共享环境；校内源和包版本由 `environment_overlay.py` 固定。正式提交前必须完成安装及真实CPU校准拟合/序列化回读检查。

训练开始前的失败只能在实际确认没有 student launch claim、更新或 checkpoint，并且原计算节点清理通过后，使用 `retry_before_training.py` 保留400 Adam的探针状态重试。训练开始后必须走真实完整checkpoint恢复，不得使用此前入口重置 student。

`reconcile_cleanup.py` 仅在旧失败作业原计算节点上的新 CPU Slurm allocation 内检查旧作业资源。它复核账号、作业、节点、出生身份、所有本任务session、CUDA上下文与私有staging；登录节点的 `/proc` 或 `/dev/shm` 不能作为计算节点清理证据。源模型、数据及checkpoint不由此清理。

## 最终验收和评测补全

`final_audit.py` 独立核验200轮实际更新账本、保留的partial更新、全部周期checkpoint、8个连续探针阶段、7次在线rollout/refit、固定验证和最终评测。它不生成缺失的训练记录，不把checkpoint200伪装成原始第200轮完成日志。

`final_eval.py` 用于真正存在且通过认证的CP200评测恢复。它使用无actor优化器的rollout角色，只补缺失的512/51713评测；64题标量chunk可经认证后继承到新命名空间。它不能恢复student训练，也不能补造原训练row200。

大型checkpoint认证/最终审计必须在实际本人所有、RUNNING、当前节点匹配且实际AllocTRES及cgroup允许至少64GiB的CPU Slurm allocation内执行；8GiB登录节点不得加载四rank Adam。实际GPU保存/恢复及最终验收只有运行回执通过后才算完成，纯CPU测试不能替代。

新启动或恢复必须观察连续五个完整outer和下一轮继续；提交成功或RUNNING不代表有效训练。低优先级先尝试，依据真实排队情况逐级提高，不默认使用emergency。不给本任务新增已被用户取消的定时监控。

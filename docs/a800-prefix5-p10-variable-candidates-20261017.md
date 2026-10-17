# A800 Prefix5 / P10 候选题数切换 — 2026-10-17

本次在当前 I1–I6 run 的完整 outer 边界续训，保留 actor、probe、两者 Adam、scheduler、四 rank RNG 和原题池 permutation/cursor。不会重新开始官方模型训练或重复完整更新。实际生效轮次由停止边界 `N` 确定，为 `N+1`，记录在新执行的 `preparation/launch-config.json` 与恢复回执。

| 轮次 | 候选题 / G5轨迹 | actor 使用 | probe 更新 |
| --- | --- | --- | --- |
| 生效前历史1..N | 保留原始2560 /12800记录 |512 /2560 |保留已发生记录 |
| 普通轮 |5120 /25600真实Prefix5 |预测Mixed概率Top512，续写原五条前缀 |0 |
|10、20、…、190 |2560 /12800完整轨迹 |真实strict EM Mixed优先，按探针分数排序；不足512以既有分数补齐 |全部2560G5，一遍，50次Adam |
|200 |5120 /25600真实Prefix5 |预测Mixed概率Top512，续写原五条前缀 |0 |

更新轮 probe 的 LR=5e-6、global batch平均256条轨迹、1 epoch 均不变。两种轮次 actor 都是512题×5＝2560轨迹、mini128、20个更新机会。I1–I6 actor loss、数值保护、57完整outer warmup、全正确组虚拟0、grounding刷新、评分和交互协议保持原实现。

`candidate_questions=2560` 只保留旧配置和 probe 训练合同的历史字段。生产候选数量统一从 `candidate_schedule.candidate_count(config, outer)` 获取，不能直接读取该旧字段。累计题池消费量通过逐段求和计算；保留原 permutation、seed、cycle 和 cursor，不将历史消费重算为 `outer*5120`。跨题池周期保证同轮无重复问题。

## 执行顺序

1. 在隔离 worktree 修改、提交并推送新源码及以下入口；核对远端 Git SHA/blob 后精确 stage 新恢复 namespace。
2. 已发布的 `reproduction.operations.a800_i1_i6_boundary_handoff` 只冻结轮询 parent，允许当前普通轮 actor 子作业自然完成并保存。真实 wait status0、四 rank完整机会/返回Adam及全部 owned 服务退出后，调用原不可变源码的 metadata commit 完成同一普通轮 joint。边界回执明确记录这一 CPU 收尾；不运行额外训练、评测或读回模型。拒绝更新轮、里程碑轮和任何下一轮预留冲突。
3. `reproduction.operations.a800_prefix5_variable_pool_control --action prepare --handle ABS_HANDLE` 对真实停写边界做一次完整 SHA/finite/optimizer/scheduler/RNG/probe/题池资格核验，硬链接精确原状态；CPU证明不冒称 CUDA 恢复成功。
4. `--action launch` 仅执行一次。新 runtime 的 actor wrapper 校验显式 source compatibility 后恢复原科学绑定；源码、runtime 身份与旧 checkpoint 来源分别记录。每次实际子进程恢复继续进行原生状态检查。
5. `--action observe` 为唯一有界只读监控入口。普通30分钟检查不添加GPU测试、额外评测、概率扫描或重复源码审查。

恢复只允许枚举的候选调度和控制代码差异。actor数学源码逐字相同，probe训练/optimizer/加载关键AST相同；本次候选差异绑定最新人工指令 SHA。checkpoint、日志与旧源码完整保留，禁止将新 runtime 冒称旧 commit。

## 实际验收及历史分段

- 首个新普通轮：真实5120题／25600前缀，真实Top512／2560续写轨迹；随后四 rank完整actor机会/Adam及joint保存，核对至少两次正向流程推进。
- 下一次自然到达的更新轮：真实2560题／12800完整轨迹，actor512题／2560轨迹，probe每rank50次真实Adam（四rank副本计一次）。未到达时明确记为待验证，不能插入测试更新轮。
- 历史曲线以 `effective_from_outer` 分段标注；旧2560与新普通5120对应的筛选比例不能混称同一候选预算。真实EM oracle更新轮也须与普通预测筛选分开。
- 保存5/50/100/150/200、最终51713评测及only-owned资源收尾要求不变。未完成200或最终验收不得标记目标完成。

大产物全部留在远端。Git仅源码、配置、执行文档及去敏scalar/hash；不得加入权重、优化器、原始题目、token、rollout、环境或凭据。

# A800 Prefix5 / P10 候选题数切换 — 2026-10-17

本次在当前 I1–I6 run 的完整 outer 边界续训，保留 actor、probe、两者 Adam、scheduler、四 rank RNG 和原题池 permutation/cursor。不会重新开始官方模型训练或重复完整更新。实际生效轮次由停止边界 `N` 确定，为 `N+1`，记录在新执行的 `preparation/launch-config.json` 与恢复回执。

**现场恢复补充（2026-10-17 14:24）：** 原79c执行在第5轮续写rank1的vLLM TCPStore初始化发生`EADDRINUSE`并自行退出。native cleanup证明全部owned进程已退出；最后完整joint4／80次actor Adam，probe0，第5轮尚无actor或probe更新。计划中的交接入口在发信号和写onceclaim之前拒绝执行，没有人为打断训练。

本次因此直接从完整joint4恢复，并保留已经真实完成的第5轮2560题／12800前缀、Top512评分和after5题池游标；失败续写的原日志保留，不导入未来partial。**第5轮继续旧候选预算，新规则从第6轮生效。** 这不是将已完成更新重跑或将数据游标退回4。资格证明必须覆盖pending5全部前缀/评分文件与原始SHA，下一轮预留transition与joint4历史快照分别保留。不要再对已退出的原进程调用handoff，也不要启动此前仅stage的69e0命名空间。

新恢复源码额外修复这次真实的vLLM端口初始化冲突，绑定故障终态SHA；模型、采样seed、请求参数、轨迹与训练数学不变。该修复仅涉及初始化通信端口隔离，无额外GPU验收或评测。

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

## 2026-10-17：保留pending5与已清理历史快照的恢复修正

最新人工再次确认：只修正式训练I1–I6，保留当前探针初始化，不重新开启200epoch warmup；Prefix5与trajectory BCE＋G5 mixed BCE的1:1损失不变。I1–I6的代码对应关系见[a800-prefix5-i1-i6-fresh200-20261017.md](a800-prefix5-i1-i6-fresh200-20261017.md)。

同run54ac原执行在第5轮continuation初始化因TCPStore端口冲突真实退出。已完整完成4轮，四rank各80次返回Adam（副本只计一次80），scheduler4；第5轮已完成2560题／12800条前缀和Top512评分，但没有actor/probe更新。恢复保留这些真实产物和已预留题池，因此第5轮沿原2560配置完成，新的普通轮5120规则从第6轮生效，P10更新轮仍2560。生效前历史与生效后曲线分段，不能把待验证计数写成已经运行。

CPU恢复入口原来把被正常存储清理的第1至3轮probe快照当成必要外部资产，导致在加载当前状态之前退出。现在历史轮只核对保留的JSON lineage，当前完整joint4的probe payload、快照同inode、完整SHA、finite、actor/Adam/scheduler/RNG与题池仍严格验收，缺失当前状态继续拒绝。另修复driver里“probe回执JSON的SHA”与“回执value中的权重payload SHA”误比较：分别校验原JSON序列化SHA及模型SHA，不放宽两种身份检查。

这些修正只涉及恢复元数据；actor数学、探针模型与损失、grounding和交互实现不改变。失败的CPU准备namespace保留且不重放onceclaim；新版使用独立执行namespace。CPU证明不等于CUDA加载或训练推进；实际普通轮5120／25600、actor512／2560以及自然更新轮50次探针Adam，均须以随后真实日志验证。

恢复操作入口必须在`/data/search-g1/env/bin/python`下验证源码AST。物理SSH入口默认系统Python与训练环境版本不同，虽然每个文件SHA完全相同，Python AST字段差异仍会造成假拒绝。本次只在本地已发布操作入口为原路由增加stdin读取前的execv；远端已部署source75及已验收状态不修改，CPU fullSHA/finite证明直接复用。原launch在`prepared()`阶段、onceclaim之前被拒绝，无pipeline/submit intent；修复操作入口后才进行首次实际launch，不重放已发生训练。

只读监控现投影现成milestone QA标量和其metadata SHA，不触发生成或重评分。更新机会中的真实skip允许成功写入对应不连续opportunity编号；复用的I1–I6 reader已经严格检查0..19内递增机会与实际恢复Adam连续计数，保持该检查，不再覆盖成旧版reader的断言。此变更仅在发布的本地操作入口，运行source75未改。

恢复后的连续五轮跟进仅读取已有`complete.json`标量与SHA，最多五条，核对每轮真实20机会／保留Adam／skip／scheduler及probe边界；不加载checkpoint，不为验收重复训练或插入评测。

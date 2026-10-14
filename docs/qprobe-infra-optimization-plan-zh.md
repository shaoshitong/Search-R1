# Search-G1 / QProbe 四项 infra 改造方案

源码基准：`c10c295e465e13cd038adf8e4f4f0840a5722d91`，分支 `codex/qprobe-resilient-infra-20261014`。

本方案只整理执行路径优化。重复监测、零KL reference前向、零entropy词表计算已有源码修复；逐轨迹异步还需迁入。发布源码不等于已部署到现有训练。无需另建GPU验收、压测或额外评测流程。

| 改造项 | 目标 | 当前源码状态 |
|---|---|---|
| 重复前向 | 上次成功post检查复用为紧邻下次pre检查 | 已实现 |
| 批次等待 | 独立轨迹交错生成/检索；整轮只同步一次采样权重 | 待实现，可参考已发布CoSearch框架 |
| KL系数为0 | 跳过没有消费者的固定reference前向和KL损失计算 | common8的零KL loss路径已实现 |
| entropy系数为0 | 完全跳过词表entropy计算 | 已实现 |

## 1. 去掉重复安全监测前向

原顺序：`pre1 → Adam1 → post1 → pre2 → Adam2 → post2 …`。

修改后：`pre1 → Adam1 → post1 → 复用post1作为pre2 → Adam2 → post2 …`。

仅复用同一轮、紧邻上一mini、实际计算且allowed的post报告。模型/参数/buffer版本、冻结样本、温度、阈值、校准身份、attempt和执行配置都必须一致；所有rank先一致决定是否复用，再进入模型前向。写入后计算新的post，回滚、拒绝、更换样本、新attempt或恢复均使旧缓存失效。

实现位置：

- `verl/workers/actor/dp_actor.py`：`_monitor_old_policy`、`_monitor_cache_key`及失效路径。
- `search_g1/monitor_forward_reuse.py`：复用来源及真实计数。
- 已有worker/update/checkpoint记录消费者：区分逻辑检查与真实计算。

完整20次安全写入且缓存条件一直满足时，仍有40份逻辑pre/post记录，但只有21个真实监测批次，19份复用；每个监测批次内部的micro forward另计。日志保留 `actual_monitor_forward_count`、`reused_monitor_forward_count`、`forward_source`。条件改变就重新计算，实际数量以日志为准。

## 2. 解除不同轨迹之间的批次等待

### 当前等待发生在哪里

现在每个交互轮收集所有active轨迹，等待整批生成返回，再统一检索，再进入下一轮。等待既跨同题的5条响应，也跨不同问题。vLLM内部虽能批量解码，外层交互仍有整轮屏障。

新顺序是每个 `(question_id, response_index)` 各自运行：

`生成查询 → 等自己的检索 → 继续生成 → 最终回答`。

某条轨迹完成检索后即可继续，其他轨迹可以仍在上一轮生成。单轨迹内部保留因果依赖；收齐一轮512×5轨迹后再归组、算优势并更新actor。

### 先做整轮权重驻留，再接独立轨迹调度

**第一步：增加完整rollout phase。**

- 在 `verl/workers/fsdp_workers.py` 和 `verl/workers/sharding_manager/fsdp_vllm.py` 定义整轮begin/end边界。
- begin时统一加载/同步当前actor权重并绑定 `policy_snapshot_id`。
- 全部生成/检索交互期间模型版本冻结、采样权重驻留。
- 所有轨迹结束后end，再卸载并进入actor更新；RNG按现有采样协议管理。
- phase内部的单请求不再触发FSDP state_dict、权重同步或卸载。

**第二步：增加轻量独立轨迹状态机。**

- 将 `reproduction/common8/generation_behavior.py` 的全批ordinal循环改成逐轨迹状态机，或提取状态机供新的 `search_g1/async_rollout_runtime.py` 调用（该文件是建议新增，尚未实现）。
- 请求进入持续接收任务的vLLM推理调度器。工具返回后，该轨迹立即提交下一次生成，空出的生成槽位可以处理其他轨迹。
- 沿用已有Ray/vLLM/HTTP能力。工具HTTP使用异步请求或受限线程池，明确超时、重试及并发上限。
- 在 `verl/trainer/ppo/ray_trainer.py` 收集完成结果并按身份归组，结束采样phase后才调用actor更新。

不能直接对当前 `generate_padded()` 包一层 `asyncio.gather`：它的下游每次调用都会进入FSDP-vLLM sharding manager并执行同步/collective；不同轨迹的请求顺序可能不同。异步请求层必须位于统一采样phase内部，避开逐请求collective。

### 保留当前采样与训练语义

- 每条轨迹携带 `question_id`、`response_index=0..4`、`trajectory_id`及固定的 `policy_snapshot_id`；返回顺序不会改变题组归属。
- 继续保存每次实际输入token、原始模型输出token、selected log-prob、观察mask、停止原因及交互序号。
- 原来按批次行号绑定的采样记录改为按 `request_id` 绑定。尤其 `reproduction/common8/sampling_capture.py` 的采样调用记录不能直接依赖异步完成顺序；每个请求登记自己的真实caller、输入和身份。
- 沿用4096总预算、回答空间预留和观察截断。模型输出不重新编码，PPO概率使用实际生成上下文。
- 512题每题5条完整响应，随后正常计算严格EM、GRPO优势、qprobe标签与head更新；不增加候选题的agentic rollout。
- 基础设施失败通过受限重试/恢复处理，不直接变成EM=0或丢掉一条响应。正常预算耗尽后得到的真实输出仍按现有严格规则评分。
- 保留当前mini/micro、optimizer预算和长短轨迹loss归约；本次调度优化不顺带改变它们。

并发暴露为超参数：`agent_workers`、`max_inflight_trajectories`、`generation_max_num_seqs`、`generation_max_batched_tokens`、`retrieval_concurrency`、`tool_timeout`、`tool_retries`。按GPU和检索服务配置，不硬复制另一台机器的workers数。

### 已有可复用源码

远程 `codex/cosearch-hpc-migration-20261008`，SHA `4fc8bd6b17b4412f4cb454c4e457caffa4e500bd`，嵌套作者源码：

- `reproduction/code-backup/cosearch/author_source/verl/verl/experimental/agent_loop/agent_loop.py:398-407`：每输入row创建覆盖完整轨迹的task。
- 同目录 `co_search_agent_loop.py:156-191`：独立状态机，生成和工具返回交错执行。
- `agent_loop.py:2106-2135`：完整采样阶段wake/sync，全部结束后sleep。
- `verl/tools/co_search_tool.py:317-362`：受限并发工具请求。

借用调度、服务接口和驻留方式，保持G1 reward、选题head和当前训练损失。迁入的是“轨迹交错、外层更新同步”，全轮仍使用一个actor版本。

## 3. KL系数为0时跳过固定reference计算

省掉的是整批reference模型前向及不参与损失的KL计算，不是一个便宜的系数标量运算。

当前实装条件：common8、`use_kl_loss=true`、`kl_loss_coef=0`。

- `ray_trainer.py` 不调用 `compute_ref_log_prob(batch)`。
- `dp_actor.py` 不要求batch有 `ref_log_prob`，不调用无用的 `kl_penalty`，零系数的损失项直接为0。
- 保留固定reference的配置身份，不制造全0的伪reference概率。
- 推广到其他配置时，同时检查reward KL shaping、非零KL loss及显式reference诊断；其中任何一项需要reference，就保留相应计算。

必须保留PPO的old-policy log-prob、概率比、old-policy安全KL及负优势尾部刹车。固定reference正则系数为0，不等于停掉这些防护。G1奖励中的必要probe推理也不能因名称类似reference而删除。

现有轻量日志：`infra/reference_forward_skipped`。上一版A800第一轮reference前向用5.75分钟；这是已观测的删除对象，不是新分支端到端测速结果。

## 4. entropy系数为0时完全跳过词表entropy

`dp_actor.py` 根据 `entropy_coeff != 0` 决定是否请求entropy；零系数时调用 `_forward_micro_batch(compute_entropy=False)`，跳过entropy softmax、求和及计算图。old-prob和监测路径也不请求entropy。

只使用 `detach` 或 `no_grad` 不够：它们可以省反向图，但前向词表运算依然发生。

正常选中token的log-prob仍需计算，包含 `z[label] - logsumexp(z)` 的完整词表归一化。这里去掉的是额外的entropy计算，不近似softmax，也不改变PPO损失。

现有日志 `actor/entropy_diagnostic_computed=0` 明确表示未计算，不能把它解释为模型真实entropy等于0。

## 实施顺序与轻量观察

1. 直接沿用已发布分支的重复监测、零KL reference和零entropy修复。
2. 迁入完整采样phase，整轮一次权重同步。
3. 接逐轨迹状态机和受限并发检索，保留原始采样记录及五条响应归组。
4. 使用正常训练日志观察采样/工具/更新耗时、权重同步次数、实算/复用监测批次及完成轨迹数；不增加额外512评测、GPU验收或压测门槛。

最终执行链路：

`选题512 → 固定actor并同步一次 → 2560独立轨迹交错生成/检索 → 按题收齐5响应 → G1 reward/优势 → 正常actor更新（post检查复用为下一pre） → qprobe head更新`。

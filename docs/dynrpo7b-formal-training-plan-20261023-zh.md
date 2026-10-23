# DynRPO 7B 正式训练实现计划

日期：2026年10月23日。范围：制定并固定实现方案；本轮不修改生产训练代码，不提交训练作业。

目标是在独立运行目录组合 **Soft Reward + GRPO + Update Probe + Progressive Prefix + Post Selection + Backstop Funding**。学生从官方 Qwen2.5-7B-Instruct 初始化，选题探针继承本线程已验收的 L15、Prefix80、整组方差 MSE 模型。保留原200个完整outer、固定数据划分和评测安排。用户已确认：outer1～100用Prefix80，完成100后切换160；每10轮复用该更新轮2560题×5条旧actor轨迹拟合3epoch；采用4张训练卡加2张独立检索卡。

## 1 实现来源与独立部署

| 部分 | 固定来源 | 复用范围 |
|---|---|---|
| 四项算法措施 | 聊天“审查七种算法实现”，`01a101e0-90d3-7731-bae9-58c8f1fdf7e2` | 已向该聊天咨询并取得实现说明；四项措施分属不同历史分支，并非已有完整7B成品 |
| Progressive Prefix与Post Selection | `work/prefix-schedule-cp50-20261018`，`3c08089fea344cd9084ceb26676644218a45f14c` | `progressive_prefix.py`、`two_stage_selection.py`、`periodic_data.py`、`actor.py`、`pending_probe_resume.py` |
| Backstop Funding | `/Users/shaoshitong/Documents/Search-G1/actor50-grpo-mixed30-20261018`，实际运行版本`2d9712395f6c666cd0b53f843f8bc1bfb6e5d11d` | `actor50_grpo/{schedule,mixed_floor,rerank}.py`；后续审计版本`9ebb1ea4cd4377bdb85c404cd44cba3e8a3eaa67`仅作修复参考 |
| 7B基础设施 | 聊天“U-验证 CoSearch 在 Search Agentic RL 的效果 (3)”，`01a163ab-97c3-7831-9d1d-41be25280063` | 实际运行源码`dde1452d2ec65fb3f4a0464d4beacf3b8136414c`，本地`control-6/work/easy7b-fa3-fastpath-20261022`；后来的归档提交不冒充运行版本 |
| 本线程探针 | `7a8ff6ae8a33324846d85f77ffc28e9ab699c028` | `reproduction/operations/qwen7b_warmup/online_packed_full169028_MSE_P80.py`、`packed_variance_probe.py`；验收归档`d466f60f65d1e2fceff61ae28cd0bd5302551c30` |

新模块放在`reproduction/dynrpo7b/`，使用独立配置、启动入口、Slurm名称、锁、端口、缓存和输出目录。只读复用已经校验的官方权重、检索资产及已编译FA3；不修改另一个聊天的运行源码、环境或服务。正式实现完成后提交、推送，并记录部署文件SHA与实际运行commit。

## 2 普通outer的完整数据流

1. 冻结本轮actor快照、当前选题探针版本和前缀阶段。从训练池取5120道不同问题，每题生成5条真实policy前缀。采样与模型输入均不读取EM、答案或组别。
2. 将一个question与其5条前缀打包，用当前探针输出组方差分数；按分数取1024题。保存原问题顺序、轨迹编号、原始token、采样log-prob及actor版本。
3. 从原来的前缀继续生成同一组G5，完成1024题的完整交互。不得重新生成一组回答来冒充原前缀的续写。检索返回文本不计policy token，也不计前缀预算。
4. 计算这1024题各5条回答的strict EM。优先选真实Mixed题，再按冻结的探针分数降序、原ordinal稳定排序，得到512题。不足512道Mixed时，其余全对与全错题统一按探针分数补齐。
5. 根据下面的Backstop规则，必要时继续原5120候选池中尚未续写的问题，替换已选非Mixed题。
6. 对最终512题、2560条真实轨迹计算独立Soft Reward，执行一次完整GRPO outer，再提交完整恢复状态。

`Mixed = 1[0 < sum(EM_j) < 5]`。Post Selection用真实EM参与训练选题是本方案的明确设计；这些标签绝不进入探针预测输入。报告分别列出候选池、探针Top1024、最终512的Mixed比例，避免将Post Selection的收益全部记成探针预测收益。

本分支普通轮候选预算固定5120，不混入历史动态候选上限20480或其他分支的768/1536续写预算。探针更新轮完成2560题G5供actor选512，并保留全部2560组供本节点拟合；不再为探针额外生成一套新actor轨迹。

## 3 Backstop Funding的实际含义

Funding指额外续写算力，不是奖励加分。最终512题中Mixed数量不超过153时触发，目标为154题，即至少30.078125%。先使用已完成但尚未入选的Mixed；仍不足时，按本轮冻结的探针分数，从原候选池剩余问题每批继续128题G5，直到达到目标或候选池耗尽。

保留初选Mixed。替换时先移除全错，再移除全对，各类内部移除探针分数最低者，以原ordinal打破同分。追加期间actor与探针均不更新，不重打分、不引入新问题、不重生成前缀。候选池耗尽仍不足时如实记录shortfall，不能声称数学保底。

每批记录候选预留、完成G5、strict EM、替换前后列表和提交游标；恢复从已提交批次继续，不能重复消费问题或重新抽取结果。最终actor更新量保持512题×5条，不随追加续写量增加。

## 4 选题探针与当前warmup严格对齐

默认主模型采用预先选定的 **seed1 E9 raw**，不根据最终确认池挑seed或改用EMA。

```text
/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/
algorithm-improvement-20261021/
full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3/
checkpoint-epoch-09.pt
SHA256 5f90fb17bee993353f47a7d687bdf2ea235dcd3d2eba287465567f4c286b13d7
```

以上换行仅用于展示，实际为一个连续路径。

固定官方7B的索引0～14产生表征，训练继承的索引15 Qwen block及完整组聚合头；L15在这里指零起始索引15。**变化中的actor产生前缀，表征编码器仍为与warmup相同的官方冻结模型**，不能静默替换成变化中的actor hidden。一个question只出现一次，随后拼接五个分隔符及五条policy前缀，沿用padding、mask、位置编号、token type和轨迹顺序规则。

```text
mu = sum(EM_j) / 5
v = mu * (1 - mu)
s = sigmoid(group_logit)
loss = mean((s - 4*v)**2)
predicted_variance = s / 4
selection = descending(s)
```

目标值仅为0、0.64、0.96。保持整组方差MSE，不添加轨迹BCE、Mixed BCE、Pearson或历史排序window；当前胜出方案没有使用这些损失。`s`是归一化方差分数，不是Mixed发生概率。BA诊断沿用固定阈值`s >= 0.32`，TopK不使用分类阈值。

raw接受梯度，EMA以0.9995跟随；继承raw、EMA、Adam、RNG，保留warmup的11889次更新计数，在线更新另设计数。LR5e-6、Adam默认betas/eps、weight decay 0、grad clip 1保持不变。默认选题继续使用raw，以便对应E9 raw；EMA独立保存和报告，不在每个节点事后择优切换。

已有两个独立确认池都已查看，不能继续用于在线节点选模型。探针诊断与RL固定512题问答评测分开；在线更新使用训练池数据，另固定训练池内不参与该次拟合的诊断问题。

## 5 每10轮更新与前缀阶段

outer0直接加载已完成warmup；在完成10、20、30……190后更新探针。沿用原200轮结束约定：200保存与最终评测，不再做没有后续选题用途的探针拟合。两节点之间参数固定，新探针只从下一outer使用。

**已确认的更新顺序**：actor9为第10轮生成2560题×5条完整轨迹 → 从中选512题完成actor10更新 → 保存并验收完整actor10及旧探针 → 使用原2560组轨迹继承拟合3epoch → 原子发布新probe → outer11。所有更新节点按此顺序执行：在完成outer t后拟合时，复用的轨迹来源为actor(t−1)，不是actor t。特征仍由官方冻结编码器计算；保存`rollout_actor_outer=t-1`、`completed_actor_outer=t`、`probe_effective_from_outer=t+1`，以区分这三个计数。

用户已确认的配置：

| 项目 | 固定配置 |
|---|---|
| 在线拟合预算 | 每10轮复用本更新轮全部2560题×5条旧actor轨迹，拟合3epoch，不新增完整rollout |
| 选题前缀 | outer1～100为Prefix80；完成outer100后切换，outer101～200为Prefix160 |
| 拟合前缀 | 完成outer10～90后的节点用P80；完成outer100～190后的节点用P160；与下一轮实际选题长度一致 |
| GPU布局 | 4张H100训练＋2张独立检索，共6张；探针阶段复用训练资源，不暗增第三组GPU |

**第100轮边界**：第100轮本身仍按P80打分和选题，其完整G5由actor99生成。actor100保存后，从同一批2560组完整轨迹中提取每条前160个policy token，重新计算P160表征并拟合3epoch；不重新采样轨迹、不追加完整rollout，也不把P80缓存假装成P160。短于160的回答沿用padding及无效位置mask。发布P160探针后，outer101才使用P160选题。checkpoint分别记录“actor100完成/P160探针待拟合”和“P160探针已发布”，避免恢复时多训actor100或重复完成的拟合epoch。

拟合输入必须包括未经Mixed筛选的本次完整数据，不仅训练入选512题。保持自然问题分布、独立shuffle与轨迹随机排列，不按EM凑50/50，不重复困难题补类。若仅有G5，每epoch不得伪称重新从G10抽样；真实G5复用次数与生成量分别报告。

Progressive Prefix需要同时作用于候选打分、拟合、诊断和恢复。当前组头支持变长输入，P80切P160不因为输入变长就更换linear、Qwen block或重置Adam。按实际question与五条前缀的有效长度核对2048位置容量，不静默截断question；若需要扩展位置embedding，保留旧权重、EMA和Adam坐标及step。切换先保存边界checkpoint，再按新长度拟合/验收后供下一轮选题。

## 6 Soft Reward与GRPO的绑定

选题方差探针只负责选题，**不是Soft Reward的grounding评分器**。参考中的独立boundary/commit评分器及其版本、编码器、校准集需要分别迁入7B。旧3B评分头不能直接接7B hidden。

参考Soft Reward为：非法输出或无有效答案为-0.1；错误答案为0.05；正确且无搜索为`1 + 0.5*b`；正确且有搜索为`1 + 0.6*(1-b)*f - 0.02*min(1,max(0,n-1))`。`b`为boundary分数，`f`为flip分数，`n`为搜索次数。实现时同时固定环境开关和奖励版本，不混入基础设施参考实验的Easy Selection专用奖励或额外grounding训练目标。

原协议为300题校准，round0真实拟合及50/100/150刷新。已经定位到既有7B round0候选，优先核对并复用，不能默认重跑：

```text
/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021/artifacts/round_0-restored-cb419d0b950846208121053d3d382e62/version-manifest.json
manifest SHA256 893fe8c21df2c30044d63a03884bef6ef98443f0010d38219e7dc96ae0b17286
boundary head SHA256 a2c33da3d37f779a6792b36aea80b4e095c2e6cfaed2900a75042de2e0af7b75
commit head SHA256 b4e402c811e5d09a1df15716666528e008e9bc576ccb1a9fc0ccb4ef85a4a143
```

来源为基础设施checkout中的`reproduction/operations/easy7b_20261021/evidence/actual-cp5-dependency-closure-and-future-wait50-20261021.json`，属于既有本地验收证据，本轮未远端fresh读取。部署时读回manifest，比对官方7B encoder完整文件映射、同300题、层21/24/27、Xa/flip及correction-feedback协议；一致则仅导入静态资产，为新run创建独立adoption，不继承另一实验的动态状态。不一致才补做round0 fit。

50/100/150先完成actor并保存完整joint，再用对应actor快照刷新双head，四个实际服务adopt同版本后才继续下一outer。拟合失败恢复待fit/adopt阶段，不能重做已完成的actor；已fit则复用，不重复rollout。200不刷新。未完成奖励绑定不能把系统标为Soft Reward已就绪。

GRPO保留原G5组内归一化。全对组只在均值/标准差统计中加入一个虚拟0，实际训练仍为5条真实轨迹；其他组使用原G5。std correction 1、epsilon 1e-6。保留token ratio、clip0.2、dual clip3、grad clip1、entropy/reference KL系数0、LR1e-6、57 outer warmup后恒定、每outer一次scheduler、global minibatch128与1个PPO epoch。

每outer20个更新机会，记录实际Adam写入和被数值/KL门控跳过的机会，不能将20机会写成20次真实更新。原虚拟8卡×micro2的masked-token均值权重需要与4张物理卡映射对齐；不能因为更改batch组织而静默改成全局token加权目标。保留原始sampling log-prob、policy mask和检索mask；额外续写不破坏其来源。

## 7 基础设施移植边界

复用实际运行的`fa3-lean-batched-search-stop-v1`组件，包括actor前反向FA3、vLLM prefill/decode FA3、Q28/KV4紧凑GQA、每个actor快照一次权重同步、原始token续写、批量检索和已验证的搜索终止边界。使用现有已编译FA3及依赖SHA，不重编译、不升级共享环境。

两种“词表投影”必须分清：已经跑通的是仅对response预测行做FP32全词表归约；`selected_qwen_head.py`在LM head之前仅投影policy行是更进一步的可选路径，参考正式7B作业中`selected_policy_logits=false`。计划先以已跑通路径接入；更进一步路径需在目标7B CUDA/FSDP上对齐log-prob和梯度后才启用。两者均保留完整词表归一化，不能截词表。

tokenizer加速使用固定Qwen2TokenizerFast的底层`backend_tokenizer.decode`绕过HF包装；vLLM仍接收`prompt_token_ids`，`detokenize=false`，不是彻底移除tokenizer。不得对已生成回答decode后再encode来构造训练token。

具体移植点：`flash3_attention.py`、`vllm_flash3.py`、独立FA3 runtime、`dp_actor.py`、`fsdp_workers.py`、FSDP/vLLM sharding manager、vLLM rollout、`generation_behavior.py`、async rollout与retrieval、rolling checkpoint/restore。Easy专用“没有question probe”的约束改为本方案明确阶段契约，而非直接删除检查。

按固定官方7B revision`a09a35458c702b33eeacc393d103063234e8bc28`、28层、hidden3584、Q28/KV4、vocab152064绑定。复用参考Python3.10/vLLM0.6.3及实际兼容环境。所有权重、checkpoint、轨迹、tensor、环境和cache均在远端共享存储，本地只留源码、配置和小摘要。

## 8 恢复状态与评测

每个完整outer保存学生各rank模型/Adam、scheduler、RNG、数据顺序/游标，以及选题probe的raw/EMA/Adam/RNG、prefix阶段、冻结encoder身份、奖励评分器版本和当前执行阶段。探针拟合按epoch保存，区分“actor已完成、probe待拟合”与“probe已发布”。

候选/续写/Backstop中间状态包括原问题及轨迹编号、actor/probe版本、原始token与采样log-prob、候选分数、EM、完成标记和替换账本。仅已完整写入并提交的批次可恢复。actor中途已有Adam写入而缺对应完整状态时不能假称无损恢复，也不能自动重跑旧outer掩盖回滚。

保持train169028、dev512、final51713及既有manifest；固定512题评测和长期保留checkpoint节点5/50/100/150/200，200结束后完整51713题评测。每outer的滚动恢复文件与这些长期保留节点分别管理，不先删除最近完整恢复点。

每轮报告候选数、完成G5数、probe Top1024和最终512的全对/全错/Mixed比例、Backstop追加量、actor与probe实际更新计数、分阶段时间及累计GPU时间。主要probe指标为Mixed BA、Mixed@K和实际方差误差；不以单轨迹准确率替代目标。

## 9 编码顺序与必要验证

| 顺序 | 新模块或接线 | 完成条件 |
|---|---|---|
| 1 | `recipe.py`、`source_manifest.json`、独立部署入口 | 源码/资产/数据/环境版本可追溯；使用已确认的前缀、复用预算及4＋2卡布局 |
| 2 | `group_probe_adapter.py`、`prefix_schedule.py` | E9 raw重放分数与原实现一致；raw/EMA/Adam继承正确；变长mask/位置一致 |
| 3 | `selection.py`、`mixed_backstop.py` | 同一G5贯穿prefix与续写；选题顺序、153/154边界、池耗尽及中断恢复正确 |
| 4 | `infra_profile.py`、rollout/actor适配 | 接入成熟FA3、token与检索路径；新增投影优化独立开关，数值一致后启用 |
| 5 | `reward_adapter.py`、GRPO入口 | 7B奖励校准/版本绑定完成；不混淆两个probe；损失、mask和归约权重对齐 |
| 6 | `probe_update.py`、`checkpoint.py`、`train.py` | 10轮边界顺序正确；完整状态、阶段和追加预算均可续跑 |
| 7 | 提交、推送、部署文件SHA核验 | 远端实际入口与已推送commit一致，正式运行开关仍关闭 |

只做与新增逻辑有关的验证：固定小样本的probe输入/分数重放；Post Selection与Backstop边界/恢复；目标7B的FA3和log-prob/梯度对齐；一次真实CUDA Adam恢复下一步一致性；完整actor→probe→下一outer的阶段切换。当前warmup只完成CPU状态与GPU推理验收，CUDA下一步恢复验证需要补上。

随后获得执行指令时，先跟进至少一个完整正常outer，连续确认五个完整outer及下一轮推进，并跟进首个10→11探针更新边界。检查实际更新、checkpoint和错误状态，不能仅凭Slurm RUNNING确认成功。这里的验证安排不代表本轮已启动任何作业。

## 10 当前交付范围

本轮交付实现计划和来源定位。正式训练代码尚未组合；三个配置选择已全部确认，并固定于本文及同目录`dynrpo7b-formal-training-decisions-20261023.json`。下一步按第9节编码顺序实现，不再因这三项配置停下询问，也不反复展开全仓review。本轮没有提交训练作业。

相关结果：[本线程warmup方案与独立验证结果](../reproduction/reports/qwen7b-algorithm-improvement-20261021/fullP80-selected-root-control/QWEN7B-WARMUP-SCHEME-AND-RESULTS-20261023.md)。

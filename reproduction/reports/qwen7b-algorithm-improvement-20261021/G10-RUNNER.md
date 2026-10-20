# 独立真实G10→G5探针runner（仅代码，尚未GPU运行）

入口：`reproduction.operations.qwen7b_warmup.probe_g10_train`。
新增文件：`probe_g10_train.py`、`test_probe_g10_train.py`。现有runner未改，未远端执行或提交作业。

## 输入和恢复

- 必须输入 `extra_g5_merge.merge()` 真正发布的 `bank/complete.json` 的绝对路径、文件SHA及字节。状态必须为 `PILOT_ACTUAL_ORIGINAL5_PLUS_EXTRA5_G10_SHA_OWN_EM_VERIFIED`。重新核对groups、original bank、各extra shard acceptance及own EM/ID拼接；不能拿planned G10代替真实响应。
- 原始5和新增5的Native L26特征来源保持独立，通过原 `packed_features.FeatureReader` 合并读取，Prefix20/QMAX273/D3584。核对index/prototype文件SHA、每次实际重构的特征字节SHA和轨迹身份；不在启动时再次串读全量tensor文件。不会构造不存在的特征或EM。
- 从固定E1 checkpoint SHA `02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724` 加载全部raw/EMA、block、flatten readout、Adam10464、native buffers、source四rank RNG archive，canonical rank0作为新单GPU分支RNG。LR5e-6/weight_decay0不变，不重置任何参数或Adam。
- 本入口仅支持现有flatten模型、二元Mixed或三事件损失。训练保留原题G10自有EM，随机/逐题均衡/全局均衡G5视图复用 `g10_question_loss`，同题全部视图总权重严格为1。homogeneous G10不能伪造Mixed；逐题均衡可显式 `on_unbalance=report` 保留真实不可均衡情况并报告，或 `raise`。
- 独立单H100探针分支，源四rank状态保留，但**不宣称四rank等价或无拓扑变化**。不更新actor，不提交采样/训练作业。

## 显存策略和预算

原 `g10_question_update.question_weighted_update` 保留Q×10全部前向图；这里仅在CPU测试中作为参考。

新实现第一遍逐micro做正常grad-enabled训练前向，立即detach logits并释放该micro图，保存每micro的RNG/native buffer状态。完整题权重损失仅对小型 `[Q,10]` logits求 `dL/dz`。第二遍同micro、同输入、同RNG/buffers重算，逐元素核对logits完全相同，再以该导数执行VJP累积参数梯度；最后Adam和EMA各更新一次。native-buffer/RNG时钟恢复到一次逻辑前向后的状态。

- 完整Q51：真实轨迹510；两遍实际前向轨迹曝光1020，反向510。若每题4个G5视图，逻辑view曝光51×4×5＝1020；这不是1020条不同真实轨迹。
- 相比旧G5 B51的255条真实轨迹，本次用510条真实轨迹；再增加一遍前向是显式计算成本，不能假装预算完全相同。
- 只保留一个micro transformer图。默认建议 `micro_trajectories=5`，视图/跨view pair另有显式上限，native重放CPU快照累计硬限制256MiB。
- 浮点梯度累积顺序可能不同于单体反向；数学链式法则未改，**不承诺与单体GPU逐位一致**。但本分支从同一来源执行两次完整真实更新，必须raw/EMA/Adam/native buffers/RNG hash完全相同才接受第一步。
- 可选原题batch小于51，但必须 `smaller_question_budget_acknowledged=true`，报告实际原题数，不能冒充matched B51。

## config.json必需结构

所有 `REF` 都为实际远端文件 `{path:绝对路径, sha256:实测SHA, bytes:实测字节}`；不能使用占位路径运行。

```text
source_checkpoint/source_config/source_bank/model_config: REF
model_dir: 已验收官方7B目录
source_commit: 父任务发布commit
code_hashes: {仓库相对代码路径: 文件SHA}（runner会检查所有必需文件）
g10_complete: 已验收真实merge bank/complete.json的REF
plan: REF，内容见下
update_budget: 明确优化器更新次数
global_questions: 51（或1..50并显式承认预算变化）
micro_trajectories: 5
save_every: 正整数
max_virtual_views: 2048
max_inter_view_pairs: 1000000
grouping: {mode: random或per-question-balanced或global-balanced,
           views_per_question: 4, seed: 1729, on_unbalance: report或raise}
recipe: {id: 固定实验名,
         loss_kwargs: {trajectory_weight: 1, three_event_weight: 1}}
evaluations: [{name: validation2048, role: validation,
               bank: 真实带own EM的G5或G10 bank REF,
               split: 该bank中的明确split键,
               features: Native feature manifest列表JSON的REF,
               fixed_g5: 预先冻结label-blind G5选择计划的REF}]
```

`plan` 必须包含 `cohort_qids`、恰好update_budget个 `batches`，以及相同数量非负整数 `batch_epochs`。每batch包含恰好global_questions个不同真实G10 qid；明确允许题在不同epoch再次出现，但不能把重复曝光当新增独立题。随机/均衡对照必须使用同一原题batch plan、epoch、源checkpoint与预算，仅改变声明的grouping/loss。均衡mode的两类“数量平衡”和“加权类别mass平衡”分别记录。

验证的 `fixed_g5` 支持 `fixed_g5()` 直接结果，或 `validation_g10_materialize` 的 `planned-g10-and-fixed-g5.json` 中 `fixed_G5`。必须 `role=validation`、`labels_consulted=false`，文件SHA预先绑定。先确认固定五条ID再join真实EM；绝不调用训练的label-aware均衡函数选验证G5。final池的role会被拒绝，不能在调参runner中评估。所有验证题的qid和question_hash与训练真实G10都必须不交叉；若配置原始固定512诊断，保留其原始固定5 IDs。

## 运行与验收边界

发布并部署完整import闭包后，由父任务在新目录准备config/plan，再显式调度运行：

```bash
export QWEN7B_CODE_ROOT=/actual/immutable/repo
export FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" \
  -u -m reproduction.operations.qwen7b_warmup.probe_g10_train \
  --base /data/user/jxu957/experiments/actual-isolated-g10-variant
```

入口拥有独立lock/controller、first-update replay、update-intent、progress、curve、immutable full checkpoints、failure/complete回执。真正E1首更新双重放后先保存完整10465状态，再进入下一更新；预算结束先保存完整state，再外部验证。未解决的后续update-intent/progress高于最新checkpoint时拒绝静默回滚。收信号后在当前完整更新边界保存。

CPU toy已通过13项测试（本runner7项＋现有G10 loss6项），包括三种grouping的VJP/完整图梯度和Adam/EMA一致、10464→10465、含dropout/native state的真实首更新双重放、fullstate恢复后10466下一步一致、第二遍输入改变时在Adam前拒绝、真merge引用与own EM拼接检查、label-blind验证身份和final-role拒绝。toy使用合成小张量，仅验证代码；**没有生成实际G10、没有真实GPU首更新或显存验收，不代表已训练成功**。实际FA重算logits若不逐位一致会在Adam前拒绝，须保留故障回执进行真实修复。

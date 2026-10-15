# 原生 Qwen 最后一层二分类探针：执行说明（2026-10-15）

文档日期：2026-10-15。**状态截至 2026-10-15 22:35（Asia/Shanghai）**。目标为 **10 个 warmup epoch，随后完成 200 个正式 actor outer**，每个正式 outer 更新探针 2 个 epoch。本文件记录该时间点的实际版本、配置和执行状态；后续进展以同一 run/execution 的实际 receipt 为准。

native warmup、完整 token bank、初始 Head 绑定和 initial0 采用均已 accepted。预启动 claim 文件名碰撞的独立修复入口正在准备，正式 actor **尚未 launch，尚无已完成的正式训练 outer**。历史 handle 记录一次性操作，不能重复执行已经发出的 action。

## 固定源码与现场身份

| 项目 | 固定值 |
|---|---|
| 实际正式 runtime commit | `927163041cf0ddf2c2c517efc466177d76dc2a18` |
| runtime 分支 | `codex/qprobe-qwen-native-formal-cpu-fixture-repair-20261015` |
| runtime source map SHA256 | `7ce4d02996393270a47e1d3840ab5ee793205e17e3154de18193c19b4120851e`，508 个运行文件 |
| OPS / observer / formatter commit | `03608b925c8c2948eff4ef5827064a04372ca504` |
| OPS claim 修复与续接入口 commit | `d95b13c9762823982c7238eaef5742f53b44d871` |
| 有效 native warmup commit | `e81445905bf8b15dbb0bdec3afa29d99ab0b11d8` |
| warmup source map SHA256 | `006912cc7817b2ae8abde32a85c42d21b2c1976bf0b5618066c6aa9d2f0674ad` |
| 正式 run / execution | `862bccb5fcf348ce8454f9a332f37bdd` / `027e65da351a4e018dd377b806c19434` |

当前正式工作区为 `/data/shaoshitong/Search-G1-a800-fresh200-92bb58c18ae74ca9bc59fe1ba5dc02b6`；有效 warmup 位于 `/data/shaoshitong/Search-G1-qprobe-qwen-lastblock-warmup10-096c8297cba74f8089063e763686671e/warmup`。

日期为 2026-10-15 的独立 release 包保存 runtime 源码、OPS 原始文件和执行说明，其来源清单为 `docs/releases/qprobe-qwen-native-block-2026-10-15-manifest.json`，实际绑定的科学参数和 artifact hash 为 `docs/releases/qprobe-qwen-native-block-2026-10-15-config.json`。表中的 OPS03608 是原操作入口的固定版本；新增修复入口的来源以 release 清单为准。**现有任务继续使用已绑定的 runtime927，不用 release checkout 替换活动 repo。** OPS 针对 A800 现有资产、认证路由和现场目录；复现前需具备这些现场依赖。

当前公开仓库的发布许可仅覆盖代码、配置、执行文档和经清理的 scalar/hash 记录。凭证、模型权重、Head/Adam checkpoint、token features、原始 QA/rollout、完整逐题评测及环境包均不进入 Git。

## 模型、数据和算法

官方模型为 `Qwen/Qwen2.5-3B-Instruct`，revision 为 `aa8e72537993ba99e69dfaafa59ed015b17504d1`。正式 actor 与 reference 从同一官方模型初始化，actor/Adam/scheduler/RNG/sampler 均 fresh。目标环境为 Torch2.4.0+cu121、Transformers4.46.3、Hydra1.3.2，环境包留远端。共同数据 manifest SHA256 为 `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`，训练全集为 169,028 个 canonical question。

特征由冻结官方 encoder 的 **layer27** 提取：保留每个问题的全部真实 token，每 token 2048 维，输入探针前 detach。全训练集扫描得到 global Bmax=244；不做 mean、CLS 或最后 token 池化，也不截断真实 token。注意力只省去整个 batch 末尾的纯 padding，随后 masked-zero、右补齐至 244，再 flatten 成 499,712 维。

探针直接实例化 Hugging Face 原生 `transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer`，独立复制官方 **layer35 的全部 12 个参数张量**，再连接 fresh `Linear(499712, 2)`。该分支的原生 block 与 readout 都可训练，不与 encoder/actor 共享可写参数。原生层保留 16 个 Q heads、2 个 KV heads 的 GQA、RoPE θ=1,000,000、RMSNorm ε=1e-6、残差和 SwiGLU；MLP intermediate=11,008，输出仍为 2048。没有自行重写 transformer block，也没有把 KV heads 改为 16。

令一个完整 G5 的严格正确数为 K，binary 标签为 `0 < K < 5`，目标样本标准差为 `sqrt(K*(5-K)/20)`。两个输出分别是 binary logit 和经 sigmoid 限定到 `[0, sqrt(0.3)]` 的标准差预测。每组唯一训练目标是：

```text
L_i = BCEWithLogits(logit_i, 1{0<K_i<5})
      + 0.3 * ((std_hat_i - sqrt(K_i*(5-K_i)/20)) / sqrt(0.3))^2
```

没有 CE3、CE6、ranking loss 或类别重加权。分类主指标为 binary accuracy；selected512 的拟合前统计、recent8 回放的拟合后统计和 warmup 固定5120统计分别报告。

Warmup 使用完整 5120 组已验证 teacher 标签，共复用 25,600 条历史 teacher trajectories，未重新生成 teacher rollout。fresh 官方 block、fresh readout、fresh Adam/private RNG 训练 10 epoch，batch256、micro32，LR=3e-4，Head seed=1729；micro 梯度按真实 batch mean 累积，每256组只进行一次 Adam，共 **200 次实际 Head Adam**。

正式选题从 5120 候选问题按 binary pMixed 取 Top512，E=0，平局按 question-ID hash；回归输出不参与二次排序。每个已完成正式 outer 的512组标签进入当前及前7个已完成正式 outer 的回放，teacher warmup 标签完全排除。年龄权重为：

```text
age:       0    1    2     3      4       5        6         7
w(age):    2    1   0.5   0.25   0.125   0.0625   0.03125   0.015625
formal objective = mean(w_i * L_i)
```

不按全局 weight mass 或 minibatch weight sum 重新归一化。每 outer 的探针更新为2 epoch，batch256、micro32。raw-age-weighted loss 会随回放样本和权重质量变化，不把跨 outer 的下降称为固定训练集拟合改善；未记录的未加权 loss 明确保持 unknown。

正式 actor 使用4张实际分配的训练 GPU、BF16、每 rank actor micro4（global forward micro16、累积8）；PPO mini128、每 outer20个 optimizer opportunities，G5总计2560条轨迹。old/ref log-prob global32、每 rank8。actor 全程使用 Search-G1 双探针奖励、group advantage 和原状态 mask。LR=1e-6，57个完整outer线性warmup后常数，首outer实际LR=0；entropy/reference KL系数均为0。保留 PPO clip0.2、DualClip3、sampling KL brake0.2、negative-tail0.01、exp前finite/bound30和FP64全局L2 clip1。相邻监测复用、零系数前向跳过、整轮同步权重及逐轨迹异步生成/检索均保留。20 opportunities 全部处理且至少一次成功返回Adam才计一次正式 outer / scheduler；actual Adam、skip、rollback 和 partial 单独记录。warmup 的200次 Head Adam 不计入正式200 outer。

## 已有实际结果和资产

| 项目 | 已有凭据与结果 |
|---|---|
| 有效 warmup | 10 epoch / 200实际HeadAdam / 11份epoch记录 / 200份minibatch记录；epoch10 accuracy=`0.998828125`（**99.8828%，5114/5120**） |
| warmup 统计口径 | 同一5120训练集，无独立验证集；不据此声称正式训练或泛化结果 |
| warmup ready SHA256 | `2f13faecb07a98a8fc87a0e9d52d1e93f1ae720d8043bef21193205807b2c466` |
| warmup Head SHA256 | `9b32f40ea573b855dc9e3be627fce7be1c03a1ba0b0e374808a81db311bb6849` |
| token feature spec SHA256 | `fcc9d61db156ddbb4e96dcd6c9437797b978dbbfc9774145a28e0c761de1a9e7` |
| 完整 bank | 169,028行；复用5120行、补编码163,908行；完整SHA与 encoder owned-resource release 已 admit |
| full-bank manifest SHA256 | `fa8c9513f89b3700851f8fe75665bf515a82e7f491bc3a52a643a6e82aa243e9` |
| converted formal initial Head | logical0 / 实际warmupHeadAdam200 / 空formal replay / teacher excluded；SHA256=`82e995c1aae9d01833b6229bc879c98d20bb84e107301a2ff2c1885b23e773d0` |
| converted initial ready SHA256 | `c5bdbdb02ca5477235fc8b857dbd56baea13f4c6f54f67d44ad7c3b191316fb7` |

Warmup 的两个 loss 部分并非都持续下降。标准差辅助贡献在 epoch1–10 几乎保持 `0.0884375`；下表只比较同一5120训练集的起点和终点。

| 同一训练集统计 | epoch0 | epoch10 |
|---|---:|---:|
| total loss | 1.280718 | 0.1250769 |
| BCE | 1.122461 | 0.0366394 |
| 0.3 × normalized std MSE | 0.158257 | 0.0884375 |

99.8828% 是固定 warmup 训练集 accuracy，不等于正式 rollout 的 mixed 比例，也不是独立验证集或泛化 accuracy。

完整 bank 位于 `/data/shaoshitong/Search-G1-qprobe-fulltoken-feature-bank-8be88f0d4ac1470f95fe13b4fe146a3f/token-features`。缓存编码进程已退出0，admit 已确认 full SHA、0 owned CUDA、0 owned sessions。converted Head 采用本次 native warmup 的 block/readout/Adam/private RNG。

本地状态依据为 `preparation/qprobe-qwen-native-formal-remaining-execution-handoff.json`、`preparation/qprobe-teacher-warmup-a800-progress-handoff.json` 中的 `current_authoritative`，以及下列实际 receipt：

- `preparation/qprobe-qwen-json-binding-repair-completion-observation.json`
- `preparation/qprobe-qwen-native-formal-cpu-repair-v2-bank-admitted-observation.json`
- `preparation/qprobe-qwen-native-formal-cpu-repair-v2-bound-observation.json`
- `preparation/qprobe-qwen-native-formal-cpu-repair-v2-bind-launch.json`

旧 c83d 在 Head 初始化前因 JSON binding 键类型失败、0 Adam，故障历史保留。有效096c warmup已完成；两个命名空间都不能通过重复 action 重新执行。

## 操作入口与执行顺序

从对应 checkout 的项目根目录使用以下相对入口：

- `reproduction/operations/a800_qprobe_qwen_native_formal.py`：现场一次性 stage/prepare/bank/bind/launch 与 metadata observe。
- `reproduction/operations/a800_qprobe_qwen_prelaunch_claim_repair.py`：当前预启动 claim 碰撞的独立一次性修复入口。
- `reproduction/a800_common/qprobe_qwen_formal_ops.py`：远端 source-bound helper，由前述 OPS 调用。
- `reproduction/common8/qprobe_qwen_bind.py`：CPU核验并采用本次 native warmup，保存并回读 formal initial Head。
- `reproduction/operations/a800_qprobe_qwen_native_observer.py`：正式训练的单次有界 primary metadata read。
- `reproduction/operations/a800_qprobe_qwen_hourly_format.py`：纯本地 hourly scalar/hash formatter。

参数含义：`HANDLE` 是当前本地 prospective deployment JSON 的绝对路径；当前文件名为 `preparation/qprobe-qwen-native-formal-cpu-repair-v2-deployment.json`。`RECEIPT` 必须每次使用新的本地小型JSON路径；`RUNTIME_REPO` 是干净且精确位于 source927 的 runtime checkout；`RELEASE_REVIEW` 是当前实际部署的有效 release JSON；`DELIVERED_STEP` 是这个 run/execution 上真实已交付报告的最后完整 outer，不能用最新观察值替代。

新实验的依赖顺序为 `create → stage → prepare → warmup-admit → bank-prepare → bank-encode → bank-admit → bind → adopt-initial0 → prepare-launch → launch`。`create` 使用新 handle、已发布源码/清单/代码 bundle 和经验证的 warmup handle。**本次 bank、Head 绑定和 initial0 采用均已验收，不重跑这些步骤。**

本次 `prepare-launch` 失败于 OPS 外层 guard 与 helper 同时独占创建 `prepare-launch-once.json`。失败发生在 actor claim 之前，runtime927 仍 git clean。修复入口 `a800_qprobe_qwen_prelaunch_claim_repair.py` 沿用 source927 原 `prepare_launch` 的严格校验，以独立 claim 保留旧失败记录；它不重跑 cache/warmup，也不修改活动 runtime。后续新部署的 OPS guard 使用 `native-formal-operator-` 前缀，与 helper claim 分开。**截至本状态时间，修复仍在准备，尚无修复成功凭据。**

当前继续顺序为：观察同一 handle → 独立预启动修复成功 → 实际 idle admission → launch 一次。修复入口须使用已发布且与 release 清单一致的文件。

```bash
# 在 OPS checkout 根目录；观察不会重复 execute。
python3 reproduction/operations/a800_qprobe_qwen_native_formal.py observe \
  --handle "$HANDLE" --receipt "$RECEIPT"

# 修复入口发布就绪后执行一次；不重复原 prepare-launch action。
python3 reproduction/operations/a800_qprobe_qwen_prelaunch_claim_repair.py \
  --handle "$HANDLE" --receipt "$RECEIPT"

# 修复实际成功后读取 idle inventory；release 有效后只 launch 一次。
python3 reproduction/operations/a800_qprobe_qwen_native_formal.py observe \
  --handle "$HANDLE" --admission --receipt "$RECEIPT"
python3 reproduction/operations/a800_qprobe_qwen_native_formal.py launch \
  --handle "$HANDLE" --review "$RELEASE_REVIEW" --receipt "$RECEIPT"
```

每条命令的 `RECEIPT` 都要更换，不能复用同一路径。传输断开或返回 launch receipt 后，继续用相同 handle 的 `observe` 确认实际流程；不要再次发送原 action。一次提交成功、PID存活或首条日志都不等于训练有效推进。正式启动后至少核验两次实际前进，并确认至少一个完整正式 outer 正常返回。

训练 GPU 由启动时实际 idle/UUID placement 绑定，训练选四张且排除辅助5/6；辅助5/6也需满足其角色的空闲条件。`head_device=cuda:0` 指当前训练 CUDA namespace 的第一张已分配卡，不是静态指定物理0。不能抢占或停止别人的进程，也不用全局 Ray stop 清理。

正常监控每30分钟一次有界 metadata read，不做分钟级 SSH 轮询，不为监控加载 Head、tensor、QA/token 或重新评分。正式 primary observer 在 runtime已发布、实际启动/部署凭据齐全后使用：

```bash
python3 reproduction/operations/a800_qprobe_qwen_native_observer.py \
  --handle "$HANDLE" --runtime "$RUNTIME_REPO" \
  --expected-run 862bccb5fcf348ce8454f9a332f37bdd \
  --expected-execution 027e65da351a4e018dd377b806c19434 \
  --after-step "$DELIVERED_STEP" --receipt "$PRIMARY_RECEIPT"

python3 reproduction/operations/a800_qprobe_qwen_hourly_format.py \
  --receipt "$PRIMARY_RECEIPT" --delivered-anchor "$DELIVERED_ANCHOR" \
  --output "$NEW_HOURLY_SCALARS"
```

`PRIMARY_RECEIPT`、`NEW_HOURLY_SCALARS` 是新的小型本地JSON路径；`DELIVERED_ANCHOR` 是同run/execution/source/base的既有实际交付记录，至少含 `delivered=true`、`completed_outer`、`delivered_unix`。先格式化和交付报告，再更新交付anchor；formatter不会改写anchor。报告区分 processed、完整outer返回的Adam、retained partial Adam、skip与unattempted，区分拟合前selected512指标和拟合后recent8指标，保留已有EM及GPU采样gap。未计算的完整步骤GPU峰值和缺失指标写 unknown，不填0或混入旧run。

## 保存、资源与最终完成条件

checkpoint、Head/Adam、packed token bank、原始 rollout 和完整逐题输出全部保存在显式远端目录；Mac只保留代码、配置、文档、小型scalar/hash与必要审计摘要。已发布代码bundle仅作源码中转；确认目标校验后按本地存储规则处理，未完成目标保留必要副本。

正式checkpoint节点为 **5、50、100、150、200**。每个保存点检查 actor/head/optimizer/scheduler/private RNG 与 replay 的配对身份和实际保存凭据；metadata observer只转述已有证明，不自行做tensor重载。恢复必须使用已验证的完整配对恢复链，不能把旧Head或孤立actor checkpoint冒充新初始化。

完整目标需要：实际完成1..200的正式outer/恢复链、所需checkpoint与有限性/SHA凭据、按原顺序覆盖最终51,713题的binary strict EM/F1与七来源统计（NQ固定3610题），以及最后的only-owned资源清理验收。大规模逐题结果留远端，公开只发布允许的scalar/hash。清理按本任务记录的 PID birth/session/UUID 所有权进行，保留源码、日志、checkpoint与必要复现资产；不能因为节点到期或文档发布提前宣告目标完成。

## 当前执行状态

截至 **2026-10-15 22:35（Asia/Shanghai）**：warmup10/200 HeadAdam及资源释放、full169028bank完整SHA及资源释放、bind和initial0采用均已 accepted。initial0 的 `adopted:true` proof SHA256 为 `811af902563e2e438f02b5ed3ae72d09eca3c813ca19559ed869ad0047c1b920`；其200次warmup Head Adam不计入正式训练进度。

独立预启动 claim 修复源码已提交并推送，旧失败 claim 保留。该修复的实际执行、正式 launch 和正式 outer 进展仍按后续真实凭据确认。本文件不把已完成的前置阶段视作正式训练已启动。

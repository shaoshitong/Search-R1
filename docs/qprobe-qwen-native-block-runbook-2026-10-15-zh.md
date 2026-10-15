# 原生 Qwen 最后一层二分类探针：执行说明（2026-10-15）

文档日期：2026-10-15。**最新状态截至 2026-10-15 23:39:13（Asia/Shanghai）**。最新人工目标为 **10 个 warmup epoch，随后完成 200 个正式 actor outer**，每个完整正式 outer 更新探针 2 个 epoch；这覆盖旧目标中20 epoch的描述。当前隔离 runtime 为 `a47abb701f62a6c4eb9e8e8c987c67a4cf804098`，已于23:28:10完成唯一一次实际 launch。23:39:13的原生 primary 只读凭据确认 pipeline `310262/birth615768175` 与 trainer `317298/birth615811482` 均实测存活，阶段已进入 `GRPO_1_TO_200`，没有记录错误。

**当前正式完整 outer、optimizer机会、成功actor Adam、reward和选中题混合比例仍为 unknown，尚未证明首个完整正式 outer。** 训练进程启动和GPU计算不等于已完成更新，不能填为0或声称正式恢复验收已经完成。既有096c的10 epoch／200次Head Adam已验收采用；99.8828%仅为同一5120题的warmup训练集准确率，不是正式 rollout 表现。下一步只沿下文当前handle做有界只读进展观察，不重跑任何准备或启动动作。旧92bb与ff1196均为失败历史，禁止重复执行。

旧92bb/source927 于 **22:47:32** 真实失败：`AssertionError: Cannot obtain actual CUDA device properties` 发生在 `main_task` 的 driver GPU 入口 hook，尚未进入原训练 body。已确认 **formal outer=0、正式 actor Adam=0、正式 Head 更新=0**；warmup 的200次 Head Adam另计。22:50:23 的观察确认 pipeline/trainer 均已退出，22:54:35 的终态审计接受34条 owned birth 的清理核验：0 owned SID process、0 owned CUDA context，项目 trainer live 列表为空。**禁止重新执行旧92bb handle 的 prepare/repair/launch action。**

修复采用出生即绑定准入 GPU UUID 的 fresh Python driver，同步接入原 Ray 集群执行原训练 body；C4审计保留所有PID birth检查并只接收有完整绑定的会话主进程。算法、科学 body、四卡 actor 资源池及 CPU RNG 移交保持不变。有效096c native warmup10/200和169,028行 token bank已经验收复用，没有重跑warmup、teacher rollout或bank编码。

## 当前现场身份

| 项目 | 当前固定值 |
|---|---|
| runtime checkout | `/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-driver-process-fix` |
| runtime commit / branch | `a47abb701f62a6c4eb9e8e8c987c67a4cf804098` / `codex/qprobe-qwen-native-driver-process-fix-20261015` |
| runtime source map SHA256 | `e15d3e34f487f154403846737e9e17a7fc9d1a6aba3785a746758e5c3b1b5916`，508个运行文件 |
| 当前 handle | `/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-driver-process-fix-v2-deployment.json` |
| 当前远端 base | `/data/shaoshitong/Search-G1-a800-fresh200-4fd508e35bc84658ba892a97248f8a7f` |
| run / execution | `524254af06a1472daa1368daba4e0b17` / `fd91bd58d4cf40a89b35427267f9d34e` |
| 当前 native primary | `/Users/shaoshitong/Documents/Search-G1/repo/reproduction/operations/a800_qprobe_qwen_native_observer.py` |
| primary SHA256 / published commit | `ed53322142808483fbc74fee758468219626b403e52f10c09c5c71d42eeb4351` / `098ae78a50945684973a41a2de47ab81825f15ba` |
| 最新实际只读凭据 | `preparation/qprobe-qwen-native-driver-process-fix-v2-native-startup2.json`，SHA256=`fd19e9866c5c730c2fac2f5cd73a807c63b1f002fb031bb6ac49256ac3452dec` |
| 当前实测进程 | pipeline `310262/birth615768175`；trainer `317298/birth615811482`，均存活且birth匹配 |

CPU26、warmup/fullbank采用、Head绑定、initial0、预启动、实际部署批准和唯一launch均已完成，不是当前待执行步骤。当前工作仅为核实正式训练真实推进，并由同一进程完成200个完整outer及既定验收；不重复warmup、teacher生成或fullbank编码。

## 历史 source927 与失败现场身份（禁止执行）

| 项目 | 固定值 |
|---|---|
| 旧92bb实际 runtime commit | `927163041cf0ddf2c2c517efc466177d76dc2a18` |
| runtime 分支 | `codex/qprobe-qwen-native-formal-cpu-fixture-repair-20261015` |
| runtime source map SHA256 | `7ce4d02996393270a47e1d3840ab5ee793205e17e3154de18193c19b4120851e`，508 个运行文件 |
| OPS / observer / formatter commit | `03608b925c8c2948eff4ef5827064a04372ca504` |
| OPS claim 修复与续接入口 commit | `d95b13c9762823982c7238eaef5742f53b44d871` |
| native observer typed-claim 修复 commit | `098ae78a50945684973a41a2de47ab81825f15ba` |
| 有效 native warmup commit | `e81445905bf8b15dbb0bdec3afa29d99ab0b11d8` |
| warmup source map SHA256 | `006912cc7817b2ae8abde32a85c42d21b2c1976bf0b5618066c6aa9d2f0674ad` |
| 旧92bb run / execution | `862bccb5fcf348ce8454f9a332f37bdd` / `027e65da351a4e018dd377b806c19434` |
| 新隔离修复 commit / source map | `a47abb701f62a6c4eb9e8e8c987c67a4cf804098` / `e15d3e34f487f154403846737e9e17a7fc9d1a6aba3785a746758e5c3b1b5916` |
| 新隔离实验 run / execution / handle | `524254af06a1472daa1368daba4e0b17` / `fd91bd58d4cf40a89b35427267f9d34e` / `preparation/qprobe-qwen-native-driver-process-fix-v2-deployment.json` |

旧92bb工作区 `/data/shaoshitong/Search-G1-a800-fresh200-92bb58c18ae74ca9bc59fe1ba5dc02b6` 保留失败源码、日志及已验资产，不作为活动训练现场。有效 warmup 位于 `/data/shaoshitong/Search-G1-qprobe-qwen-lastblock-warmup10-096c8297cba74f8089063e763686671e/warmup`。

日期为 2026-10-15 的独立 release 包保存 runtime 源码、OPS 原始文件和执行说明，其来源清单为 `docs/releases/qprobe-qwen-native-block-2026-10-15-manifest.json`，实际绑定的科学参数和 artifact hash 为 `docs/releases/qprobe-qwen-native-block-2026-10-15-config.json`。现有清单和配置精确记录新a47abb源码与已核验Head绑定；旧927/9713失败版本的清单和Git历史均保留。表中的 OPS03608 是原操作入口的固定版本；历史修复入口的来源以 release 清单为准。**保留旧 runtime927，不热改旧 repo；新实验使用另行发布并绑定的隔离源码。** OPS 针对 A800 现有资产、认证路由和现场目录；复现前需具备这些现场依赖。

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
| 当前4fd508 converted formal initial Head | logical0 / 实际warmupHeadAdam200 / 空formal replay / teacher excluded；SHA256=`a269238b0c6c2c182efc058612de895a33f65f7ce9d66966d1973c571fe0f3f1` |
| 当前4fd508 converted initial ready SHA256 | `5b707c21b91594effeef57de1b4b53e276398c4a7bfc837b131986aa7880e462` |

Warmup 的两个 loss 部分并非都持续下降。标准差辅助贡献在 epoch1–10 几乎保持 `0.0884375`；下表只比较同一5120训练集的起点和终点。

| 同一训练集统计 | epoch0 | epoch10 |
|---|---:|---:|
| total loss | 1.280718 | 0.1250769 |
| BCE | 1.122461 | 0.0366394 |
| 0.3 × normalized std MSE | 0.158257 | 0.0884375 |

99.8828% 是固定 warmup 训练集 accuracy，不等于正式 rollout 的 mixed 比例，也不是独立验证集或泛化 accuracy。

完整 bank 位于 `/data/shaoshitong/Search-G1-qprobe-fulltoken-feature-bank-8be88f0d4ac1470f95fe13b4fe146a3f/token-features`。缓存编码进程已退出0，admit 已确认 full SHA、0 owned CUDA、0 owned sessions。converted Head 采用本次 native warmup 的 block/readout/Adam/private RNG。

前置资产历史依据包括 `preparation/qprobe-qwen-native-formal-remaining-execution-handoff.json`、`preparation/qprobe-teacher-warmup-a800-progress-handoff.json` 及下列实际 receipt；旧92bb最新终态以新增的 terminal audit 为准：

- `preparation/qprobe-qwen-json-binding-repair-completion-observation.json`
- `preparation/qprobe-qwen-native-formal-cpu-repair-v2-bank-admitted-observation.json`
- `preparation/qprobe-qwen-native-formal-cpu-repair-v2-bound-observation.json`
- `preparation/qprobe-qwen-native-formal-cpu-repair-v2-bind-launch.json`
- `preparation/qprobe-qwen-native-formal-92bb-terminal-failure-owned-audit.json`，SHA256=`214270332335330c32851b8db31e5e4d14d70cd734633837a16bf4adc13c2191`

旧 c83d 在 Head 初始化前因 JSON binding 键类型失败、0 Adam，故障历史保留。有效096c warmup已完成；两个命名空间都不能通过重复 action 重新执行。

## 当前操作与监控

当前a47／4fd508已完成 `stage → prepare → warmup-admit → qualified-bank-admit → bind → adopt-initial0 → prepare-launch → launch`，且实际部署批准已闭合。**不得再执行这些动作，也不得重跑warmup、teacher rollout、bank-prepare或bank-encode。** OPS正式入口和远端helper保留为已执行历史与故障处理依据，不是当前待执行清单；旧92bb、ff1196的handle及旧claim修复入口禁止用于当前作业。

当前唯一primary为上表固定SHA的 `a800_qprobe_qwen_native_observer.py`。它每次只读现有JSON/scalar、四rank机会账本、PID birth、Head/checkpoint/评分metadata及已持久GPU采样；一次有界SSH上限65秒／64MiB读取／128KiB输出，不读取模型、tensor、QA/token，不调用native verifier，也不新增GPU测试、概率扫描或评测。正常每30分钟一次，不进入分钟级SSH轮询，不为了试查同时调用其他reader。

下面是**当前同一handle**的唯一只读命令；receipt使用新时间戳路径，不能覆盖已有凭据。在真正同run/execution/source/base的正式小时交付anchor建立前，`--after-step`保持fresh初始边界 **0**，不能混用warmup或旧run的anchor。正式anchor建立后，只换成该anchor实际已交付的完整outer值，不能用最新观察值替代。

```bash
QPROBE_RECEIPT="/Users/shaoshitong/Documents/Search-G1/progress-checks/native-qwen-primary-$(date -u +%Y%m%dT%H%M%SZ).json"
python3 /Users/shaoshitong/Documents/Search-G1/repo/reproduction/operations/a800_qprobe_qwen_native_observer.py \
  --handle /Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-driver-process-fix-v2-deployment.json \
  --runtime /Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-driver-process-fix \
  --expected-run 524254af06a1472daa1368daba4e0b17 \
  --expected-execution fd91bd58d4cf40a89b35427267f9d34e \
  --after-step 0 --receipt "$QPROBE_RECEIPT"
```

23:39:13凭据确认trainer存活及阶段进入`GRPO_1_TO_200`，但尚无闭合的正式attempt前缀；completeouter、actor Adam、skip和reward均保持unknown。还须沿同一进程核实至少两次真实训练推进及首个完整正常outer，首五轮按既有native gate验收后继续6..200，不为里程碑停训或另加200轮。SSH超时、缺失账本或unknown不等于退出，不能据此重复launch；真实故障须先保存错误与终态身份，并确认本项目trainer退出后再采用独立修复／兼容恢复入口。

训练GPU由唯一启动时的实际idle／UUID placement绑定，实际训练卡为0/1/2/3，辅助5/6；活跃进程不重新选卡。`head_device=cuda:0`指当前训练CUDA namespace的第一张已分配卡。不能抢占或停止他人进程，也不用全局Ray stop。GPU当次占用、本execution累计采样峰、完整步骤窗口峰及采样空档分别报告；低瞬时利用率不能判定退出，未计算的窗口峰保持unknown。

`a800_qprobe_qwen_hourly_format.py`是已发布的纯本地formatter，仅消费已保存的这一次primary receipt与真正同run的已交付anchor，不访问远端或改写anchor。正式小时报告按交付anchor到期执行：先发布允许的scalar/hash、实际向用户交付，再推进anchor；普通30分钟观察不推进。报告区分processed、完整outer成功Adam、retained partial Adam、skip、unattempted和可证inflight，四rank副本只计一次；区分拟合前selected512指标与拟合后recent8指标。无新增完整outer、reward或评分须明确说，缺失指标保留unknown，不填0或混入旧run。现有warmup phase anchor不是当前正式训练小时进度。

### 旧操作历史（禁止重新执行）

旧92bb依赖顺序曾包含 `bank-prepare → bank-encode`；全bank已完成，当前不重复该编码。旧92bb的`prepare-launch`曾因OPS与helper创建同名独占claim失败；历史一次性claim修复已执行，后续OPS guard改用`native-formal-operator-`前缀。22:47:32的trainer GPU入口故障是另一真实故障，不能靠重试旧claim修复或launch解决。旧源码、claim和失败凭据保留，但旧92bb／ff1196命令不作为当前监控或启动模板。

## 保存、资源与最终完成条件

checkpoint、Head/Adam、packed token bank、原始 rollout 和完整逐题输出全部保存在显式远端目录；Mac只保留代码、配置、文档、小型scalar/hash与必要审计摘要。已发布代码bundle仅作源码中转；确认目标校验后按本地存储规则处理，未完成目标保留必要副本。

正式checkpoint节点为 **5、50、100、150、200**。每个保存点检查 actor/head/optimizer/scheduler/private RNG 与 replay 的配对身份和实际保存凭据；metadata observer只转述已有证明，不自行做tensor重载。恢复必须使用已验证的完整配对恢复链，不能把旧Head或孤立actor checkpoint冒充新初始化。

完整目标需要：实际完成1..200的正式outer/恢复链、所需checkpoint与有限性/SHA凭据、按原顺序覆盖最终51,713题的binary strict EM/F1与七来源统计（NQ固定3610题），以及最后的only-owned资源清理验收。大规模逐题结果留远端，公开只发布允许的scalar/hash。清理按本任务记录的 PID birth/session/UUID 所有权进行，保留源码、日志、checkpoint与必要复现资产；不能因为节点到期或文档发布提前宣告目标完成。

## 执行状态记录

截至 **2026-10-15 22:35（Asia/Shanghai）**：warmup10/200 HeadAdam及资源释放、full169028bank完整SHA及资源释放、bind和initial0采用均已 accepted。initial0 的 `adopted:true` proof SHA256 为 `811af902563e2e438f02b5ed3ae72d09eca3c813ca19559ed869ad0047c1b920`；其200次warmup Head Adam不计入正式训练进度。

这是22:35时的历史状态。之后独立预启动 claim 修复和一次 launch 已执行，旧失败 claim 保留；最终 trainer 入口故障及清理结果见下文。本文件不把已完成的前置阶段视作正式训练进度。

### 2026-10-15 22:44:07（Asia/Shanghai）历史实测

本次 primary receipt 为 `preparation/qprobe-qwen-native-formal-cpu-repair-v2-native-startup2-claimfixed.json`。源码/发布与实际部署资格、native initialization 均为 observed。pipeline PID=`271789`、birth ticks=`615472668` 实测存活；retriever PID=`272694`、birth ticks=`615481120` 实测存活且状态为 `R`。阶段为 `STARTING_ORIGINAL_FLAT_FP16_TWO_SHARD_RETRIEVER`，正在加载索引，本次 metadata read 未记录错误。

在22:44:07这次观察中，trainer 尚未创建；complete prefix、optimizer opportunity 和正式 Adam 计数当时均为 unknown。该历史存活状态已被之后的真实失败与终态审计取代。

先前 primary 的 `execution_identity` unknown 来自 observer 将 native launch claim 套用了包含 backend 的另一种 schema，不是已验证的 trainer 故障。OPS `098ae78a50945684973a41a2de47ab81825f15ba` 改为校验 source927 实际生产的六字段 typed claim；launch receipt 的 backend 校验以及4rank、source、bank、Head和计数规则保持严格。该修复只更正 metadata reader，不修改 runtime927、模型、奖励、算法或本次启动；原日期 release 文件证明与失败记录保留。

### 2026-10-15 22:47:32失败与22:54:35终态验收（Asia/Shanghai，历史）

source927 的 CPU Ray `main_task` 在 driver GPU hook 中调用实际 CUDART UUID 核验，返回 `AssertionError: Cannot obtain actual CUDA device properties`，尚未进入原训练 body。22:50:23观察已确认 pipeline/trainer gone；终态凭据进一步确认正式 outer=0、正式 actor Adam=0、正式 Head更新=0，没有可伪装为已更新 state 的恢复点。有效096c warmup的200次 Head Adam和已验169,028行 bank保留。

22:54:35的 `qprobe-qwen-native-formal-92bb-terminal-failure-owned-audit.json` 核验34条 owned birth，`terminal_owned_cleanup_accepted=true`，owned SID process、owned CUDA context及项目 trainer live均为空。旧源码、claim、日志和资产保留；禁止重复旧 handle。

当时新隔离fresh-process修复尚在准备：在进程出生前设置已绑定GPU UUID，连接同一私有Ray集群，CPU wrapper只等待；原训练body和科学配置不变，并移交恢复原CPU RNG、保留Head private RNG。当时发布commit/source map、run/execution、handle及启动结果均为pending；此状态已被下文a47／4fd508的实际发布、唯一launch和23:39:13观察取代。warmup与bank复用不重跑。

### 2026-10-15 启动故障修复源码发布（9713／ff1196历史）

修复源码已发布为 `9713a0ce3f33ef22ca5acfa5442ffbcec0bee3f5`，分支 `codex/qprobe-qwen-native-driver-process-fix-20261015`，508 个 runtime 文件的 source map SHA256 为 `c95e96b93cfedc19f1fdff1aff93e1854cdd1e6cd777e5330abe757889929582`；全508文件已从全新Git对象库核验。OPS qualified-bank 复用入口为 `e093c4d3c67a8ee70e8b7a364fc10a9ac99789bb`。新 prospective handle 是 `preparation/qprobe-qwen-native-driver-process-fix-deployment.json`，远端独立目录为 `/data/shaoshitong/Search-G1-a800-fresh200-ff1196a974114d7abe9b1be0c764b721`。此处发布的是修复代码，实际 CPU 资格和正式 launch / 完整 outer 仍须后续 receipt 确认。

CPU Ray 包装任务不再晚改 CUDA 可见性；它同步等待一个在进程出生时绑定准入 GPU UUID 的 Python driver，后者接入同一个 Ray GCS/namespace。四 GPU actor pool、原始科学循环和 CPU RNG 状态保持原样，不增加 GPU 资源请求。旧 namespace 没有 actor 更新，因此新 namespace 是正式0起点的继续执行，不声称恢复了未产生的 actor state。

### 2026-10-15 C4资源审计接口修正与a47部署准备记录

上一修复源码9713的独立准备目录ff1196完成CPU26和warmup验收，但在bank-admit把子进程传给只接收会话主进程的C4接口，实际退出1。无GPU/actor启动、无正式Adam、缓存和warmup未改，旧claim保留且禁止重复执行。

修复runtime为 `a47abb701f62a6c4eb9e8e8c987c67a4cf804098`，source map为 `e15d3e34f487f154403846737e9e17a7fc9d1a6aba3785a746758e5c3b1b5916`（508文件，fresh remote Git核验）。保留每一个PID的实际birth检查，要求每条记录的SID/UID都有已绑定主进程，再仅把主进程交给原C4接口；原PID/SID复用规则未放宽。新handle `preparation/qprobe-qwen-native-driver-process-fix-v2-deployment.json`，base `/data/shaoshitong/Search-G1-a800-fresh200-4fd508e35bc84658ba892a97248f8a7f`，run `524254af06a1472daa1368daba4e0b17` / execution `fd91bd58d4cf40a89b35427267f9d34e`。CPU26/noCUDA与warmup采用已实际通过；169028 bank已从原fullSHA封存+五文件stat/原producer/终态证明验收复用，bank-admit实际exit0，没有再次编码或重hash75GB。随后bind／initial0／prelaunch／launch均已有实际成功receipt，见下文；这些准备结果不等于完成正式outer。

当前4fd508实际Head绑定已exit0：initial Head SHA256 `a269238b0c6c2c182efc058612de895a33f65f7ce9d66966d1973c571fe0f3f1`，ready SHA256 `5b707c21b91594effeef57de1b4b53e276398c4a7bfc837b131986aa7880e462`。这是同次096c nativewarmup的200HeadAdam/private RNG在新formal0 namespace的采用，teacher replay为空，不是actor恢复或新的Head拟合。

实际预启动已exit0：`NATIVE_FRESH_FORMAL200_PRELAUNCH_QUALIFIED` SHA256 `84275048987d2e00f1d5984d4f38b42cafbc706f1e1b9aef09b8505d6e334fa4`。CPU26 receipt `781251a64a91ffc8c064ed8cd6115917b1282d122e9cd63e65e9fc9f3d481b2c`，original initial0采用proof `f646f6a5ae991d8743cdcc09ca58ada97097028376a8fe782db773cc9442a4a8`。所有CPU步骤通过不等于GPU启动或正式训练推进。

### 2026-10-15 23:28:10（Asia/Shanghai）实际启动更新

本次实际部署批准 SHA256 为 `5be741c36e322e9973a2cf299a344ec091886337ae22851b29ba8cb96f18879e`。四张训练卡与辅助5/6均通过空闲准入，唯一 launch 实际 exit0；deployment launch receipt SHA256 为 `ff4e503f23143fe396eb34578a43679fdfb7d8d04a1fe56e0b5da9684f1bf3a8`。新4fd508 pipeline已经一次启动，正在准备检索服务，正式trainer/完整outer/Adam仍待真实推进确认。禁止重复launch；首个有界native只读检查不早于23:33:10，正常监控每30分钟一次，不增加GPU测试或额外评测。

### 2026-10-15 23:39:13（Asia/Shanghai）当前实际观察

`preparation/qprobe-qwen-native-driver-process-fix-v2-native-startup2.json`实际exit0／`READ_RETURNED`，文件SHA256为`fd19e9866c5c730c2fac2f5cd73a807c63b1f002fb031bb6ac49256ac3452dec`。source／run／execution／base与当前a47／4fd508精确相符，已发布runtime和独立actual release资格均为observed。pipeline `310262/birth615768175`、trainer `317298/birth615811482`均存活且birth匹配；retriever与probe-server存活，probe-admission实际exit0，阶段为`GRPO_1_TO_200`，没有记录错误或bounded failure。

已有采样显示四张训练卡均开始占用并有计算活动；这只能证明当前启动流程进入训练阶段，不能单独证明Adam成功返回。正式attempt账本前缀尚未闭合，因此完整outer／optimizer机会／actor Adam／skip／reward／selected512 mixed与accuracy仍为unknown；stage64、dev512和nativefive凭据尚未生成。保留缺失状态，不把warmup的200次Head Adam或99.8828%训练集accuracy当成正式训练结果。下一步沿上述同handle唯一primary核实真实推进和首个完整outer，禁止重复launch或准备动作。

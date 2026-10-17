# Four-GPU prefix20 joint actor/probe experiment

The user confirmed 200 formal actor outers on four HPC3 GPUs. In each outer,
sample 5120 candidate questions, obtain five authentic policy prefixes for each,
rank by `1 - product(p_correct) - product(1-p_correct)`, and take Top512 with
no random exploration. Complete those same 2560 trajectories, preserving their
generated tokens and behavior log probabilities. Actor learning rate and Search
G1 reward remain unchanged.

The downloaded epoch68 probe is the pretrained warmup: no extra warmup is
authorized. Its frozen official Qwen2.5-3B encoder sees the real question/context
and first20 retained policy tokens; detached layer27 post-block features at the
twenty policy positions enter copied block35 and Linear20. Question hidden
positions are not separately passed to the head. Preserve this exported
architecture and standalone safetensors weights exactly. The head is fitted
with its own-trajectory strict terminal EM BCE.

After each complete actor outer, fit the probe on only the newly selected,
completed 512×5 trajectories, once, with Adam LR1e-6, global batch256,
local batch64 across four ranks: ten optimizer updates. No historical replay,
old sigma3 loss, count classification, or group-joint BCE is included.

For strict-EM all-correct G5 groups only, normalize the original five soft
rewards using the mean and sample standard deviation of `[0, *real_rewards]`.
Return the five real advantages, with the original response mask. All-wrong and
mixed groups retain the original GRPO formula. A positive soft reward does not
establish strict correctness. Labels are recomputed at the balanced batch order.

## Immutable sources

- Code origin: `2fd5c19fa0d9c5d6c4eb514a7e9a308eb6bc6479`.
- HF dataset repo: `Sutoonq/Search-G1-prefix20-dataset-probe-epoch68-20261016-111a2d`.
- HF revision: `d794759ab9f28cf184128b227fb38f40b59fcedc`.
- Export frozen implementation: `4bc683883a569d7feb9d5da87307612b872e8f3a`.
- Pretrained head SHA256: `535c7bfee07374dde88b7d7f9e4f08ac874d46088ef961f15ea6ab7dd849a560`.
- Probe archive: 1,145,972,749 bytes, SHA256
  `83d6b834b1d3243085f74a27c7c021722e43ba3fd5004fcf41eb6202954132a5`.

## Implementation state

Acquisition mathematics, single-pass batching and strict-EM virtual-zero
advantages are implemented and CPU tested. The remote-only asset downloader
verifies official HF LFS metadata, both complete part hashes, concatenated
archive SHA, every extracted member, and fresh full reads.

The native trainer now has an independent `prefix20_probe` selection branch.
It closes the candidate-prefix phase, predicts and durably seals Top512 before
continuation, then completes exactly the selected original trajectories. It
reads actual actor Adam/scheduler counters at each boundary and checks actor RNG
equality across probe prediction. Both native phases report their real four
rank receipts; the acquisition pause never masquerades as one native500 call.

Checkpoint schema3 contains four complete probe model/Adam/private-RNG states,
the selection RNG and source stream, bound to the actor snapshot. The existing
actor paired-checkpoint framework validates all four files and compares actual
restored worker state. CPU fixtures do not establish physical GPU acceptance.
The formal monitor readers bind the actual actor micro: four-rank/micro4 has
global masked-mean monitor mass8, while historical eight-rank/micro2 has16.
Both ordinary and transactional attempt readers check this without changing
worker losses, rewards or safety thresholds.

`control.py bootstrap` prepares read-only environment/data links and copies all
published source into a real `base/repo` directory with fresh file SHA checks.
`control.py submit` is the once-only launcher: four GPUs, 48 CPUs, 512GiB,
`emergency_acd` partition, `jxu957` account and `formal-user` QoS. It consumes
fresh selected-asset SHA verification and a complete personal-storage observation
before writing an exclusive submission intent. A submission timeout must be
resolved by inspecting this intent and Slurm; it must never be blindly replayed.

The worker entry is `train.sh` → `prepare_training.py` → the existing common8
pipeline. It verifies the actual submitted Slurm identity, resources, command
and working directory. No feature job or extra teacher warmup is launched.
The original reward probe calibration and retriever still execute. Native
retriever UUID queries use the actual four/eight device count.

Storage layout is explicit: five complete actor/probe sources on HPC3, five
complete archives on HKUST3, and only one simultaneous full recovery target on
HPC3. The reservation includes FP32 actor weights plus both Adam moments,
four complete probe replicas, reward-encoder dependencies, and a separately
identified 96GiB estimate for raw/log artifacts. That estimate must be checked
against this run's measured artifact growth. Existing checkpoints are retained;
this layout does not authorize deletion of other experiments.

Actual deployment and job state are recorded separately in the operator's
durable handoff and measured observations. CPU tests, preparation, submission,
and a Slurm RUNNING state do not prove a complete four-GPU training iteration.

Never treat the artificial20-token pause as an invalid action or insert an
environment correction at that boundary. Preserve actual20/remaining-budget
sampler records; do not relabel a20-token request as a500-token request. The
pre-launch reviewer must check split boundaries, short terminal trajectories,
cross-boundary closing tags, exact row/group identity, masks and original LP.

Checkpoints, original trajectories and tensors remain remote. Local storage is
limited to code, configuration, manifests and small audit summaries. Existing
healthy experiments are not restarted or edited. Formal update201 is forbidden.

## 实测运行记录：2026-10-17 04:46 CST

HPC3 作业 `727586` 使用四张 GPU 运行，训练源码为
`478f46315147ebb0d4d1c556fcdd356c658e117e`，run ID 为
`e4cf46d47ae24434a177931df5530751`。正式目标仍为 200 个 outer，禁止
update201。直接加载导出的 epoch68 探针权重，重新初始化探针 Adam，不增加
warmup 或 replay。每轮从 5120 道候选题取得各五条真实 20-token 前缀，按
`1 - product(p_correct) - product(1-p_correct)` 选择 Top512，无随机探索；
续跑同一批 2560 条轨迹，先更新 actor，再用这些新轨迹的逐轨迹 strict-EM
标签训练探针。探针使用 mean BCEWithLogits、LR1e-6、global batch256、
每卡 batch64，单遍十次 Adam 更新。Actor 沿用原 Search G1 设置，仅 strict-EM
全对组使用加入虚拟零后的 soft-reward 均值和样本标准差计算五条真实优势。

截至该次实测，已完整完成 15 轮 actor＋探针联合更新，第 16 轮开始；每卡
累计 actor Adam300 步、scheduler15 次、探针 Adam150 步，未发现新的流程
错误。第 15 轮选中题目的真实混合比例为 205/512＝40.0391%；同轮新
2560 条轨迹的探针 BCE 为训练前 0.710061、训练后 0.701158，二分类准确率
为 52.0703%→53.5156%。混合比例描述选中题目五条真实轨迹的标签组成；
BCE 和二分类准确率描述逐轨迹 EM 预测。训练前预测尚未拟合本批标签，
训练后指标属于同批数据拟合，二者均不能代替独立评测的泛化结果。

第 5 轮固定评测已完成，其中独立 dev512 的 EM 为
134/512＝26.171875%，F1 为 0.350591。初始评测按用户要求跳过，记录中
`before=null`、`changes={}`，因此没有相对初始评测的改善结论。固定
50/100/150/200 轮评测，以及最终七源 native51713 的 EM/F1 macro 与
weighted 结果，尚待实际完成。

第 5 轮完整检查点已完成 HPC3 源文件→HKUST3 归档→HPC3 源文件的全文件
SHA 读回验收，含依赖与边界记录共 56 个文件、48,560,081,233 字节；原始
源文件保留。这不等于实际 CUDA 恢复验收。最终仍须完成 200 轮、其余四个
里程碑归档和实际四卡 model/Adam/scheduler/RNG/probe/dataequal 恢复验收。

本条记录依据操作目录中的 `handoff.json`、
`727586-goal-actual-outer15-completion-1792183563.json`、
`727586-actual-first-five-summary41-1792176988.json` 及
`checkpoint5-archive-acceptance-summary.json`；这些是已保存的实际观测，
运行进度以其后带时间的观测为准。

## 固定第 50 轮评测实测：2026-10-17 10:45 CST

同一作业 `727586`、同一 run `e4cf46d47ae24434a177931df5530751` 已完整
完成 50 轮 actor＋探针更新；每卡累计 actor Adam1000 步、scheduler50 次、
探针 Adam500 步。固定 dev512 的实际结果如下，F1 为逐题 F1 的加权平均：

| 固定评测轮次 | EM 正确数 | EM | F1 |
| --- | ---: | ---: | ---: |
| 5 | 134/512 | 26.171875% | 0.35059090280429955 |
| 50 | 199/512 | 38.8671875% | 0.4654389880952381 |

两次评测的 512 个有序题目 ID 及对应来源逐项完全相同，均含 NQ256 题和
HotpotQA256 题；共同 validation SHA256 为
`3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb`。
第 50 轮较第 5 轮多答对 65 题，EM 提高 12.6953125 个百分点，F1 提高
0.11484808529093855（11.4848085291 个百分点）。初始评测仍为跳过状态，
这些差值只比较第 5 轮与第 50 轮。

该次观测中，第 50 轮 actor 保存及评测已记录，原 Search G1 奖励探针的
第 50 轮刷新位于后续 completed-pair 发布之前；`pair50=null`。因此当时
尚未验收第 50 轮完整配对检查点及其归档，也未验收实际 CUDA 恢复，不能
据此认定第 51 轮已开始。未出现新的 pipeline-error、training-return 或
cleanup 记录。最终 200 轮、其余固定评测、native51713 七源评测、全部
里程碑归档及实际四卡恢复验收仍待完成。

训练运行源码保持 `478f46315147ebb0d4d1c556fcdd356c658e117e`；归档操作器
修复源码为 `8239ab35d72533afe7fdd63ec61c5334b9e72cab`；此前第 15 轮文档
记录提交为 `18fe022c662e60e77a127e70f2afc188275a38c6`。本条仅补充文档，
不改变运行源码、算法、超参数或已有作业。依据已保存的实际观测
`727586-fixed5-50-paired-evaluation-summary-1792205155.json`，其中有序 ID
与来源相等均为实际比较结果。

# 当前10项目标的证据与收尾缺项

本表按当前goal编号整理，依据已有本地报告；本次没有HPC访问、新推理、实验、测试或数字重算，不阻止正在进行的训练。**最新人工选择为L15／Prefix80，覆盖更早“重点L21”的安排。** L21/P20及其它层的已完成结果保留为历史对照，不能据旧安排继续扩网格或替换当前主实验。

当前2048题NQ开发集已经被反复查看，用于层、前缀、损失、epoch及raw/EMA探索，不能再称未触碰测试集。当前全量L15/P80两seed均已完成10epoch、13210次Adam/EMA更新并以`COMPLETED0:0`退出；两份E10与按原seed1-only规则选定的**E9/raw**断点均已通过实际CPU终验。当前P80换序audit作业750474已唯一提交并为RUNNING，尚待实测推进与结果；actual candidate freeze及两个盲2048池的最终确认仍未完成，final4096保持盲态。旧fullP20三臂结果只作历史对照。

状态含义：“已有实测”仅覆盖所链接的历史/阶段工作；“仅部署”不等于运行或验收；“待完成”不以Slurm RUNNING、source-stage、模板或CPU元数据测试代替实际结果。

| goal | 已完成或已有实测证据 | 现有源码／回执入口 | 尚缺事项与解释边界 |
|---|---|---|---|
| **1. 固定基线；开发与final4096分离；同question/G5/strictEM** | [开发决策表](CURRENT-DEVELOPMENT-DECISION.md)区分原E1、同2048 ownG5开发池及trainA/trainB；[开发与固定B物理源核验](viewed-label-ownEM-audit/README.md)、[全81920 pilot验收](viewed-label-ownEM-audit/actual-all81920-pilot-physical-accepted.json)已核真实ownEM、执行/seal、完整payload SHA及前缀。 | [全量physical-v2最终输入验收](full169028-P80-physical-v2/actual-final-acceptance.json)；[真实源问题说明](full169028-P80-physical-v2/PHYSICAL_SOURCE_INCIDENT.md)。 | 开发指标有可解释性，但已经参与选型。**最终4096仍须在实际freeze之后按原两个固定2048 ownG5分别确认**，不得借旧开发提升宣布未知问题泛化成功。协议tid重复不等于同一物理执行；已证实的源选择问题不能误写成canonical标签错误。 |
| **2. 已修采样对照；全169028 ownG10每epoch洗牌；随机ownG5；实际子集标签** | 已保留[修正去重随机化对照](unique-question-g5-randomization/RESULT.md)、[均衡与天然G5对照](unique-vs-labelblind-g5/RESULT.md)。当前全量源使用169028题每epoch一次、独立洗牌、自然uniform5-of10；[750190实际E1计划统计](fullMSE-750190-actual-E1-sampler/RESULT.md)直接读取真实计划，确认每题唯一、ownID/EM及自然组合覆盖。 | [全量运行源](../../operations/qwen7b_warmup/online_packed_full169028_MSE_P80.py)，已部署源commit `7a8ff6ae8a33324846d85f77ffc28e9ab699c028`；[当前协议](full169028-L15P80-MSE-ownG5/PROTOCOL.md)。 | 当前两seed已完成full10epoch，E10及selected E9的CPU回执均核过10份实际计划SHA与恢复状态：[两E9汇总](full169028-L15P80-MSE-CPU-acceptance/BOTH-SELECTED-E09-CPU-ACCEPTANCE-SUMMARY.json)。旧强制EM组合／hard replay结果须保留其采样差异，不能充作现在自然随机采样的公平前后比较；本目标不恢复强制类别组合。 |
| **3. 并行直接Mixed与方差；同packed block／预算BCE对照；MSE与Z+anchor** | [matched packed四格实跑记录](matched-packed-objective/actual-four-latest.json)及[已验收共用输入](matched-packed-objective/ACTUAL-STAGED-FIXTURE-ACCEPTED.md)保留BCE/MSE × fixedG5/randomM的匹配设置；[决策表](CURRENT-DEVELOPMENT-DECISION.md)保留fullP20 MSE/MAE/Z E10 raw及EMA，未只报best。 | [matched实现](../../operations/qwen7b_warmup/online_packed_matched_objective.py)、[Z+anchor源](../../operations/qwen7b_warmup/online_packed_variance_scaling.py)、[当前full MSE源](../../operations/qwen7b_warmup/online_packed_full169028_MSE_P80.py)。 | 已选MSE-M5 E7与CURRENT64 E5的差异混合epoch、global batch及更新预算，**不是单一loss因果结论**。当前只扩展领先全量L15/P80/MSE配置，不恢复轨迹BCE，不因为尚待全量结果再扩损失网格。 |
| **4. 有界层／前缀探索；最新L15/P80；冻结actor、训练block＋head** | [层/前缀历史表](CURRENT-METRICS-TABLE.md)与[各planned pilot逐epoch结果](planned-pilot-all-epochs-viewed2048/actual-result.json)保留已做有界探索；当前匹配packed架构及全量入口为L15/P80，official actor冻结、一个复制block＋2048位置head可训练。 | [current full协议](full169028-L15P80-MSE-ownG5/PROTOCOL.md)、[两seed实际v3入口与ready](full169028-L15P80-MSE-ownG5/actual-two-source-stage-v3-handoff.json)。 | 最新人工L15/P80覆盖旧L21重点；旧L21结果只作对照。当前full P80两个seed已完成全10epoch；[实际family选择](full169028-P80-MSE-family-selection/actual-run.json)按原seed1-only规则选E9/raw并原样应用seed2，没有根据seed2重选。 |
| **5. 固定G5与randomM3..10；padding／顺序；防未来答案／EM泄漏** | fixedG5/randomM控制共用真实ownG10及预算。原[Z L21/P20 E8换序实测](packed-Z-E8-trajectory-permutation-audit/RESULT.md)已完成：original精确复现旧分数，reverse/cyclic1揭示可测顺序敏感，未选择有利顺序。当前pack仅含问题与实际截断前缀，EM/trueMixed/目标不进入输入；短回答按原mask padding。 | [全量L15/P80换序审计source-stage](full169028-L15P80-permutation-audit/actual-stage.json)、[协议和边界](full169028-L15P80-permutation-audit/README.md)；源commit `97f57e4fbed1467507fee364a23da2c15091aca6`。 | **L15/P80当前full模型的换序审计已绑定两seed同一E9/raw并唯一提交750474**：[提交回执](fullP80-selected-root-control/actual-audit-submission.json)。仅有RUNNING，[首读](fullP80-selected-root-control/actual-audit-live.json)尚无实测评分进度，不能称审计通过；须对original/reverse/cyclic1全部报告，不按审计结果再挑顺序。真实前缀可能自然含回答内容；禁止的是越过prefix截断的未来后缀和EM标签泄漏，不能宣称真实prefix里永不出现答案词。 |
| **6. Mixed／全对／全错拆分；困难负例覆盖；组间Mixed排序** | [Top512成分分析](viewed2048-two-seed-P80-paired-comparison/RESULT.md)区分全错/Mixed/全对；已有提升主要减少全对干扰，余下错误以全错为主。[真实全量E1采样报告](fullMSE-750190-actual-E1-sampler/RESULT.md)已量化G10-Mixed抽成G5全错/全对的困难负例，并与精确超几何期望对照。 | [同池详细classwise表](CURRENT-DEVELOPMENT-DECISION.json)、[实际E1 sampler独立回读](fullMSE-750190-actual-E1-sampler/actual-independent-result-confirmation.json)。 | 自然采样已覆盖困难负例，不能据单轮统计声称完全解决排序。当前两seed逐epoch诊断与[实际family同题比较](full169028-P80-MSE-family-selection/actual-run.json)已保留；这些仍是已浏览开发集结果，需最终盲池确认，不以轨迹准确率替代本目标。 |
| **7. 按目标选CP；raw梯度／EMA.9995；完整Adam/RNG/采样恢复** | 当前raw持续训练、更新EMA.9995；[提前公布的选型规则](full169028-L15P80-MSE-ownG5/PREDECLARED-FULL-MSE-SELECTION.json)明确只用seed1开发Top512选择epoch/state，ties取早epoch再EMA，原样应用seed2。历史pilot有[seed1已选CP接受](matched-window-CPU-acceptance/three-selected-actual-accepted.json)、[seed2接受](seed2-matched-window-CPU-acceptance/seed2-selected-current64-E5-and-MSE-E7-actual-accepted.json)。 | [full保存／resume实现](../../operations/qwen7b_warmup/online_packed_full169028_MSE_P80.py)、[full terminal CPU验收源](../../operations/qwen7b_warmup/accept_full169028_L15P80_MSE_checkpoint.py)。 | **当前full实际selected为E9/raw，两seed对应E9均已CPU验收通过**：[完整汇总](full169028-L15P80-MSE-CPU-acceptance/BOTH-SELECTED-E09-CPU-ACCEPTANCE-SUMMARY.json)，两回执核实际CP SHA、raw/EMA各45张量、Adam11889、两rank RNG、ownG5计划与physical/P20链。两E10也已通过；CPU可加载不等于CUDA下一步恢复已实测，family内选择不等于actual candidate freeze。 |
| **8. 同训练／开发池BA、召回／FPR、错误拆分、Top512；分清方差与概率** | [CURRENT-DEVELOPMENT-DECISION](CURRENT-DEVELOPMENT-DECISION.md)的15条固定诊断分开trainA/trainB与开发；BCE、variance MSE/MAE单位分开，缺失固定TopK loss保持缺失，未拿在线batch loss补。后续P80已有同trainB分数。 | [统一固定trainB scorer部署](selected-state-fixed-trainB/actual-stage.json)、[E1固定32 adapter回放协议](selected-state-fixed-trainB/README.md)、[root绑定与唯一提交控制](selected-state-fixed-trainB/root-control/actual-stage.json)；scorer `5899b504`、控制 `45bccf6e`。 | **E1 EMA的统一trainB补评分和固定32旧开发adapter数值回放已实际完成**：[750334 v2完整结果](selected-state-fixed-trainB-v2/RESULT.md)，Slurm `COMPLETED0:0`，32组最大概率差5.21e-8＜0.001。首次v1绑定前发现旧`strict_em`字段兼容错误且未提交，独立v2已修复并保留历史；两Z E10/raw已有B预测只作精确同qid/orderedtid/EM复用。不得把旧trainA并入B，方差0.08诊断、TopK排序分数也都不是校准Mixed概率。 |
| **9. 少数领先方案≥2seed；同题配对CI；freeze后两个盲2048** | [P80两seed配对报告](viewed2048-two-seed-P80-paired-comparison/RESULT.md)已有历史CURRENT64 E5、MSE-M5 E7的两seed证据；[同E10窗口比较](viewed2048-current64-window2018-sameE10-comparison/RESULT.md)匹配epoch/预算。其窗口平均提升+2.2461pp的题目配对95%区间触零，不能写成稳定收益；这些CI不代表训练seed不确定性。 | [full final admission协议](frozen-full169028-final4096-admission/README.md)、[最终架构适配source-stage](frozen-full169028-MSE-final4096-scoring/actual-stage.json)、[actual独立源码回读](frozen-full169028-MSE-final4096-scoring/actual-independent-source-readback.json)。 | **当前全量P80两seed完整10epoch、E10/selected E9 CPU验收及同E9/raw开发比较已完成**：[实际family结果](full169028-P80-MSE-family-selection/actual-run.json)。尚缺实际换序审计结果、actual候选freeze、两个盲2048的真正打分与配对结论；final scorer部署或模板不能替代这些事项，不得挑best seed或根据最终池重新选epoch/state/阈值。 |
| **10. 吞吐／两卡累积／复用；及时push；真实进展和最终验收** | 当前full入口两H100、local64/micro8/accum8，正常global128；真实tail34/rank、global68按实际归一化，full1321更新/epoch。输入轨迹/Policy80/P20已验收复用；[seed1两次真实推进](full169028-L15P80-MSE-ownG5/seed1/actual-two-reads-real-training-proof.json)、[seed2首两轮完整进展](full169028-L15P80-MSE-ownG5/FIRST-TWO-EPOCHS-SEED2.md)属于实际进度证据。 | [数据最终ready](full169028-P80-physical-v2/FINAL_READY.md)、[发布/入口清单](full169028-L15P80-MSE-ownG5/actual-two-source-stage-v3-handoff.json)、[全量terminal验收入口](../../operations/qwen7b_warmup/deploy_full169028_L15P80_MSE_CPU_acceptance.py)。 | 当前两训练作业均已`COMPLETED0:0`，完整保存/评价、双seed与E10/selected E9 CPU验收已完成。**audit提交或RUNNING仍不是实测审计通过，也不是全目标完成**；剩余为audit实际推进/验收、actual freeze、final确认、任务资源收尾及最新源码回执归档。这份证据文档不触发新review、测试或实验。 |

## 已有数字的正确用途

这里只引用既有报告，不重新选择或计算：历史8192题MSE-M5 E7/raw两seed的Top512平均54.20%是已浏览开发集选题Mixed占比，对应开发BA平均64.44%；不是54.20%的BA，也不是当前full10epoch的结果。CURRENT64与WINDOW2018的正确机制对照是同E10/raw，窗口平均+2.2461pp、题目配对95%区间[0,+4.6875]触零。详见[决策表](CURRENT-DEVELOPMENT-DECISION.md)及两份已存[选定CP配对报告](viewed2048-two-seed-P80-paired-comparison/RESULT.md)、[同E10机制报告](viewed2048-current64-window2018-sameE10-comparison/RESULT.md)。

新full两seed的完整开发结果已由[实际family选择与配对报告](full169028-P80-MSE-family-selection/actual-run.json)按同qid/ordered ownG5/strictEM比较；E9/raw来自seed1-only原规则，seed2未重选。开发仍参与探索，这份完整开发比较不等于两个盲2048的最终确认。

## 前瞻最终判据：仍不是actual freeze

根已在最终池未读取前写下[PREDECLARED-FINAL-CONFIRMATION.json](frozen-full169028-MSE-final4096-scoring/PREDECLARED-FINAL-CONFIRMATION.json)：主指标固定为Top512真实Mixed占比；每个seed×每个最终2048池相对E1的点差均须严格大于零，且两个池分别的two-seed均值相对E1的配对问题bootstrap95%下界须严格大于零。该区间不冒充训练seed不确定性。BA、召回、全错/全对误报、方差误差及AUC只是secondary，最终读取后不能改用BA替换Top512主指标。

该文件状态为`PREDECLARED_ANALYSIS_RULE_NOT_A_CANDIDATE_FREEZE`。family内epoch/state现已实际选定E9/raw；actual candidate freeze尚未形成，最终标签/分数仍未读取。若未通过前瞻主指标判据，不得宣布goal达成；这些最终池随后须标记为已查看，再选型必须另留新的未知问题。

## 最终收尾顺序

1. **已完成**当前full L15/P80 MSE两个seed的10epoch、逐epoch固定诊断、E10 CP及`COMPLETED0:0`真实退出，两E10 CPU终验均通过；旧fullP20首seed终态不替代本项。
2. Z第二seed实际CPU终验和统一trainB／E1固定32 adapter回放（750334）均已完成；现有Z/P80同池预测经精确身份校验后复用，旧trainA继续单列。
3. **已完成**按原**seed1开发集**规则选择E9/raw，两个seed对应E9实际CP均通过CPU验收；没有按seed2或trainB重新选型。
4. **已唯一提交、待实测与验收**：750474以normal/acd_u、两GPU执行当前full E9/raw预设换序审计；须报告全部顺序，不按结果选择有利顺序，并确认原顺序与同CP已存score一致。
5. 写下实际candidate configuration、主指标、epoch/state、阈值、两seed CP/source/CPU proof及原两个固定G5身份，形成**actual freeze**。stage模板不能充当freeze。
6. 之后才在两个未参与选型的盲2048问题池运行最终统一评价，报告每seed、每池及预先定义的比较/不确定性，完成checkpoint、最新source和任务资源收尾验收。

**目标必须在未参与选型的新问题上形成稳定、可复核的提升；当前前瞻规则已经具体固定为Top512主指标的上述判据，不能只凭BA改善替代。作业正常结束、开发指标变好、源码已部署或模型可加载都不能独立满足这一判据。**

## 2026-10-23 08:33 CST 实际收尾进展

两full P80作业750189/750190均已完成10epoch、13210次Adam/EMA更新并实际`COMPLETED0:0`。原有协调器顺序完成E10 CPU终验后自然退出，无failure：[seed1原始E10回执](full169028-L15P80-MSE-CPU-acceptance/seed1-actual-E10-terminal-accepted.json)、[seed2原始E10回执](full169028-L15P80-MSE-CPU-acceptance/seed2-actual-E10-terminal-accepted.json)。这是CPU恢复验证，不宣称GPU下一步恢复已实测。

[实际family选择](full169028-P80-MSE-family-selection/actual-run.json)遵循已预先公布的seed1-only规则，选定**E9/raw**并原样应用seed2；[两selected E9验收汇总](full169028-L15P80-MSE-CPU-acceptance/BOTH-SELECTED-E09-CPU-ACCEPTANCE-SUMMARY.json)与两份原始回执已齐，分别为11889次Adam/EMA更新，完整恢复链通过。E10通过不能替代selected E9，现两者分别留证。

[根控制器](fullP80-selected-root-control/README.md)已将两E9/raw与CPU proof绑定到换序audit并唯一提交**750474**，normal/acd_u、两GPU、绑定SHA `1bca7df0a5ba108758f7082469b2352e260fbc711047382a99ea1f8925a1acdc`。[提交回执](fullP80-selected-root-control/actual-audit-submission.json)及[首读RUNNING元数据](fullP80-selected-root-control/actual-audit-live.json)不等于评分推进或审计通过；仍待真实进度、原顺序数值回放及全部预设顺序结果。

**actual candidate freeze、最终评分和最终确认尚未完成，final4096仍盲。** 当前剩余为audit实测与验收、actual freeze、按原两个固定2048池执行最终确认及资源收尾；不能把family内E9/raw选择或训练/CPU正常完成当成整个goal完成。本次只更新本地既有文档，未访问远端、CP、最终池或修改任务。


### 08:36 CST actual audit retry

First owned audit750474 failed during initial CUDA device initialization on the previously avoidedACD1-13, before model loading or prediction. The original entry omitted this known node exclusion; no algorithm/runtime change was needed. Root retained actual FAILED1:0 and both rank failures, verified no live same-name job/free task locks, then uniquely submitted unchanged bound audit750476 excludingACD1-13. Actual retry RUNNING onACD1-11, inference progress/complete pending. See [retry receipt](fullP80-selected-root-control/actual-audit-CUDA-retry-submission.json). Final freeze/scoring remain pending and final4096 remains blind.

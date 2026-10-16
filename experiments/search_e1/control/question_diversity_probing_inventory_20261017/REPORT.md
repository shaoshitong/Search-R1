# Question diversity probing 全历史实验清单

历史完整扫描：16个置顶会话，1,224,052条JSONL记录，4.001GB原始历史；扫描完成于2026-10-17 01:27:44（北京时间）。本地实验receipt补查至本报告生成时。

本清单共 **44项设置／阶段／诊断记录**，另列 **116条执行、恢复、准备或辅助尝试记录**。前者不是44个独立RL实验；同一设置的warmup和正式RL有继承关系，恢复job也不是新算法。

读取了完整会话快照，而不是仅最近10轮。历史中的goal反复回放已去重；内部工具、父子agent与历史方案没有当作新人工授权。没有向原agent发消息，没有改变训练，也没有下载权重、tensor或完整rollout。

## 指标口径

- **QA EM**：模型实际回答的严格正确率；固定512题QA评测与51713题终评分别标明。
- **mixed分类accuracy／balanced accuracy**：预测同题五条轨迹是否有对有错；不是QA EM。
- **轨迹correctness分类accuracy**：从question/prefix预测其自己的最终EM；也不是模型推理QA EM。
- warmup的epoch／HeadAdam、正式actor outer／Adam分别记；部分outer不计完整outer。
- Prefix-only的hidden仍含question上下文；“只接生成位置”不等于“没有问题信息”。
- 同题G5样本std采用ddof=1时只有0、sqrt(.2)、sqrt(.3)；Tree二次挑题的population variance与其相差常数因子，不能把训练代码里的variance误写成std。

## 全部设置与阶段

### 原始设计与未执行消融

| ID | 实验／阶段 | 实际进度 | 结果／异常 |
|---|---|---|---|
| D01 | Question-only QDP 原始设计与探索比例消融 | 已形成方案、实现与CPU检查；不能把四档都称为已跑完的RL对照。 | 已运行的E=64和E=0分支分别见下；没有找到E=128/256独立正式训练结果。 |
| D02 | Pairwise ranking／RankNet 增补建议 | 找到建议文本；未找到独立run或实际loss启用凭证。 | 不能当作已做过的消融。 |

**D01 — Question-only QDP 原始设计与探索比例消融**

冻结官方模型第27层末有效question-token表示；LN→MLP64→K=0..5六分类；CE6+0.1均值MSE+0.1归一化样本std MSE；按期望样本std选题。E=0/64/128/256四档，入训均512×5。

身份：原实现734f05ead727430e96178f7aa23ddd0c57d38226；branch codex/search-g1-question-diversity-probe。所属会话：`01a11c5d-303b-7441-8db7-c645df252713`。

证据：[question-diversity-probe.md](/Users/shaoshitong/Documents/Search-G1/qprobe-teacher-warmup-a800/docs/question-diversity-probe.md) · [01a11c5d-303b-7441-8db7-c645df252713-chronology.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a11c5d-303b-7441-8db7-c645df252713-chronology.json)

**D02 — Pairwise ranking／RankNet 增补建议**

若std拟合改善而Top命中率未改善，再加入同policy高std题应排在低std题之前的排序loss。

身份：见下方实际证据；未虚构run identity。所属会话：`01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef`。

证据：[01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef-chronology.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef-chronology.json)

### Co会话：小MLP与σ3

| ID | 实验／阶段 | 实际进度 | 结果／异常 |
|---|---|---|---|
| C01 | HPC3 小MLP 原始预热 | 40次HeadAdam完成；actor更新0。 | 同一缓存可复用，但不同loss的head不能冒称同一warmup。 |
| C02 | HPC3 原始MLP QDP 正式RL | 完成117/200后按人工新loss指令停止。 | 固定512：iter50 35.9375%（184）；iter100 41.2109375%（211）。 |
| C03 | HPC3 σ3-loss 预热 | 40次HeadAdam完成。 | warmup总loss1.1206166；不是QA EM。 |
| C04 | HPC3 σ3-loss 正式RL | 200/200训练与51713题评分完成；清理／归档验收尚未全部通过。 | 固定512 iter200 41.9921875%（215）；终评20949/51713＝40.5101%，七集等权32.9841%。 |

**C01 — HPC3 小MLP 原始预热**

5120题×5 frozen-teacher；MLP64，135622参数；原六分类＋均值/std辅助；2epoch，batch256。

身份：run c3856315d3f345d7920ebf5725a083b4；恢复后主job719225。所属会话：`01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef`。

证据：[handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json)

**C02 — HPC3 原始MLP QDP 正式RL**

Top448＋随机64；每outer入训512×5；在线recent16 replay，head每outer2epoch。

身份：job719225；run c3856315d3f345d7920ebf5725a083b4。所属会话：`01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef`。

注意：人工更换设置，不标为数值崩溃；早期启动失败见附录。

证据：[handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) · [01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef-chronology.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef-chronology.json)

**C03 — HPC3 σ3-loss 预热**

相同MLP64；CE3＋1×normalized σ MSE＋0.1 CE6，取消mean MSE；5120×5，2epoch。

身份：job721485；run70827e85fe584f0c84453aae907918a4。所属会话：`01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef`。

证据：[handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-sigma3-loss-20261015/handoff.json)

**C04 — HPC3 σ3-loss 正式RL**

λσ=1，Top448＋随机64；recent16×512组，2epoch，head batch256；其余保留Search-G1 shaped reward。

身份：job721485；run70827e85fe584f0c84453aae907918a4。所属会话：`01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef`。

注意：包装器报trainer left owned subprocesses；评分完成不等于全部收尾验收。handoff旧进度已由实际终评读回覆盖。

证据：[handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-sigma3-loss-20261015/handoff.json) · [jxu957-current.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/session-refresh-20261017-pinned/jxu957-current.json) · [all-seven-results.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/session-refresh-20261017-pinned/all-seven-results.json)

### Search-G1会话：MLP、完整Qwen及Prefix5

| ID | 实验／阶段 | 实际进度 | 结果／异常 |
|---|---|---|---|
| A01 | A800 原始teacher warmup | 首次采样快照契约失败；修复与缓存复用后的实际warmup完成。 | 用于后续micro2／micro4／async分支；warmup不算actor outer。 |
| A02 | A800 question-only QDP micro2 | 1完整outer；第二轮6次部分Adam；合计26次／rank，人工换micro4。 | 首轮reward0.2495036；无可冒称的200轮结果。 |
| A03 | A800 question-only QDP micro4 | 完成4个outer，随后故障；旧run保留。 | 未找到该run可用的固定512里程碑成绩。 |
| A04 | A800 async Top512 原始MLP | 首执行完成5后native校准identity故障；同run恢复链后达到38完整outer，再按人工换loss。 | 固定512 iter5 EM24.609375%；32个普通outer计时曾被独立分析。 |
| A05 | A800 σ3-loss λσ=3 预热 | 真实warmup完成。 | 总loss约1.6003；不能与Co的λσ=1直接称同设置。 |
| A06 | A800 σ3-loss λσ=3 正式RL | 至少3完整outer，下一outer有部分更新；后续人工切10epoch warmup。 | 无完整固定512里程碑可据此排名。 |
| A07 | A800 σ3-loss warmup10 | 10/10完成。 | 训练题二分类accuracy69.86%；不能当QA准确率。 |
| A08 | A800 σ3 warmup10 后正式RL | 2完整outer、40成功Adam／rank、0skip；人工换架构，资源清理验收通过。 | 无该分支固定512里程碑。 |
| A09 | 全question-token MLP128＋4-head attention warmup10 | 10epoch／200更新，exit0；没有正式actor训练。 | loss0.765503→0.587306；训练accuracy41.54→72.23%；Top512真实混合39.26→79.69%，epoch8曾84.38%。 |
| A10 | 完整原生Qwen最后block question-only warmup10 | 10epoch／200HeadAdam完成。 | 训练accuracy99.8828125%（5114/5120），BCE0.0366394；Top512全mixed。 |
| A11 | 原生Qwen question-only 正式RL | 1完整outer＋第二轮10部分Adam，总30／rank；人工停止后新512诊断。 | 单轮主流程约21分30秒；无5轮QA EM可报。 |
| A12 | Head200 CPU复现＋全新512题G5泛化诊断 | 新512×5真实轨迹全部完成。 | 旧训练99.88%；新题mixed分类58.7891%（301/512），低于总预测非mixed基线64.2578%。 |
| A13 | A800 question＋Prefix5 单轨迹BCE，balanced warmup20 | 实际6完整epoch＋第7轮91次，合计691Adam后人工停止；旧不平衡fit从未启动。 | 新题轨迹判别balanced accuracy最好epoch4 58.4280%，epoch6 51.7827%。 |

**A01 — A800 原始teacher warmup**

冻结teacher5120×5；question-only MLP六分类＋均值/std；2epoch／40HeadAdam。

身份：见下方实际证据；未虚构run identity。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

证据：[qprobe-teacher-warmup-a800-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-deployment.json) · [qprobe-micro4-superseded00028-terminal.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-micro4-superseded00028-terminal.json) · [qprobe-async-top512-terminal-authority-before-fullouter-restore.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-async-top512-terminal-authority-before-fullouter-restore.json)

**A02 — A800 question-only QDP micro2**

四卡actor micro2版本；原六分类QDP；不是后来micro4正式新run。

身份：run00028e46b70b40a2ba6980b1c988f92c。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：人工改micro配置；不把部分outer算完整。

证据：[qprobe-micro4-superseded00028-terminal.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-micro4-superseded00028-terminal.json)

**A03 — A800 question-only QDP micro4**

四卡micro4／global16；保持mini128及20Adam/outer；原六分类head。

身份：run34fffb9fe8ad4d75b54b9c9747c256fb；base d8569124…。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

证据：[qprobe-micro4-terminal-authority-before-async-top512.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-micro4-terminal-authority-before-async-top512.json)

**A04 — A800 async Top512 原始MLP**

候选5120→Top512，无随机64；整outer采样权重同步、逐轨迹状态机；保留原六分类loss。

身份：run1f34290a70454871aea9d368dc8a0b36；a365→0065（被替代修复）→af113恢复。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：恢复与目标回放不是新算法；原5轮checkpoint、继承计数与新写入分开。

证据：[qprobe-async-top512-terminal-authority-before-fullouter-restore.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-async-top512-terminal-authority-before-fullouter-restore.json) · [qprobe-async-restore5-terminal-metadata.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-async-restore5-terminal-metadata.json) · [01a0fb1b-d3ff-74b3-aa13-49ca57871f92-chronology.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fb1b-d3ff-74b3-aa13-49ca57871f92-chronology.json)

**A05 — A800 σ3-loss λσ=3 预热**

CE3＋3×normalized σ MSE＋0.1 CE6；相同小MLP；5120×5，2epoch40更新。

身份：见下方实际证据；未虚构run identity。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

证据：[qprobe-warmup10-oldf847-terminal-cleanup-accepted-archive.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-warmup10-oldf847-terminal-cleanup-accepted-archive.json) · [qprobe-sigma3-f847-hour01.json](/Users/shaoshitong/Documents/Search-G1/repo/reproduction/reports/qprobe-sigma3-f847-hour01.json)

**A06 — A800 σ3-loss λσ=3 正式RL**

Top512，无探索；teacher-v4修复后正式训练。

身份：runf84749ca43be42bd91f2bc63d73347dd；base86cf…。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：前两个失败namespace/c528与1573单列为执行失败。

证据：[qprobe-warmup10-oldf847-terminal-cleanup-accepted-archive.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-warmup10-oldf847-terminal-cleanup-accepted-archive.json) · [01a0fb1b-d3ff-74b3-aa13-49ca57871f92-chronology.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fb1b-d3ff-74b3-aa13-49ca57871f92-chronology.json)

**A07 — A800 σ3-loss warmup10**

沿用小MLP／σ3监督，把同5120题预热延长到10epoch／200HeadAdam。

身份：runf4fb19b1fede4dffaac823b8937f35cf。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

证据：[qprobe-warmup10-f4fb-terminal-cleanup-accepted-archive.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-warmup10-f4fb-terminal-cleanup-accepted-archive.json) · [qprobe-token-binary-warmup10-fulltoken-results.md](/Users/shaoshitong/Documents/Search-G1/repo/reproduction/reports/qprobe-token-binary-warmup10-fulltoken-results.md)

**A08 — A800 σ3 warmup10 后正式RL**

与A07同run，预热后从官方actor开始。

身份：runf4fb19b1fede4dffaac823b8937f35cf。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：不是训练故障，不能误报未启动或仍运行。

证据：[qprobe-warmup10-f4fb-terminal-cleanup-accepted-archive.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-warmup10-f4fb-terminal-cleanup-accepted-archive.json)

**A09 — 全question-token MLP128＋4-head attention warmup10**

全部有效question token→MLP128→单层4-head attention→补到244后flatten；BCE(mixed)+0.3 normalized std MSE。

身份：完成base494eafd18b004cb980121e46b188db44。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：不是完整Qwen block。epoch8权重没单独保存；两次失败/缓存复用在附录。

证据：[qprobe-token-binary-warmup10-fulltoken-results.md](/Users/shaoshitong/Documents/Search-G1/repo/reproduction/reports/qprobe-token-binary-warmup10-fulltoken-results.md) · [qprobe-token-binary-warmup10-fulltoken-results.json](/Users/shaoshitong/Documents/Search-G1/repo/reproduction/reports/qprobe-token-binary-warmup10-fulltoken-results.json)

**A10 — 完整原生Qwen最后block question-only warmup10**

冻结encoder layer27所有question hidden→完整复制原生layer35→masked flatten244→Linear2；BCE mixed+0.3 normalized std MSE。

身份：warmupbase096c8297cba74f8089063e763686671e；Head SHA9b32f40e…。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：这是同一训练题拟合，不是泛化；目标文件名有warmup20，不得覆盖实际10epoch/200计数。

证据：[qprobe-native-qwen-warmup10-096c-training-scalars.json](/Users/shaoshitong/Documents/Search-G1/repo/reproduction/reports/qprobe-native-qwen-warmup10-096c-training-scalars.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json)

**A11 — 原生Qwen question-only 正式RL**

Top512；每outer head2epoch／recent8 replay、指数年龄权；只导入独立探针，不改actor初始官方权重。

身份：run509c391b7e5c4174bfb55f724f41a3d6；execution60fca3cc2edd4dedb3773230b91f34ea。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：多个启动修复只复用同一已完成warmup，不能每次算新10epoch实验。

证据：[qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) · [qprobe-native-human-stop-new512-diagnostic-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-human-stop-new512-diagnostic-20261016.json)

**A12 — Head200 CPU复现＋全新512题G5泛化诊断**

独立复现旧Head200训练集表现、比对formal初始权重、同题G5标签审计；新512题不与预热/正式候选重叠；不训练actor/head。

身份：新512base16034e59031e4f21a323ebb2556bef7f。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：分类目标是G5是否混合，不是单轨迹QA EM；std输出全0的诊断亦保留。

证据：[qprobe-native-new512-cpu-weight-compare-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-new512-cpu-weight-compare-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json)

**A13 — A800 question＋Prefix5 单轨迹BCE，balanced warmup20**

原生Qwen block＋flattenLinear1；每条轨迹自己最终EM；batch256强制128对128错有放回，lr3e-4；20epoch目标2000Adam。

身份：baseaa4a9ebd63ac4683985bf74a748dcfb2。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：文件旧名prefix20，但实际policy prefix是5；训练分类指标不是QA EM。

证据：[qprobe-prefix-correctness-balanced-warmup20-human-intent-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-prefix-correctness-balanced-warmup20-human-intent-20261016.json) · [qprobe-prefix-correctness-balanced20-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-prefix-correctness-balanced20-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json)

### Gi会话：三block与早期warmup

| ID | 实验／阶段 | 实际进度 | 结果／异常 |
|---|---|---|---|
| H01 | H100 三个原生Qwen block，σ3 warmup2 | 真实warmup完成，仅保存最终统计，没有逐mini曲线。 | 总loss1.23408，对照常数1.23985；CE3 .90922，std RMSE .23872，改善很小。 |
| H02 | H100 三block σ3 正式RL | 该阶段完成首5轮及dev5；3399/3400均留有故障终态。 | 固定512 dev5 EM26.3671875%（135/512）。 |
| H03 | H100 三block mixed-BCE warmup10 | 10epoch200HeadAdam完成，teacher mixed1893/5120。 | 最终BCE约.5146、总loss.5267、训练accuracy77.13%、Top448混合92.41%。 |
| H04 | H100 三block mixed-BCE 正式RL | 停止前账本口径不齐：18轮完整耦合已严证、19轮console、actor更新达20而probe19；不计为20完整耦合outer。 | 探针更新普通轮平均约6分29秒；第14轮外层523.5秒。 |
| H05 | Question-only vs Question＋Prefix5 原拟议AB | 随后人工改为Question＋10/20两臂；未发现该AB的完整GPU训练产物。 | 不能补造两臂准确率。 |

**H01 — H100 三个原生Qwen block，σ3 warmup2**

question全token表示；3个完整Qwen decoder block；CE3+1 normalized σ MSE+0.1 CE6；5120×5；2epoch40HeadAdam。

身份：jobs3399/3400；后者runc025948ebc0947f69d86af84871a9237。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

证据：[qprobe-state-pre-binary-normalization-20261015.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe-state-pre-binary-normalization-20261015.json) · [qprobe-stdloss3399-terminal-archive-v1.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe-stdloss3399-terminal-archive-v1.json)

**H02 — H100 三block σ3 正式RL**

Top448＋随机64；actor8H100，probe单GPU7更新；frozen官方question编码。

身份：3396/3397等初始启动→3399→3400。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：初始六分类计划后来被σ3替代；未把仅计划的配置冒称独立完成。

证据：[qprobe-state-pre-binary-normalization-20261015.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe-state-pre-binary-normalization-20261015.json) · [qprobe3400-terminal-and-warmup10-fresh-submit-v2.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe3400-terminal-and-warmup10-fresh-submit-v2.json)

**H03 — H100 三block mixed-BCE warmup10**

BCE(mixed)+0.1 normalized sample-variance MSE；注意辅助项为variance，不是A800的0.3 std；5120固定题。

身份：job3402；runf77c3245dcfe49a88a5a99b1dec161e6。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

证据：[qprobe3402-warmup10-training-curve-summary-v1.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe3402-warmup10-training-curve-summary-v1.json)

**H04 — H100 三block mixed-BCE 正式RL**

actor八卡，probe GPU7单卡FP32/micro4；replay每outer2epoch；第14轮96HeadAdam，6144串行micro反传。

身份：job3402；runf77c3245dcfe49a88a5a99b1dec161e6。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：更换前缀设置后人工停止；未把摘要的19/20口径混为完成轮数。

证据：[qprobe-h1008-transformer3-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe-h1008-transformer3-state.json) · [qprobe3402-prefix5-stop-summary-20261015T174356636271Z.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe3402-prefix5-stop-summary-20261015T174356636271Z.json) · [REPORT.md](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/h100-qprobe-vs-mixed-latency-20261016/REPORT.md)

**H05 — Question-only vs Question＋Prefix5 原拟议AB**

两臂比较问题表征与问题＋5生成token；原拟10epoch。

身份：见下方实际证据；未虚构run identity。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

证据：[qprobe3402-prefix5-stop-summary-20261015T174356636271Z.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe3402-prefix5-stop-summary-20261015T174356636271Z.json) · [01a0fcbb-863f-7450-926d-db537f648265-chronology.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbb-863f-7450-926d-db537f648265-chronology.json)

### Gi会话：前缀长度、在线数据及联合监督warmup

| ID | 实验／阶段 | 实际进度 | 结果／异常 |
|---|---|---|---|
| P01 | Question＋Prefix10 balanced warmup20 | 20epoch／每rank2000HeadAdam；actor0；exit0。 | epoch20原始train accuracy96.2148%；原始test balanced accuracy53.6135%；test BCE2.3099。 |
| P02 | Question＋Prefix20 balanced warmup20 | 20epoch／每rank2000HeadAdam；actor0；exit0。 | epoch20原始train accuracy95.6797%；原始test balanced accuracy54.0187%；test BCE2.8186。 |
| P03 | Prefix-only10 balanced warmup20 | 20epoch／每rank2000HeadAdam；actor0；exit0。 | epoch20原始test balanced accuracy54.2696%，AUROC0.5997。 |
| P04 | Prefix-only20 balanced warmup20 | 20epoch／每rank2000HeadAdam；actor0；exit0。 | epoch20原始test balanced accuracy53.0330%，AUROC0.5833。 |
| P05 | Prefix-only40 balanced warmup20 | 20epoch／每rank2000HeadAdam；actor0；exit0。 | epoch20原始test balanced accuracy52.4446%，AUROC0.5786。 |
| P06 | Prefix-only80 balanced warmup20 | 20epoch／每rank2000HeadAdam；actor0；exit0。 | epoch20原始test balanced accuracy56.7286%，AUROC0.7042。 |
| P07 | Prefix-only20 在线换题 warmup1000 | 账本停在71epoch／7100Adam每rank；第70轮已有独立不可变CP与全SHA核验；不是1000轮完成。 | epoch70派生G5 mixed test balanced accuracy58.5987%、普通accuracy59.5703%；epoch68权重被导出给Co。 |
| P08 | Question＋Prefix20 replay70 | 70/70完成；前6轮只有test，后64轮有配对train/test；14100累计Adam／rank；exit0及CP验收通过。 | 最终train mixed balanced73.9199%、test60.9107%；最好test64.9410%在iteration30。 |
| P09 | Question＋Prefix20 joint BCE100 | 100/100epoch，每rank10000HeadAdam，actor0；10/20/…/100保存；最终验收通过。 | 最好观测test mixed balanced62.9619%在epoch45，但无该权重；最好实际保留CP是epoch100，59.8759%。 |

**P01 — Question＋Prefix10 balanced warmup20**

question＋本轨迹前10个policy token；冻结encoder27→原生block35＋flattenLinear1；单轨迹BCE，exact1:1采样；每臂4H100/global256。

身份：job3405；base94cc1d4。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：raw与prior-corrected指标分开，不报告成QA EM。

证据：[FINAL.json](/Users/shaoshitong/Documents/Search-G1/qprobe-prefix10-20-h1008-20261016/results/prefix10-20-h1008-20261016/FINAL.json)

**P02 — Question＋Prefix20 balanced warmup20**

question＋本轨迹前20个policy token；冻结encoder27→原生block35＋flattenLinear1；单轨迹BCE，exact1:1采样；每臂4H100/global256。

身份：job3405；base94cc1d4。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：raw与prior-corrected指标分开，不报告成QA EM。

证据：[FINAL.json](/Users/shaoshitong/Documents/Search-G1/qprobe-prefix10-20-h1008-20261016/results/prefix10-20-h1008-20261016/FINAL.json)

**P03 — Prefix-only10 balanced warmup20**

encoder仍看到完整question上下文，但probe只收到前10个生成位置hidden；原生block＋flattenLinear1；每臂2H100/global256；单轨迹BCE。

身份：job3406；base58d687c。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：不能把prefix-only说成无问题信息；其hidden已条件于question。

证据：[FINAL.json](/Users/shaoshitong/Documents/Search-G1/qprobe-prefix-only-four-h1008-20261016/results/prefix-only-four-h1008-20261016/FINAL.json)

**P04 — Prefix-only20 balanced warmup20**

encoder仍看到完整question上下文，但probe只收到前20个生成位置hidden；原生block＋flattenLinear1；每臂2H100/global256；单轨迹BCE。

身份：job3406；base58d687c。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：不能把prefix-only说成无问题信息；其hidden已条件于question。

证据：[FINAL.json](/Users/shaoshitong/Documents/Search-G1/qprobe-prefix-only-four-h1008-20261016/results/prefix-only-four-h1008-20261016/FINAL.json)

**P05 — Prefix-only40 balanced warmup20**

encoder仍看到完整question上下文，但probe只收到前40个生成位置hidden；原生block＋flattenLinear1；每臂2H100/global256；单轨迹BCE。

身份：job3406；base58d687c。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：不能把prefix-only说成无问题信息；其hidden已条件于question。

证据：[FINAL.json](/Users/shaoshitong/Documents/Search-G1/qprobe-prefix-only-four-h1008-20261016/results/prefix-only-four-h1008-20261016/FINAL.json)

**P06 — Prefix-only80 balanced warmup20**

encoder仍看到完整question上下文，但probe只收到前80个生成位置hidden；原生block＋flattenLinear1；每臂2H100/global256；单轨迹BCE。

身份：job3406；base58d687c。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：不能把prefix-only说成无问题信息；其hidden已条件于question。

证据：[FINAL.json](/Users/shaoshitong/Documents/Search-G1/qprobe-prefix-only-four-h1008-20261016/results/prefix-only-four-h1008-20261016/FINAL.json)

**P07 — Prefix-only20 在线换题 warmup1000**

冻结actor，每epoch5120×5新真实轨迹、仅训probe；全pool轮转；实际global256/local32、100Adam/epoch。LR从3e-4在CP7后改3e-5。

身份：run8a6b24dc19664cb3a3fb97101f21c54e；jobs3407→3408→3409。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：实际batch不是原计划25600；CP7实际LR3e-4→3e-5，不是3e-5→3e-6。70与停训时71口径分开。

证据：[prefix20-online1000-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix20-online1000-state.json) · [MIXED-G5-THROUGH70-SUMMARY.json](/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016/results/prefix20-online1000-lr3e5/MIXED-G5-THROUGH70-SUMMARY.json)

**P08 — Question＋Prefix20 replay70**

从旧71轮Head/Adam状态转入question＋20表示，lr3e-6，回放70份已有G5数据；每iteration100HeadAdam。

身份：run80cb2d8592304bc197614be9b04911c9；jobs3410→3411。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：这不是fresh actor RL，也不是70个新outer。

证据：[FINAL-ACCEPTANCE-SUMMARY.json](/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016/results/question20-replay70/FINAL-ACCEPTANCE-SUMMARY.json) · [BALANCED-ACCURACY-SUMMARY.json](/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016/results/question20-replay70/BALANCED-ACCURACY-SUMMARY.json) · [prefix20-question-replay70-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix20-question-replay70-state.json)

**P09 — Question＋Prefix20 joint BCE100**

fresh probe；逐轨迹EM BCE＋同题G5 mixed BCE，权重1:1；h=1−∏p−∏(1−p)；lr3e-5 cosine→3e-6。

身份：job3412；runcee20bcd24194b79a9a8a88809bd5886；HeadSHA8df6bd44…。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：epoch68 HF权重来自P07，不是本实验；这个区别会影响后续初始化对比。

证据：[FINAL-ACCEPTANCE.json](/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016/results/question20-groupjoint100/FINAL-ACCEPTANCE.json) · [FINAL-METRICS-100.json](/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016/results/question20-groupjoint100/FINAL-METRICS-100.json)

### 后续正式选题RL

| ID | 实验／阶段 | 实际进度 | 结果／异常 |
|---|---|---|---|
| L01 | Question＋Prefix20 冻结encoder Top512 RL | 21完整outer后人工停止；无偏随机候选未进入head训练引发选择偏差担忧。 | 最后selected trainEM7.34375%；固定512 iter5 EM24.21875%（124）。 |
| L02 | 当前actor-hidden Prefix20＋period20 RL | 本次本地实际状态记录18完整outer；head更新账本[0, 0, 0, 0, 0, 0, 0, 0]；早前远端冻结审计严证8。 | 固定512 iter5 EM23.6328125%（121）。 |
| L03 | Prefix Top1024→真实diversity Top512 RL | 本次本地原始reward回执已到39；早前冻结远端审计严证32。 | 固定512 iter5 EM24.609375%（126）；训练EM因真实diversity二次筛选而不可当测试EM。 |
| L04 | HPC3 独立原生Qwen QDP warmup10／每outer刷新原方案 | 新5120特征准备/初始warmup有执行；旧正式job723510零actor更新时被后续period25设置替代。 | 不能报成一个已完成200轮的TOP-D蒸馏实验。 |
| L05 | Periodic25 原生Qwen QDP 初始及周期warmup | 边界0/25/50完成；每次400HeadAdam，累计1200。 | 后续新边界是同一RL的refit阶段，不是独立fresh模型实验。 |
| L06 | Periodic25 原生Qwen QDP 正式RL | 本地STATE最近记录72完整outer；此前原始远端冻结读回68。 | 固定512 iter5 EM25.390625%（130）；iter50 36.9140625%（189）。 |
| L07 | HPC3 Prefix-only20＋全对组虚拟0 RL | 当前727586仅有提交凭证，完整outer0；此前727414真实payload失败，不能算训练已完成。 | 没有该新run可报告的固定5轮EM；旧dev0不是训练后成绩。 |

**L01 — Question＋Prefix20 冻结encoder Top512 RL**

P09实际保留CP100初始化head；officialactor fresh；5120×5前缀→预测Top512→续写原前缀；选中2560轨迹jointBCE训head1epoch/lr1e-6，每outer10HeadAdam。

身份：run5cba22ca409a4571a1e6b87d89c86b09；jobs3413/3414/3415/3419/3420恢复链。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：输入是独立冻结官方encoder hidden，不能说成当时actor hidden；QA曾有Triton布局编译失败。

证据：[h100-8-current.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/session-refresh-20261017-pinned/h100-8-current.json) · [prefix-selection200-resumed6-10-runtime-summary.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix-selection200-resumed6-10-runtime-summary.json)

**L02 — 当前actor-hidden Prefix20＋period20 RL**

当前actor layer27捕获question＋20 hidden，采样21丢第21以对齐；5120候选Top512；每20outer才用全5120×5训probe1epoch（100HeadAdam），其余冻结。

身份：runa008d7ce983347bbb6e81051b25a7c4e；3423→3424→3425。所属会话：`01a0fcbb-863f-7450-926d-db537f648265`。

注意：前两次分别存到1/2轮后storage校验失败，恢复属于同run，不是三个科学实验。

证据：[search-g1-actorhidden-period20-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/search-g1-actorhidden-period20-state.json) · [actorhidden-period20-first-five-actual-facts.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/actorhidden-period20-first-five-actual-facts.json) · [h100-8-current.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/session-refresh-20261017-pinned/h100-8-current.json)

**L03 — Prefix Top1024→真实diversity Top512 RL**

P09 CP100 head；先预测Top1024并完整续写，再按实际hard-EM population variance挑512入训actor；head用全部1024×5标签，20更新/outer。

身份：runf3ee1435e5014ba1969387c91878893a；job3422；之前3421存到5。所属会话：`01a0fcbc-0b46-7502-8b66-d8b21a0e032d`。

注意：与Gi单纯预测Top512不同；已发生恢复/评测Triton故障，保留原失败证据。

证据：[CURRENT_CONFIG.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/search-g1-prefix1024-diversity512/CURRENT_CONFIG.json) · [TRAINING_REWARD_HISTORY.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/search-g1-prefix1024-diversity512/TRAINING_REWARD_HISTORY.json) · [STATUS.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/search-g1-prefix1024-diversity512/STATUS.json) · [TERMINAL3421_SMALL.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/search-g1-prefix1024-diversity512/TERMINAL3421_SMALL.json)

**L04 — HPC3 独立原生Qwen QDP warmup10／每outer刷新原方案**

独立head/Adam；question-only原生Qwen最后block，BCE mixed+std；原计划warmup10后每outer2epoch/replay8，Top512。

身份：723434准备；723510旧正式取消；base native-thread5-20261015。所属会话：`01a0fcbe-15f8-7600-ae84-9487d924560a`。

注意：这是TOP-D会话承载的Search-G1 QDP，名称不代表使用TOP-D loss；原方案与正式period25分开。

证据：[deployment-plan.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/deployment-plan.json) · [hpc3-native-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/hpc3-native-submission.json) · [periodic25-change.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/periodic25-change.json)

**L05 — Periodic25 原生Qwen QDP 初始及周期warmup**

0/25/50/75/100/125/150/175边界用当前student5120×5重新标注，head20epoch；不重置block/readout/Adam；200只存评测不refit。

身份：run1cd20aedc5e94511be8ea8b7db76933c；currentjob725286。所属会话：`01a0fcbe-15f8-7600-ae84-9487d924560a`。

证据：[periodic25-change.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/periodic25-change.json) · [refit25-latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r16/refit25-latest-sample.json) · [STATE.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/STATE.json)

**L06 — Periodic25 原生Qwen QDP 正式RL**

Top512/no random；fourGPU；question-only head只按period25刷新，actor每outer20Adam。

身份：canonical1–5来自r13，6–10来自r15，11+来自r16；currentjob725286。所属会话：`01a0fcbe-15f8-7600-ae84-9487d924560a`。

注意：r15曾完成但未保存的outer11被放弃，从CP10重做；不累计两次11。17个部署目录不是17种算法。

证据：[STATE.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/STATE.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r16/latest-sample.json) · [mou176-current.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/session-refresh-20261017-pinned/mou176-current.json)

**L07 — HPC3 Prefix-only20＋全对组虚拟0 RL**

导入P07 epoch68，而非P09 CP100；head只接20生成位置；逐轨迹BCE，不含group joint BCE；selected512×5每outer训head1epoch/lr1e-6；只EM全对组加虚拟0。

身份：727586；base qprobe-teacher-warmup200-prefix20-joint4-recovery727414-20261017。所属会话：`01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef`。

注意：名称含joint4，但实际不是逐轨迹BCE＋groupBCE联合监督；虚拟0是另一算法改变。

证据：[handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/handoff.json) · [727414-actual-payload-failure-and-zero-write-evidence-1792170376.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/727414-actual-payload-failure-and-zero-write-evidence-1792170376.json)

### 扩大数据、EMA与A800新方案

| ID | 实验／阶段 | 实际进度 | 结果／异常 |
|---|---|---|---|
| N01 | Prefix5 scale20 teacher数据／特征构建 | 200chunk完成；最终数据CPU审计和HF追加验收通过。 | 此数据阶段不是200outer RL，也不是200epoch probe拟合。 |
| N02 | Question＋Prefix20 EMA joint200 原方案 | CPU核心已实现和发布；合并数据只有新prefix5，后由人工改为全部Prefix5；无该20-token独立GPU训练run。 | 无完成epoch／测试准确率。 |
| N03 | Question＋Prefix5 EMA0.99 joint200 全量warmup | 4完整epoch，epoch5活动；global256改1024时从CP2完整状态恢复；actor更新0。 | epoch4 EMA：train mixed balanced约84.75%，test约56.57%；单轨迹train/test accuracy约83.50/77.03%；泛化gap明显。 |
| N04 | A800 actor-hidden Prefix5＋period20 新正式RL | 截至本次本地记录，新v2源码已stage、CPU prepare待完成；completeouter0；旧EMA warmup尚未停止。 | 无新actor训练／dev5成绩。 |

**N01 — Prefix5 scale20 teacher数据／特征构建**

frozen官方teacher；20×5120题／102400题／512000条轨迹；第一份旧5120复用，额外97280题；保存prefix5与各自最终EM，不更新actor/head。

身份：datasetbuildbase0850db2977bd4801b9ced300b0ab387f；run62407ad0ed404586998439ca1f55b902。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：前期c6缓存scope adapter失败与修复单列；历史prefix20数据追加后统一裁到5。

证据：[qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) · [qprobe-prefix-scale20-final-audit-1792141000881306000.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-prefix-scale20-final-audit-1792141000881306000.json)

**N02 — Question＋Prefix20 EMA joint200 原方案**

question＋20；轨迹BCE＋groupBCE 1:1；EMA0.99评估；目标200epoch。

身份：current_question20_ema_joint_warmup。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：不能用后续5-token成绩填到20-token方案下。

证据：[qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json)

**N03 — Question＋Prefix5 EMA0.99 joint200 全量warmup**

合并去重后445440个G5事件组／2227200轨迹；全部prefix5；nativeblock＋flattenLinear1；轨迹BCE＋mixed-groupBCE权重1:1，EMA0.99；每epoch全数据一遍。

身份：run8af926a49630428097b94027864da68b；executionbcfa1de0a90a4fcda70263a75153fcee。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：445440是采样事件组，不等于445440个唯一问题；离线probe分类指标不进QA榜。

证据：[qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) · [question5-global1024-resume-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/question5-global1024-resume-deployment-20261016.json) · [question5-global1024-epoch4-training-window-publication-20261017.json](/Users/shaoshitong/Documents/Search-G1/preparation/question5-global1024-epoch4-training-window-publication-20261017.json)

**N04 — A800 actor-hidden Prefix5＋period20 新正式RL**

参考Gi actorhidden20；改question＋5；四卡5120→Top512；导入切换时实际已完成EMA head、fresh在线Adam与officialactor。

身份：原准备run8e6ced4e…因rotary buffer loader假设失败；新v2 run411c44aad7ba4e8c90f5fbec9e5ecbaa。所属会话：`01a0fb1b-d3ff-74b3-aa13-49ca57871f92`。

注意：实际current_authoritative以新v2为准；只是prepare尝试，不把它叫新RL已经跑起来。

证据：[a800-actorhidden-prefix5-human-intent-20261017.json](/Users/shaoshitong/Documents/Search-G1/preparation/a800-actorhidden-prefix5-human-intent-20261017.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json)

## 失败、恢复、停止及准备尝试附录

这张表保留失败和未完成执行。多个部署可能共用一个run；没有明确退出原因的地方保留“不确定／见原终态”，不根据超参数猜测失败原因。部分辅助job只做CPU准备、清理或归档。

### G1会话执行尝试

| 执行标识 | 状态 | 实际情况／故障 | 证据 |
|---|---|---|---|
| Search-G1-a800-fresh200-27bab88929e048b4869f3889b99d8e18 | 失败 | 首批teacher QPROBE_POLICY_SNAPSHOT_CHANGED_DURING_GENERATION；不能据此声称actor漂移；formal0 | [qprobe-teacher-warmup-a800-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-deployment.json) |
| Search-G1-a800-fresh200-c52b44701a0544bd9e219122e53d3c77 | 启动失败／后续替代 | c52b preclaim/空闲GPU/端口门禁，保留原始原因未篡改；formal0 | [qprobe-teacher-warmup-a800-snapshot-repair-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-snapshot-repair-deployment.json) |
| Search-G1-a800-fresh200-b9860d6d2a5e43a59d0b25f99fa13bce | 人工换micro配置 | micro2已写26Adam／rank；其中仅1完整outer；1＋部分6Adam | [qprobe-a800-flex-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-a800-flex-deployment.json) |
| Search-G1-a800-fresh200-d85691245df24b628b309d50b7e5472a | 故障退出 | 实际FAILED_34FF；多项failure_diagnostics保存，细分原因见终态原文件；4完整outer | [qprobe-a800-micro4-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-a800-micro4-deployment.json) |
| Search-G1-a800-fresh200-a365743b4b214cbb9693f1b53b64d811 | 故障退出 | 5轮后native calibration identity绑定失败；5完整outer | [qprobe-a800-async-top512-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-a800-async-top512-deployment.json) |
| Search-G1-a800-fresh200-0065aa3936ba42e19ac3ff50f5ac0240 | 修复候选被替代 | 0065 restore候选不等于新fresh模型实验；以后续fullouter恢复为准；父CP5 | [qprobe-a800-async-top512-restore5-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-a800-async-top512-restore5-deployment.json) |
| Search-G1-a800-fresh200-af113a29fa9a48a7b0c1658ce4c0396c | 同run恢复／后人工换loss | 完整状态恢复CP5，后原MLP链达到38轮；累计38 | [qprobe-a800-async-fullouter-restore5-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-a800-async-fullouter-restore5-deployment.json) |
| Search-G1-a800-fresh200-b8eb3037d2704046952e66c91e38c81d | 启动失败 | c528 preclaim；无trainer／formal更新；0 | [qprobe-a800-sigma3-loss-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-a800-sigma3-loss-deployment.json) |
| Search-G1-a800-fresh200-f7fa3a82c8144c84b3ea300e287a07e8 | 启动失败 | 1573 teacher cache namespace mismatch；0 | [qprobe-a800-sigma3-fresh-start-fix-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-a800-sigma3-fresh-start-fix-deployment.json) |
| Search-G1-a800-fresh200-86cf721183014e729036924e8978a6b9 | 人工换warmup10 | f847 teacher-v4实际warmup与3轮正式已执行；≥3完整＋部分 | [qprobe-a800-sigma3-teacher-v4-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-a800-sigma3-teacher-v4-deployment.json) |
| Search-G1-a800-fresh200-4a017644c8054de38b05314879527303 | 人工换架构／清理通过 | f4fb：warmup200HeadAdam，正式2完整outer后停止；2outer/40Adam | [qprobe-a800-sigma3-warmup10-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-a800-sigma3-warmup10-deployment.json) |
| Search-G1-qprobe-token-binary-warmup10-5f004242920f45cdb5592d33c2a16bc2 | 打包失败 | 第一次源码打包失败，不能当Head已训练；Head0 | [qprobe-token-binary-warmup10-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-token-binary-warmup10-deployment.json) |
| Search-G1-qprobe-token-binary-warmup10-494eafd18b004cb980121e46b188db44 | 完成 | 前一a799 encoder完成但allocator-zero检查失败；494复用缓存freshHead完成10epoch；200HeadAdam | [qprobe-token-binary-headfit-resume-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-token-binary-headfit-resume-deployment.json) |
| Search-G1-qprobe-qwen-lastblock-warmup10-c83d7affba4f41e78d6171a7f6fcddf7 | 启动失败 | c83d native metadata JSON键绑定错误，Head未训练；Head0 | [qprobe-qwen-lastblock-warmup10-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-lastblock-warmup10-deployment.json) |
| Search-G1-qprobe-qwen-lastblock-warmup10-096c8297cba74f8089063e763686671e | 完成 | 096c修复后完整Qwenblock warmup10完成；200HeadAdam | [qprobe-qwen-json-binding-repair-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-json-binding-repair-deployment.json) |
| Search-G1-a800-fresh200-aa975d2af56b4698bbca33a6d7f32e65 | CPU prepare失败 | aa975，未启动actor/encoder/数据复制；原warmup保留；actor0 | [qprobe-qwen-native-formal-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-formal-deployment.json) |
| Search-G1-a800-fresh200-002afa9247cd4641acec523a930b8ea7 | CPU prepare失败 | 002a修复候选再次prepare失败；实际原因保存于failure-preserved；actor0 | [qprobe-qwen-native-formal-cpu-repair-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-formal-cpu-repair-deployment.json) |
| Search-G1-a800-fresh200-92bb58c18ae74ca9bc59fe1ba5dc02b6 | 启动失败 | 92bb完整bank已合格，但prelaunch/入口失败；缓存不重做；actor0 | [qprobe-qwen-native-formal-cpu-repair-v2-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-formal-cpu-repair-v2-deployment.json) |
| Search-G1-a800-fresh200-ff1196a974114d7abe9b1be0c764b721 | 启动失败 | ff1196 driver/process及bank契约问题；保留失败并过渡v2；actor0 | [qprobe-qwen-native-driver-process-fix-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-driver-process-fix-deployment.json) |
| Search-G1-a800-fresh200-4fd508e35bc84658ba892a97248f8a7f | 启动失败 | 4fd508正式启动失败，exact argv终态诊断保留；actor0 | [qprobe-qwen-native-driver-process-fix-v2-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-driver-process-fix-v2-deployment.json) |
| Search-G1-a800-fresh200-e9c4c584107b4624bcc9787aaf24b7a2 | 人工停止 | 成功开始原生formal；1完整outer＋10部分Adam，非30outer；1outer/30Adam总 | [qprobe-qwen-native-selection-metrics-fix-deployment.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-selection-metrics-fix-deployment.json) |
| Search-G1-qprobe-prefix-correctness-warmup20-c3aefa891a304aae8f8f40c340fed2f8 | 旧采样协议未拟合 | unbalanced fit未启动，被balanced sampler替代；encoder产物复用；Head0 | [qprobe-prefix-correctness-warmup20-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-prefix-correctness-warmup20-deployment-20261016.json) |
| Search-G1-qprobe-prefix-correctness-balanced20-aa4a9ebd63ac4683985bf74a748dcfb2 | 人工停止／未跑满 | 6epoch＋91次部分更新；保留partial state；691HeadAdam | [qprobe-prefix-correctness-balanced20-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-prefix-correctness-balanced20-deployment-20261016.json) |
| qprobe-token-binary-a799-failed-encoder-complete-archive.json | encoder后失败 | 全token特征5120完成；allocator-zero断言退出，Head0；494后续复用 | [qprobe-token-binary-a799-failed-encoder-complete-archive.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-token-binary-a799-failed-encoder-complete-archive.json) |
| qprobe-prefix-scale20-source-publication-20261016.json | 数据链CPU修复 | 原c6缓存scope adapter失败；修复后0850新执行完成scale20，不重复计teacher数据 | [qprobe-prefix-scale20-source-publication-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-prefix-scale20-source-publication-20261016.json) |
| question5-global1024-resume-deployment-20261016.json | 同run恢复 | global256→1024，从CP2载入Head/Adam/EMA/RNG；非重新200epoch | [question5-global1024-resume-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/question5-global1024-resume-deployment-20261016.json) |
| a800-actorhidden-prefix5-human-intent-20261017.json | 准备失败后v2 | 8e6c原prepare loader假设错误；v2 411c准备中，未训练actor | [a800-actorhidden-prefix5-human-intent-20261017.json](/Users/shaoshitong/Documents/Search-G1/preparation/a800-actorhidden-prefix5-human-intent-20261017.json) |
| search-g1-prefix-selection200-a800-prefix5-8e6ced4e240a46329deee42df8819228 | CPU prepare失败 | 原8e6c：Transformers4.46.3 rotary buffers契约假设错误，训练0；v2重新准备 | [a800-actorhidden-prefix5-deployment-20261017.json](/Users/shaoshitong/Documents/Search-G1/preparation/a800-actorhidden-prefix5-deployment-20261017.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| search-g1-prefix-selection200-a800-prefix5-411c44aad7ba4e8c90f5fbec9e5ecbaa | v2准备中 | 新411c，exact源码staged；本次记录尚无actor完整outer | [a800-actorhidden-prefix5-v2-deployment-20261017.json](/Users/shaoshitong/Documents/Search-G1/preparation/a800-actorhidden-prefix5-v2-deployment-20261017.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| search-g1-question5-reference5-20261016-47500d559a9e4ee0b963a62607fd0a2a | 参考缓存准备／对齐 | 4750 H100 reference5特征/格式对齐；没有在此handle发现独立actor RL结果 | [h100-question5-reference5-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/h100-question5-reference5-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-hf-scale20-prefix5-20261016-8456acbf17144698be5c9a819908f4f4 | 资产导出／上传 | scale20 prefix5追加HF；不算新probe fit | [qprobe-hf-scale20-prefix5-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-hf-scale20-prefix5-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-hf-trajectories-20261016-7efb625ac5d040e1ab99f0bd00da520c | 资产导出／上传 | 旧teacher真实轨迹打包与HF发布；不算新probe fit | [qprobe-hf-trajectory-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-hf-trajectory-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-qprobe-native-control-c93bafa5903b4d5f8a101aed83e26c65 | 诊断完成；中间入口故障保留 | 原warmupHead200 CPU复现及tensor绑定；没有训练更新 | [qprobe-native-head200-control-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-head200-control-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-a800-fresh200-16034e59031e4f21a323ebb2556bef7f | 诊断完成 | 新512×5真实rollout，旧Head200 mixed accuracy58.7891% | [qprobe-native-new512-diagnostic-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-new512-diagnostic-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-a800-fresh200-0850db2977bd4801b9ced300b0ab387f | 数据构建完成 | 0850 scale20数据构建，不是RL | [qprobe-prefix-scale20-dataset-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-prefix-scale20-dataset-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-a800-fresh200-c6b162367f21421ca70ad58ae890fec9 | 数据准备失败 | 原c6 scope adapter问题，0新Head/actor更新；0850修复独立执行 | [qprobe-prefix-scale20-failed-cache-scope-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-prefix-scale20-failed-cache-scope-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-question5-ema-joint200-20261016-4347d238b1d64b3f95ed22fb59d1f00f | 训练入口修复尝试 | 4347直接复用已封存缓存；随后procguard新handle8af接管，原失败或退出状态见index，不猜数值根因 | [question5-cached-sourcefix-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/question5-cached-sourcefix-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-question5-ema-joint200-20261016-8af926a49630428097b94027864da68b | 实际训练执行 | 8af初始execution091f；完成epoch1/2，后按批量变更要求完整恢复 | [question5-cached-sourcefix-procguard-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/question5-cached-sourcefix-procguard-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-question5-ema-joint200-20261016-db41344371894db7a011b26adf0e7218 | 准备handle；无完整epoch凭证 | db413最初EMA方案准备入口；后由slim/direct/cache入口替代，不冒称训练成功 | [question5-ema-joint200-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/question5-ema-joint200-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-question5-ema-joint200-20261016-f4aab3c7ec084f908ecce3f49c6b552b | 准备协议被替代 | f4aa slim路径；等待/数据路由被HF-direct取代；无该handle完成epoch凭证 | [question5-ema-joint200-slim-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/question5-ema-joint200-slim-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-question5-ema-joint200-20261016-d7dd61bb36454e1e80c2306b5fb7d23a | 特征缓存完成／训练前修复 | d7dd复用/生成全量prefix5缓存；原归档/源码核验修复后直接复用缓存，不重新采样 | [question5-hf-direct-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/question5-hf-direct-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-question5-ema-joint200-20261016-8af926a49630428097b94027864da68b-resume-141a9ada1d464f0d9925d8397a194944 | 技术恢复尝试 | 141a pidfd兼容入口；后由global1024同run执行bcfa替代，不累计成独立epoch实验 | [question5-largebatch-pidfd-resume-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/question5-largebatch-pidfd-resume-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |
| Search-G1-question5-ema-joint200-20261016-8af926a49630428097b94027864da68b-resume-203d3cc051de4eae92856131cd638b6a | 技术恢复尝试 | 203d largebatch入口；pidfd Python封装兼容性阻塞，真实stop失败before-signal已保留；非训练数值故障 | [question5-largebatch-resume-deployment-20261016.json](/Users/shaoshitong/Documents/Search-G1/preparation/question5-largebatch-resume-deployment-20261016.json) · [qprobe-teacher-warmup-a800-progress-handoff.json](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-teacher-warmup-a800-progress-handoff.json) |

### Gi会话执行尝试

| 执行标识 | 状态 | 实际情况／故障 | 证据 |
|---|---|---|---|
| H100 3396 | 完成／恢复／故障，见说明 | 初始三block重连核查只返回Invalid job id；原native终态缺失，不能确定数值失败原因 | [qprobe3-job3396-reconnect-terminal-v1.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe3-job3396-reconnect-terminal-v1.json) |
| H100 3397 | 完成／恢复／故障，见说明 | 旧启动失败，原terminal保留 | [qprobe-stdloss-old-terminal-check-v4.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe-stdloss-old-terminal-check-v4.json) |
| H100 3399 | 完成／恢复／故障，见说明 | σ3三block warmup2链故障退出 | [qprobe-stdloss3399-terminal-archive-v1.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe-stdloss3399-terminal-archive-v1.json) |
| H100 3400 | 完成／恢复／故障，见说明 | 首5轮及dev5后失败；被warmup10 fresh方案替代 | [qprobe3400-terminal-and-warmup10-fresh-submit-v2.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe3400-terminal-and-warmup10-fresh-submit-v2.json) |
| H100 3401 | 完成／恢复／故障，见说明 | 新warmup10原尝试，trainer/owned cleanup异常，formal验收未完成 | [qprobe3401-binary-replacement-final-actual-v1.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe3401-binary-replacement-final-actual-v1.json) |
| H100 3402 | 完成／恢复／故障，见说明 | binary warmup10成功，后人工转prefix试验；actor/probe计数不齐 | [qprobe-h1008-transformer3-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qprobe-h1008-transformer3-state.json) |
| H100 3403 | 完成／恢复／故障，见说明 | 旧question-only／prefix5 AB计划被10/20方案替代，不能视作完成 | [prefix10-20-warmup20-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix10-20-warmup20-state.json) |
| H100 3404 | 完成／恢复／故障，见说明 | 10/20初始准备退出，后3405完成 | [prefix3404-terminal-resource-1792088260549484000.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix3404-terminal-resource-1792088260549484000.json) |
| H100 3405 | 完成／恢复／故障，见说明 | Question+10/20两臂20epoch完成 | [prefix10-20-warmup20-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix10-20-warmup20-state.json) |
| H100 3406 | 完成／恢复／故障，见说明 | Prefix-only10/20/40/80四臂20epoch完成 | [prefix20-online1000-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix20-online1000-state.json) |
| H100 3407 | 完成／恢复／故障，见说明 | online1000第2轮frozen rollout失败：GPU UUID被当整数physical_device_id解析，ValueError；不是Head loss故障 | [prefix20-online-terminal3407-20261015T194713083944Z.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix20-online-terminal3407-20261015T194713083944Z.json) |
| H100 3408 | 完成／恢复／故障，见说明 | 完成7epoch后人工LR降10倍，从CP7恢复 | [prefix20-online1000-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix20-online1000-state.json) |
| H100 3409 | 完成／恢复／故障，见说明 | 继承至71epoch后人工换question+20 replay协议 | [prefix20-online1000-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix20-online1000-state.json) |
| H100 3410 | 完成／恢复／故障，见说明 | replay最初6轮仅test，人工重启补配对train/test | [prefix20-question-replay70-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix20-question-replay70-state.json) |
| H100 3411 | 完成／恢复／故障，见说明 | 配对replay完成70/70，继承原6轮，不重复计 | [prefix20-question-replay70-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix20-question-replay70-state.json) |
| H100 3412 | 完成／恢复／故障，见说明 | jointBCE100完成验收 | [question20-groupjoint100-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/question20-groupjoint100-state.json) |
| H100 3413 | 完成／恢复／故障，见说明 | Prefix20正式初始执行，后恢复；不得与3414等合并称独立算法 | [search-g1-prefix-selection200-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/search-g1-prefix-selection200-state.json) · [01a0fcbb-863f-7450-926d-db537f648265-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbb-863f-7450-926d-db537f648265-history-hits.json) |
| H100 3414 | 完成／恢复／故障，见说明 | Prefix20早期恢复尝试；真实细分原因见保存的恢复记录 | [search-g1-prefix-selection200-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/search-g1-prefix-selection200-state.json) · [01a0fcbb-863f-7450-926d-db537f648265-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbb-863f-7450-926d-db537f648265-history-hits.json) |
| H100 3415 | 完成／恢复／故障，见说明 | Prefix20 resume2已提交；后续恢复到3420 | [prefix-selection200-resume2-submit-20261016T121739966237Z-actual.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix-selection200-resume2-submit-20261016T121739966237Z-actual.json) |
| H100 3419 | 完成／恢复／故障，见说明 | Prefix20后续恢复；当前链随后3420 | [search-g1-prefix-selection200-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/search-g1-prefix-selection200-state.json) · [01a0fcbb-863f-7450-926d-db537f648265-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbb-863f-7450-926d-db537f648265-history-hits.json) |
| H100 3420 | 完成／恢复／故障，见说明 | 继承CP，累计21完整outer，后人工停止 | [prefix-selection200-resumed6-10-runtime-summary.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix-selection200-resumed6-10-runtime-summary.json) |
| H100 3423 | 完成／恢复／故障，见说明 | actorhidden存到1轮；legacy每outer10head storage断言错误 | [search-g1-actorhidden-period20-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/search-g1-actorhidden-period20-state.json) |
| H100 3424 | 完成／恢复／故障，见说明 | actorhidden存到2轮；prune把导入history误认owned路径 | [search-g1-actorhidden-period20-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/search-g1-actorhidden-period20-state.json) |
| H100 3425 | 完成／恢复／故障，见说明 | 同run恢复继续；本次本地18完整outer | [search-g1-actorhidden-period20-state.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/search-g1-actorhidden-period20-state.json) |

### Tree会话执行尝试

| 执行标识 | 状态 | 实际情况／故障 | 证据 |
|---|---|---|---|
| H1003421 | 评测故障／从CP5恢复 | 首5训练完整；QA5 Triton mma-layout转换断言失败，checkpoint保留；不是actor第五轮失败 | [TERMINAL3421_SMALL.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/search-g1-prefix1024-diversity512/TERMINAL3421_SMALL.json) |
| H1003422 | 同run恢复后进行中 | 恢复链prefix1024-diversity512；本次本地原始reward回执39 | [CURRENT_CONFIG.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/search-g1-prefix1024-diversity512/CURRENT_CONFIG.json) · [STATUS.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/search-g1-prefix1024-diversity512/STATUS.json) · [TRAINING_REWARD_HISTORY.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/search-g1-prefix1024-diversity512/TRAINING_REWARD_HISTORY.json) |

### TOPD会话执行尝试

| 执行标识 | 状态 | 实际情况／故障 | 证据 |
|---|---|---|---|
| Periodic25 r0 / job723735 | 失败或被恢复替代 | GPU preflight失败；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016/formal-submission.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016/latest-sample.json) |
| Periodic25 r1 / job723870 | 失败或被恢复替代 | 模型初始化失败；详细诊断见model-initial-failure；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r1/formal-submission.json) · [model-initial-failure.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r1/model-initial-failure.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r1/latest-sample.json) |
| Periodic25 r2 / job723934 | 失败或被恢复替代 | retriever readiness退出；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r2/formal-submission.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r2/latest-sample.json) |
| Periodic25 r3 / job724056 | 失败或被恢复替代 | retriever／校准失败，附cleanup分区提交失败与最小资源重试；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r3/formal-submission.json) · [calibration-failure-fresh.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r3/calibration-failure-fresh.json) · [previous-retriever-failure.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r3/previous-retriever-failure.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r3/latest-sample.json) |
| Periodic25 r4 / job724192 | 失败或被恢复替代 | overlay差异／prebind失败；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r4/formal-submission.json) · [overlay-difference-and-prebind-failure.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r4/overlay-difference-and-prebind-failure.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r4/latest-sample.json) |
| Periodic25 r5 / job724245 | 失败或被恢复替代 | native process birth门禁失败；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r5/formal-submission.json) · [native-birth-failure.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r5/native-birth-failure.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r5/latest-sample.json) |
| Periodic25 r6 / job724332 | 失败或被恢复替代 | trainer启动失败；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r6/formal-submission.json) · [actual-startup-failure.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r6/actual-startup-failure.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r6/latest-sample.json) |
| Periodic25 r7 / job724376 | 失败或被恢复替代 | GPU采样BlockingIOError资源暂不可用；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r7/formal-submission.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r7/latest-sample.json) |
| Periodic25 r8 / job724404 | 失败或被恢复替代 | trainer启动失败；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r8/formal-submission.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r8/latest-sample.json) |
| Periodic25 r9 / job724424 | 失败或被恢复替代 | model coexistence门禁失败；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r9/formal-submission.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r9/latest-sample.json) |
| Periodic25 r10 / job724443 | 失败或被恢复替代 | trainer失败；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r10/formal-submission.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r10/latest-sample.json) |
| Periodic25 r11 / job724465 | 失败或被恢复替代 | trainer失败；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r11/formal-submission.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r11/latest-sample.json) |
| Periodic25 r12 / job724494 | 失败或被恢复替代 | KeyError selected_predicted_diversity，正式outer0；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r12/formal-submission.json) · [first-collection-failure.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r12/first-collection-failure.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r12/latest-sample.json) |
| Periodic25 r13 / job724530 | 失败或被恢复替代 | 完成1–5；后trainer故障，从CP5恢复；canonical1–5 | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r13/formal-submission.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r13/latest-sample.json) |
| Periodic25 r14 / job724776 | 失败或被恢复替代 | CP5恢复尝试未成功；之后转r15；完整outer0／见原receipt | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r14/formal-submission.json) · [pre-r15-terminal-locks.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r14/pre-r15-terminal-locks.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r14/latest-sample.json) |
| Periodic25 r15 / job724782 | 失败或被恢复替代 | 完成6–10可恢复；11虽完成未存；12前Ray线程池资源不足，cleanup又ENOSYS38；canonical6–10；11舍弃 | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r15/formal-submission.json) · [outer12-failure-diagnostic.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r15/outer12-failure-diagnostic.json) · [terminal-failure-and-recovery-scope.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r15/terminal-failure-and-recovery-scope.json) · [terminal-resource-and-locks.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r15/terminal-resource-and-locks.json) · [terminal-failure-and-recovery-scope.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r15/terminal-failure-and-recovery-scope.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r15/latest-sample.json) |
| Periodic25 r16 / job725286 | 恢复后进行中 | 从CP10恢复，重做11；当前实际推进中；canonical11以后；本次本地72 | [formal-submission.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r16/formal-submission.json) · [latest-sample.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-periodic25-20261016-r16/latest-sample.json) |
| 723411 | 准备／辅助／被替代 | 独立native方案的准备尝试；actual native前置准备 | [periodic25-change.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/periodic25-change.json) · [01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json) |
| 723434 | 准备／辅助／被替代 | 独立native准备job；新5120特征与warmup | [periodic25-change.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/periodic25-change.json) · [01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json) |
| 723510 | 准备／辅助／被替代 | 旧每outer2epoch正式方案，0actor时被人工改period25取消 | [periodic25-change.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/periodic25-change.json) · [01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json) |
| 723619 | 准备／辅助／被替代 | 初始warmup10→20准备延续 | [periodic25-change.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/periodic25-change.json) · [01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json) |
| 723670 | 准备／辅助／被替代 | period25 preparation提交 | [periodic25-change.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/periodic25-change.json) · [01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json) |
| 724148 | 准备／辅助／被替代 | r3清理debug分区拒绝后改acd_u最小资源 | [periodic25-change.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/periodic25-change.json) · [01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json) |
| 724250 | 准备／辅助／被替代 | 辅助收尾／检查作业，不是独立RL | [periodic25-change.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/periodic25-change.json) · [01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json) |
| 724403 | 准备／辅助／被替代 | r7失败后的cleanup | [periodic25-change.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/periodic25-change.json) · [01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json) |
| 725032 | 准备／辅助／被替代 | r16 CPU恢复检查job，未训actor | [periodic25-change.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/dynrpo-independent-20261015/periodic25-change.json) · [01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/01a0fcbe-15f8-7600-ae84-9487d924560a-history-hits.json) |

### Co会话执行尝试

| 执行标识 | 状态 | 实际情况／故障 | 证据 |
|---|---|---|---|
| 717554 | 故障退出／历史 | {"state": "FAILED1:0", "stage": "native index admission before pipeline", "formal": 0} | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| 717607 | 故障退出／历史 | {"state": "FAILED1:0", "stage": "unsupported selective minor_number query before pipeline", "formal": 0} | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| 717637 | 故障退出／历史 | {"state": "FAILED1:0", "stage": "shared JIT file visibility before actor", "formal": 0} | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| 717698 | 故障退出／历史 | {"state": "FAILED1:0", "stage": "model smoke legacy cross-backend policy gate", "formal": 0} | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| 717765 | 故障退出／历史 | {"state": "FAILED1:0", "stage": "retriever default8 CUDA probe on actual4 GPUs", "formal": 0} | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| 717827 | 故障退出／历史 | {"state": "FAILED1:0", "stage": "missing joblib at initial-calibration + missing pidfd interface in cleanup", "formal": 0, "teacher": 0, "cleanup_pending": false, "cleanup_remaining": false, "cleanup_job": "717934", "cleanup_job_state": "FAILED1:0", "cleanup_retry_job": "718279", "cleanup_retry_state": "COMPLETED0:0", "private_cache_cleanup_accepted": true} | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| 717938 | 故障退出／历史 | {"state": "FAILED1:0", "stage": "initial model internal/separate LP consistency; exact numeric cause not yet persisted", "formal": 0, "teacher": 0, "cleaned": true, "manual_cancel": false} | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| 717934 | 故障退出／历史 | {"state": "FAILED1:0", "stage": "KeyError pid: corpus claim has nested creator, fixed in2d", "kind": "old717827 private tmpfs cleanup", "cache_cleanup_remaining": false, "retry_job": "718279", "retry_state": "COMPLETED0:0"} | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| 718288 | 故障退出／历史 | {"state": "FAILED1:0", "stage": "pre-manifest storage du twoMOU700 checkpoint dirs", "formal": 0, "teacher": 0, "original_source": "e715462c55af0b836aee47432ca051cc02f5a5d9", "owner_archived": true, "permissions_fixed": true} | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| 718509 | 故障退出／历史 | {"state": "FAILED1:0", "stage": "GPU monitor NVML query timeout30s and final GPU context query also timeout30s", "formal": 0, "teacher_questions": 1024, "actual_trajectories": 5120, "original_gpu_context_query_cleanup_accepted": false, "tmpfs_cleaned": true, "node": "ACD1-31", "recovery_in_progress": true, "owner_archived": true, "preserved_directory": "/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718322-20261014/failed-job-718509", "old_node_readback_complete": true, "cleanup_job": "718646", "cleanup_state": "COMPLETED0:0", "recovery_job": "718650"} | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| HPC3 718650 | 早期执行／随后恢复 | 原MLP恢复链，勿与719225重复算独立设置 | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| HPC3 719225 | 117轮后人工停止 | 实际主run c385，切换新loss | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3/handoff.json) |
| HPC3 721449 | 启动失败 | 原启动失败保留；721485承接新σ3设置 | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-sigma3-loss-20261015/handoff.json) |
| HPC3 721485 | 训练及评分完成，收尾异常 | 200outer及51713评分已完成；原始归档/cleanup故障仍未全部验收 | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-sigma3-loss-20261015/handoff.json) |
| Prefix20 726927 | 故障退出 | pristine启动失败；初始环境/门禁诊断保留；0完整outer | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/handoff.json) · [726927-pristine-failure-evidence-1792155679.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/726927-pristine-failure-evidence-1792155679.json) · [recovery726927-prepare-once-1792156321.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/recovery726927-prepare-once-1792156321.json) · [recovery726927-native4-submit-once-1792156340.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/recovery726927-native4-submit-once-1792156340.json) |
| Prefix20 727000 | 故障退出 | 启动失败；shared JIT相关诊断；0完整outer | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/handoff.json) · [recovery727000-hpc-cpu-terminal-and-cleanup-1792157682.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/recovery727000-hpc-cpu-terminal-and-cleanup-1792157682.json) · [recovery727000-prepare-once-1792157742.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/recovery727000-prepare-once-1792157742.json) · [recovery727000-native4-submit-once-1792157770.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/recovery727000-native4-submit-once-1792157770.json) |
| Prefix20 727050 | 故障退出 | 启动失败；cross-backend numeric gate相关诊断；0完整outer | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/handoff.json) · [job727050-fresh-terminal-recovery-proof-1792159376.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/job727050-fresh-terminal-recovery-proof-1792159376.json) · [recovery727050-native4-submit-once-1792159759.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/recovery727050-native4-submit-once-1792159759.json) · [job727050-fresh-terminal-recovery-proof-1792159336.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/job727050-fresh-terminal-recovery-proof-1792159336.json) |
| Prefix20 727148 | 故障退出 | 缺joblib等启动依赖失败；0完整outer | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/handoff.json) · [recovery727148-durable-cpu-import-fit-1792162530.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/recovery727148-durable-cpu-import-fit-1792162530.json) · [recovery727148-actual-linux-cleanup-four-tests-1792162567.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/recovery727148-actual-linux-cleanup-four-tests-1792162567.json) · [deploy-recovery727148-final.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/deploy-recovery727148-final.json) |
| Prefix20 727251 | 故障退出 | Ray线程资源/启动门禁失败；后移除额外smoke后重试；0完整outer | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/handoff.json) · [fresh-recovery727251-preparation-byte-check1-1792168438.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/fresh-recovery727251-preparation-byte-check1-1792168438.json) · [fault727251-terminal-check2-1792168296.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/fault727251-terminal-check2-1792168296.json) · [fresh-recovery727251-preparation-byte-observation-1792168413.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/fresh-recovery727251-preparation-byte-observation-1792168413.json) |
| Prefix20 727414 | 故障退出 | 真实payload失败；错误base/协调器绑定已留证；无完整outer；0完整outer | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/handoff.json) · [727414-actual-payload-failure-and-zero-write-evidence-1792170376.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/727414-actual-payload-failure-and-zero-write-evidence-1792170376.json) · [727414-recovery-target-state-1792170935.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/727414-recovery-target-state-1792170935.json) · [727414-cumulative-small-recovery-storage-1792171090.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/727414-cumulative-small-recovery-storage-1792171090.json) |
| Prefix20 727586 | 提交待实测 | 纠正payload后单次提交，未读到完整outer；0完整outer | [handoff.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-prefix20-hpc3-4gpu-20261016/handoff.json) |

## 继承关系与容易混淆的记录

1. **A800小MLP链**：原teacher warmup→micro2→fresh micro4→fresh async Top512；async首5轮故障后同run恢复到38，再换σ3。不能把4轮+5轮+38轮相加为一个完整模型。
2. **损失变化**：Co的σ3 λσ=1；A800改为λσ=3；H100三block的binary辅助是0.1 normalized variance MSE；A800binary辅助是0.3 normalized std MSE。它们不是同一loss。
3. **Head200过拟合证据**：同训练题99.88%，新512题mixed分类58.79%；这是泛化失败，正式训练是否QA退化另看QA输出。
4. **epoch68与epoch100不同来源**：Co Prefix20初始化用Prefix-only在线换题epoch68；Gi/Tree后续question+20初始化用jointBCE100实际保留的epoch100。epoch45更高但未留权重，不能声称用它初始化。
5. **最新在线方案**：Gi actor-hidden period20用所有5120候选的标签周期refit；Tree先续写1024再按实际diversity选512；Co新Prefix20只用选中512拟合且另加all-correct虚拟0。
6. **period25**由TOP-D会话运行，但算法不是TOP-D蒸馏；当前是独立原生Qwen QDP。
7. **Question+5 EMA**每epoch遍历445440 G5事件组（2227200轨迹），不是每epoch5120题；actor更新0。
8. Search-G1原来的闭卷探针/证据依赖探针属于reward shaping；QDP是新增选题probe。纯Search-G1旧reward探针校准、硬件smoke、普通mixed-first全rollout、递归Tree等不冒称QDP消融。

## 全部置顶会话覆盖

| 置顶序号 | 会话标题 | Session ID | 扫描记录数 | QDP相关性 |
|---|---|---|---:|---|
| 1 | U-验证 GiGPO 在 Search Agentic RL 的效果 | 01a0fcbb-863f-7450-926d-db537f648265 | 129,184 | 存在实际QDP训练／warmup |
| 2 | U-验证 Tree-GRPO 效果 | 01a0fcbc-0b46-7502-8b66-d8b21a0e032d | 212,871 | 存在实际QDP训练／warmup |
| 3 | 验证 BAPO 在 Search Agentic RL 中的效果 | 01a0fcbd-41d8-7910-a410-372ce5c8870c | 101,317 | 设计、论文或结果汇总；无独立QDP训练run |
| 4 | 验证 search agentic RL 中 0PSD 效果 | 01a0fcbd-c50e-7cf1-8073-256a13ac9f7e | 49,170 | 设计、论文或结果汇总；无独立QDP训练run |
| 5 | U-验证 TOP-D 在 Search Agentic RL 中的效果 | 01a0fcbe-15f8-7600-ae84-9487d924560a | 113,561 | 存在实际QDP训练／warmup |
| 6 | U-验证 CoSearch 在 Search Agentic RL 的效果 | 01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef | 129,082 | 存在实际QDP训练／warmup |
| 7 | 审查七种算法实现 | 01a101e0-90d3-7731-bae9-58c8f1fdf7e2 | 22,939 | 设计、论文或结果汇总；无独立QDP训练run |
| 8 | U-Tree-Branch-GRPO-V1 | 01a1028b-b626-7f91-93db-0099ff1623c5 | 167,960 | 设计、论文或结果汇总；无独立QDP训练run |
| 9 | Search-G1分析 | 01a0fc04-fd7e-7712-8599-ea60aad9b77e | 2,914 | 设计、论文或结果汇总；无独立QDP训练run |
| 10 | 配置 chatgpt.com 代理 | 01a0f82f-35d5-7412-a380-0c84589b6cde | 1,496 | 未找到QDP关键词记录 |
| 11 | WWW论文撰写 | 01a10b9c-086f-7132-a23a-a134a2423a4c | 16,907 | 设计、论文或结果汇总；无独立QDP训练run |
| 12 | R-探索性能低的原因 | 01a11c5d-303b-7441-8db7-c645df252713 | 3,941 | 设计、论文或结果汇总；无独立QDP训练run |
| 13 | R-我们的算法 | 01a11f0e-5a3a-7583-9732-f2e146d9b17b | 6,894 | 设计、论文或结果汇总；无独立QDP训练run |
| 14 | U-Tree-Branch-GRPO-V2 精简恢复 | 01a12e27-764b-7432-a5f2-74d88ee4a8d4 | 112,395 | 设计、论文或结果汇总；无独立QDP训练run |
| 15 | 理论推导 | 01a13aef-081c-7343-88ea-e284ab5d5c01 | 2,115 | 设计、论文或结果汇总；无独立QDP训练run |
| 16 | U-Search-G1 | 01a0fb1b-d3ff-74b3-aa13-49ca57871f92 | 151,306 | 存在实际QDP训练／warmup |

完整检索索引保留每条命中的原始文件字节偏移、行号、时间和SHA；被截断的展示摘录可按偏移回到原始记录核验。

审计文件：[experiments.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/experiments.json) · [execution-attempts.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/execution-attempts.json) · [inventory-summary.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/inventory-summary.json) · [full-history-scan-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/full-history-scan-audit.json) · [actual-metadata-evidence-index.json](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/question-diversity-probing-inventory-20261017/actual-metadata-evidence-index.json)

边界：这是截至快照的历史清单和本地既有实际receipt核对；没有把当前会话活跃、Slurm RUNNING、提交成功或准备成功当作新增完整训练轮数。历史故障中仍未定位到具体根因的项已原样保留，不宣称所有故障均已修复。

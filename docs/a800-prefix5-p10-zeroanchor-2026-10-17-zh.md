# A800 question＋prefix5：每10轮更新、真实混合选题与全对组零锚点

日期：2026-10-17，Asia/Shanghai。

## 本次明确人工修改

候选从5120道题减为2560道题，每题仍5条，共12800条前缀。正式actor总200 outer，每轮仍512道题×G5＝2560条轨迹、mini128、PPO epoch1、20次真实Adam。探针每10轮更新一次，只在10、20、…、190；第200轮不更新探针，仅正常actor更新、checkpoint和既定评测。

探针LR固定5e-6；每更新轮使用该轮全部2560题G5的原生question＋前5token表示与最终二元EM，自然打乱完整训练1 epoch，共50次同步Adam。全局batch循环51/51/51/51/52个完整G5，平均256轨迹；四卡每步12/13组，每卡一轮640题/3200轨迹。19次周期共950次探针Adam，副本只计一次。第一次fresh在线Adam，后续继承其状态与四rank RNG；不重置周期Adam。

## 普通轮与更新轮选题

普通轮由更新前探针pMixed选Top512，随机0，不读取最终EM参与选题。更新轮先固定当轮actor版本，完成全部2560题原G5的续写与strictEM。真实mixed＝0<K<5，包括只有1条正确；优先从真实混合题选512组，组内仍保持自己的原G5。在真实混合题中按探针pMixed降序、原candidate ordinal升序；若不足512，保留全部真实混合题，再从非混合题按同样分数补齐。记录实际候选混合数、选中混合数、fallback数，不能把补齐后512组假称全部混合。

原label-free predicted selection与features manifest/hash保留；更新轮另生成actual-em-selection.json，投影完整pool的原始token、old-policy LP及工具上下文给actor。最终selection SHA与pool原选择/完整pool/ownEM/predicted选择的四项来源链均绑定。普通轮绝不使用这个真实EM选题入口。探针使用全部候选的自然分布，不能只拟合选中的512组。

## 探针模型与联合loss

沿用独立原生Qwen完整block35＋masked flatten＋单logit Linear。输入来自当前采样actor layer27 postblock的全部有效question token及前5个policy token；question上限273、布局278。第6个lookahead只为了取得第5个token的完整hidden，不进入探针/PPO/续写。输入detach，无独立冻结encoder前向，不把探针梯度反传actor。

loss仍为mean逐轨迹自身EM BCEWithLogits＋mean同题G5 mixed BCE，各权重1。p_j＝sigmoid(z_j)，h＝1−prod(p_j)−prod(1−p_j)，mixed标签0<K<5；32模式log域稳定等价，无新增std/六分类/类别重权。EMA仅用于初始导入，正式在线模型不做EMA递推。更新前后在该2560题与既定固定512题诊断上记录两项loss和准确率；无额外GPU验收/基线/模型概率扫描。

时序：进入outer10的actor9生成prefix/完整EM，先按真实EM选题并更新actor，再用该actor9的原hidden/EM更新probe，供outer11使用。不会对actor10重算旧表示，也不会把新的回答EM配给旧prefix。

## 全对G5的GRPO advantage

用轨迹自身真实strictEM判断分组。只有K＝5全对组加入一个虚拟0参与统计：

s＝[0,r1,r2,r3,r4,r5]，a_j＝(r_j−mean(s))/(std(s,correction=1)＋1e-6)。

最后只返回原5个a_j。虚拟0没有trajectory、token、oldLP、loss mask或梯度样本，不进入PPO条数。原grounding soft rewards原值不改。

全错K＝0和混合K＝1..4均保持原5条reward的mean/sample std公式。举例[.97,1.03,1.05]仅说明公式，加入0后mean=.7625/sample std≈.509469，保留三个优势≈[.407286,.525056,.564312]。实际生产仍严格自己的G5。该改动使全对组优势不再必须均值0，因此不同于旧组内中心化；这是本次人工要求的算法修改。

## 启动与保留

旧411c/ex9074/source2c实验在源码修改与准备期间继续。新版独立source/namespace，先提交push及远程SHA/blob，stage全部源文件，再CPU复用已合格CP5不可变文件并验完整SHA和原资格链；不重复teacher、编码、全量模型tensor读取或数据划分。固定512身份/共同manifest/原数据及检索资产不变。

新actor从官方Qwen2.5-3B-Instruct revision aa8e72537993ba99e69dfaafa59ed015b17504d1 fresh开始；probe只导入原已封存完整预热epoch5 EMA SHA b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594，不继承warmup Adam/RNG。不是旧actor full-state resume，旧完整checkpoint和真实未完整attempt ledger独立保留。

CPU准备成功后，仅向旧owned pipeline精确PID/birth发一次pidfd SIGTERM；实际确认全部本项目children退出、完整/partial记录保留，再一次启动新版。四训练卡及aux5/6实际idle准入，不能停或用外国任务。沿原control.py路由，不改SSH/VPN/auth/knownhosts。

新控制入口 reproduction/operations/a800_actorhidden_prefix5_p10_control.py，动作stage/prepare/stop-actor/reconcile-old/launch/observe均有独立once记录。元数据观察有界65秒SSH，不读模型/tensor，不重复launch。启动后两次实际增量及首完整outer必须实核，同pipeline继续200；既定5/50/100/150/200保存和QA、最终51713完整验收及only-owned清理保持。大资产全远端，Git只代码/config/docs/scalar/hash。

## 本次实际部署（启动提交阶段）

运行源码31f616264ceeb4f0e4d33b39bd85df1214afc15d，remote1277文件SHA一致。CPUprepare实际exit0，复用相同CP5完整SHA与原资格链，0次checkpoint tensor load；最终configSHA0f9fd64fbc28a1d849f7716030dd4f960e990ed9d50fd9f2b0ffe0241e3d2011。旧run411c已完整4轮/80Adam，所有707记录ownedbirth实际gone，第5轮续写期间中止、无该轮actorattempt；原checkpoint/日志/轨迹保留。

新版run387d3f6e228a488f90c418e2ba47e28e/ex90aea944c40d41eab4cf84efac02b22a，一次启动Unix1792176126.7192712，pipeline1736370/birth625571803。这仅证明launch提交；实际数据流/actor更新/首完整outer须继续核实，不能将旧run进度加到新run。小型scalar/hash原件汇总于reproduction/reports/a800-prefix5-p10-zeroanchor-startup-2026-10-17.json。

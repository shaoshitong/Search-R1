# I1–I6 修复与官方actor fresh200

按2026-10-17最新人工指令，本次执行清单仅I1–I6。旧3430已人工取消，last joint99；18888原进程birth两次核对全部消失，cgroup与自有GPU占用均空。错误实验a008／5cba的12个源命名空间仅删除自有权重／optimizer／log，代码、必要小审计与原始轨迹gzip保留；warmup、官方权重、真实round0 grounding、数据、FAISS及fixed recovery assets保留。

|项|修复实现|必要验证|
|---|---|---|
|I1|train/QA/后续grounding校准共用新interaction contract：4096total、500action/answer reserve、闭合information、预算early-final|wrapper/预算/动作CPU边界；正常QA5起验证实际输出|
|I2|physical local16不减，forward后8个原micro2 token mean等权；OOM拆到8/4/2/1仍同目标|真实PPO函数loss/梯度CPU等价、mask与空EOS组|
|I3|真实round0导入；50/100/150用当轮actor真实fit两head、8replica adopt；未adopt不得下一轮|版本时钟／恢复／CPU保存点测试；真实refit到50才发生，不冒称已GPU验收|
|I4|恢复每mini真实before/after原KL与负优势tail monitor；固定rank4条；after可作下一before|原统计CPU检查；有限阈值按已授权diagnostic-only告警、真实非有限拒绝；无独立GPU gate|
|I5|train/QA统一BF16，FP32 master/loss保持，原题greedy及strict scorer不变|新QA contract/dtype与8worker实际输出绑定|
|I6|train/QA共用首个完整action的stream停止、EOS与原IDs裁剪|prefix20/21flush不变，同prefix续写及LP/mask/position保留；CPU stream mock|

新actor：官方Qwen2.5-3B-Instruct aa8e725；旧错误actor／Adam不导入。Warmup不重跑，仅导入已实际保留最佳CP100探针模型，online Adam0。最新显式goal将探针学习率设为1e-6（替代先前1e-5）；每20轮全5120G5自然一遍100Adam，其他轮冻结。Actor仍200outer、20Adam/outer/rank、globalmini128、LR1e-6/57outer warmup；8卡大batch。固定512在5/50/100/150/200及final51713沿用原题与strict评分，新interaction contract统一执行，不把旧QA分数混入本run。

正常checkpoint/backup/archive已改为动态绑定本run，含grounding版本／encoder／head依赖；没有旧a008固定身份。旧round0实际拟合资产来自原协议，本次明确复用；后续正常校准显式启用新交互合同，不更改其他实验默认路径。初期验收须真实连续5轮＋第6轮实际开始，不能以CPU测试或RUNNING代替。

未进行额外GPU测试、基线重生成或重复warmup。新run只在源码正常commit/push及独立源码SHA后一次部署提交。当前记录不宣称新训练已完成。

## 新运行实测：前5轮、QA5及第6轮继续

2026-10-17 UTC06:08，同一3431／h100-8已完整6／200、八rank各120次真实actor Adam，第7轮continue。1–5各8rank原始账本逐一核20写入，后续第6轮的全部25600个前缀已真实生成并完成续写／actor更新；初期验收完成，不在5／6停止。探针尚未到20，online Adam0是正确时钟。

普通轮2–5的真实joint提交间隔平均337.678秒（5分37.68秒），范围332.489–346.813秒，排除启动和QA5。前5轮selected真实mixed分别为72.0703%、71.6797%、74.0234%、74.6094%、67.5781%；这些是已筛选训练组比例，不是全候选准确率，也不是QA效果。

固定512题QA5已完成：EM139／512＝27.1484375%，F1＝0.3547144455587428；8worker全部exit0，实际新统一交互合同、BF16与原question SHA均核对。仅是新运行早期结果，不宣称算法收益。完整训练／最终51713尚未完成。

CP5独立完整SHA及实际CPU状态读回通过，30文件38,613,963,993B，actor100／probe0／scheduler5及八rank恢复状态齐全；正常CPU核验工具的局部变量`digest`遮蔽已在`6e8cd92a`修复、14项相关CPU检查通过，并只部署到独立preparation工具目录，冻结训练848不改。未来正常保存点必须用该已修工具，不能再调用冻结repo内旧检查器。h100-10独立副本已once启动，165文件44,996,144,989B，PID847868／startticks1927785014，源与目标完整SHA拷贝仍PENDING，不冒称备份完成。模型／Adam／轨迹均未经过Mac。

实际证明见[小型结果摘要](first-five-qa5-and-cp5-actual.json)。管理新连接曾被拒，复用已有已认证的direct mgmt-host master已成功核对同一训练；只读观察85864已结束，未停止远端3431。App监督当前显示PAUSED／60分钟，更新提示内容时保留该界面设置，不把监督暂停当作停止训练授权。

## 三会话修复交叉核对（2026-10-17）

按人工明确授权，只读核对各自实际部署的冻结源码，不把各实验的候选数、prefix长度、virtual0或探针周期差异自动当作错误。以下结论是代码审查，未新增GPU测试、未热改任何活跃训练；未来50轮实际拟合尚不能以静态调用链冒充完成。

|会话／冻结源码|仍需处理的具体事项|通知与边界|
|---|---|---|
|Tree-GRPO／`68d2d131aee91d194c487f54a0204f53397d87ea`|`prefix_diversity200/grounding.py:54`仍调用旧`ts_probe_dump`；后者直接裁完整information包、4608引擎／4096提示尾裁，没有新答案预留。未来50／100／150的校准轨迹仍会与已统一的train／QA分叉。|已通知，对方确认在隔离副本修复；此路径尚未到50，不能据此声称新首轮已损坏。cost／joint恢复round0固定值已在其后续控制提交修正，不重复列为未修项。|
|TOP-D／`45e8f9b39cf387918a1752878858b1c4980b6349`|`common8/refit.py:31`同样仍调用旧校准生成；另`sampling_capture.py:417`仍等待native完成，`behavior_trace.py:205`对search闭标签优先，可能保留更早answer后的额外文本；无完整动作的native EOS会加correction继续，和本线程首个完整动作／EOS终止语义不同。|两项均已通知，对方已开始核实校准路径。尚无新run实际命中率或EM影响量化，不作质量因果结论。原micro归约、before／after monitor、grounding刷新调用本来存在，不能说漏掉全部链路。|
|Search-G1 Prefix5／`79c645a4a51bd0bcb0fab4ee36b8fcd634290b2f`|新train／QA／校准、动态grounding拟合／adopt已接通。旧`native_joint_resume.py:65,161,307`仍以20×outer及旧无grounding配置核验，旧CPU资格入口还固定joint27／prefix28，不能直接恢复新schema。|已通知适配独立恢复控制；未发现必须立刻停止当前fresh主循环的已确认错误。它的有限KL／tail阈值仍可skip Adam，本线程仅诊断，必须如实比较实际Adam／skip，不能只按outer等同预算。|

本线程848a67a4已显式为正常grounding校准传`--interaction-config`，与train／QA共用`calibration_interaction`；checkpoint／backup工具同时记录动态grounding版本及依赖。它们已通过必要CPU检查，真实第50轮fit／adopt仍待正常进度验证。各会话获知后自行负责发布和正常完整断点交接，通知不等于其修复已部署验收。

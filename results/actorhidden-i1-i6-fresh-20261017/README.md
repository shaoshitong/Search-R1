# I1–I6 修复与官方actor fresh200

> Latest actual 2026-10-17 10:50 UTC: same3431/h100-8 complete51/200, each rank actor1020/probe200;52prefix. QA50 original512 output/identity/strict scoring independently read back:190 correct, EM37.1094%, F1 45.4629%, all8workers0. NormalCP50 has36 full-file independentSHA/CPU-state reads; actual grounding50 fit/adopt and original51 eight-rank reward-version use are proven. Independentcopy50 is progressing, not complete. [Milestone50 actual evidence](milestone50-qa-grounding-cpu-and-actual51.json). Frozen848 unchanged; probeLR1e-6 until human's50-review reply. Older snapshots are historical.

> Latest actual 2026-10-17 09:46 UTC: same3431/h100-8 complete41/200, each rank actor820/probe200;42 prefix. Period40 original eight probe ledgers verify100 new writes each at1e-6; actual41 scoring used updated head40. [Same-rollout paired metrics and100-update curve](period40-paired-and-actual41-use.json), [selected-group history1–40](selected-mixed-outers0001-0040.csv). Frozen training848 is unchanged. QA50 and grounding50 are not yet observed.

> Latest actual 2026-10-17 09:02 UTC: same3431/h100-8 complete35/200, each rank actor700/probe100,36prefix. All25-35 original actor rank journals verify20 writes per rank/round. Two actual management/target reads recovered with the task-only macOS socket-interface mode; no training restart/config edit or global network/auth edit. [Current small progress proof](progress35-and-task-local-transport-recovery.json). ProbeLR stays1e-6; human conditional20 resume could not apply because complete20 actor/Adam/RNG files were rotated; reconsider at retained50, with no automatic LR change approved yet. Older snapshots below are historical.

> Latest actual 2026-10-17 07:37 UTC: same3431/h100-8 complete20/200, each rank actor400/probe100. First period20 trained all5120G5 once at1e-6; actual21 selection used head20 with actor0020. Train balanced mixed accuracy60.9604% ->63.6124%, heldout56.6099% ->57.0806%. Diagnostic gains are modest and ordinary heldout accuracy remains below its majority baseline. [Paired metrics and100-update curve](period20-paired-and-actual21-use.json). Current runtime848 remains frozen; older snapshots below are historical.

> Latest actual 2026-10-17 07:14 UTC: same3431/h100-8 complete17/200, each rank actor340/probe0, outer18 actor stage. Probe0 before20 follows the protocol. CP5 independent copy completed at07:19:51: all165 payloads44,996,144,989B source/target full SHA matched; a separate final metadata read verified166 final files including mapping, exact sizes, no staging and original copier birth gone. See [independent CP5 copy summary](checkpoint5-independent-backup-complete.json). Semantic reload and all-experiment backup are not claimed. This supersedes older snapshots below.

> 最新实际UTC2026-10-17 06:39：同3431已完整11／200，8rank各actor220／probe0，第12轮actor；原始8–11账本均核20真实写入。CP5副本同birth继续，16文件17,946,369,805B源与目标完整SHA已通过，尚未整体完成。中断只结束了本地等待，已恢复读取同一训练，未重启／改训练。见[第11轮与观察通道恢复小摘要](progress11-observer-recovery.json)。以下首五及QA5为历史阶段记录。

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

## First period20: paired same-rollout metrics and actual21 adoption

The sampler actor0019 generated all5120 training groups and the512 disjoint heldout groups. Every group contains its own five retained prefixes, continuations and strict-EM labels. The original eight rank ledgers each contain100 actual probe Adam writes,640 groups/3200 trajectories, with80 global batches of255 trajectories and20 of260; learning rate1e-6. Training/heldout question IDs and normalized-content hashes have zero overlap. The old probe selected Top512 before labels; selected continuations were projected from the same full pool without extra generation.

|Same data, pre/post probe update|Train5120 groups|Heldout512 groups|
|---|---:|---:|
|Ordinary mixed accuracy|60.6641% ->62.6563%|56.8359% ->56.4453%|
|Balanced mixed accuracy|60.9604% ->63.6124%|56.6099% ->57.0806%|
|Mixed recall|61.7665% ->66.2137%|55.9524% ->58.9286%|
|Nonmixed recall|60.1542% ->61.0111%|57.2674% ->55.2326%|
|Trajectory BCE|0.624287 ->0.610446|0.699766 ->0.691915|
|Group BCE|0.717979 ->0.690690|0.738013 ->0.736743|
|Total joint loss|1.342266 ->1.301136|1.437780 ->1.428659|

Training truth contains1619/5120 mixed groups; heldout168/512. Top512 contains261/512 mixed (50.9766%), against31.6211% in the full pool,1.6121x prevalence. This is selection enrichment, not full-pool accuracy or QA gain. Heldout nonmixed-majority baseline is344/512=67.1875%; post-update ordinary accuracy56.4453% remains below it, while balanced accuracy is57.0806%. Quality remains limited. The paired loss decreases use an identical head-evaluation data set before/after; the100 online update losses use different batches and cannot establish monotonic decrease.

Actual21 score completion independently binds head20 producerSHAeea4bd7dd483d3ec9f654201200ee85da55eeaf38cb41afca708eef7262a149f to actor0020 with no outcome-label use. No tensor was loaded or independently rehashed by this metadata read, and no model semantic reload is claimed. Probe100-update loop took8.24249seconds; complete probe phase40.97637seconds. Phase timings and raw counts are retained in the linked small result; neither loop time nor phase sum is claimed as whole-outer wall time. QA20 is not a scheduled evaluation; QA50/final51713 remain pending.

## Selected-group history, rounds1-20

The [20-row CSV](selected-mixed-outers0001-0020.csv) reports each selected512-group outcome histogram and mean predicted mixing probability, independently matched to the actual selection and strict-EM metadata. All20 rounds used the same pre-update initial probe, as required; updated head20 was first used at21. First-five mean selected mixed fraction was71.9922%, last-five50.2734%; round20 was50.9766% while its selected mean predicted mixing probability was89.6865%. These probabilities remain overestimated on selected groups. The actor and candidate questions change across rounds, so this history does not identify a sole cause or measure fixed-label classification performance. Ordinary rounds have no labels for unselected4608 groups. [Small source/hash metadata](selected-mixed-outers0001-0020.metadata.json) preserves the exact bindings. Training is unchanged.

## Optional local macOS socket-interface transport control

A local-only opt-in can now bind the source-bound SSH byte socket to en0 using Darwin IP_BOUND_IF, in addition to binding its source IP. The installed MacOSX SDK declares option25 and a real, unconnected local socket set/get verified interface index11. This addresses a concrete distinction: the alias specifies an en0 source address while the observed host route selects utun4; source-address binding alone does not demonstrate which interface carries the connection. This is a transport hypothesis, not a proven cause or an end-to-end recovery claim.

Default callers remain unchanged. The interface option is explicit, macOS-only, and does not alter global SSH, authentication, VPN or routing configuration. It is passed only to task-local management reads; frozen trainer848 is unchanged. Syntax checks cover the three modified control modules; actual authenticated management/native read remains necessary to establish whether this mode works. No training/GPU test was added.

The opt-in transport mode has now completed actual authenticated management, h100-8 hostname/nonce and native3431 original-ledger reads. Source-address-only and ordinary alias reads had failed earlier; this temporal comparison does not isolate a permanent network cause. The option is enabled only in this task local observer state. Frozen training848 is unchanged. The next observer uses the published interface-capable helper, reuses the verified owned master while live, and can use the same task-only socket binding on a fresh connection.

## Period40: weak discrimination despite ordinary-accuracy gain

Actor0039 generated this period's5120 training G5 groups and512 disjoint diagnostic G5 groups. Pre/post evaluation uses the exact same rollouts, hidden and own strict-EM labels within each split; all eight ranks naturally consume640 groups/3200 trajectories once. The probe takes100 real synchronized updates at1e-6. The next outer41 actually binds updated head40 to actor0040. Producer model SHA metadata is not an independent tensor readback or semantic reload.

|Same data, pre/post update|Train5120 groups|Heldout512 groups|
|---|---:|---:|
|Ordinary mixed accuracy|58.5156% →67.8516%|54.1016% →69.5313%|
|Balanced mixed accuracy|57.3133% →56.2855%|51.2043% →52.5406%|
|Mixed recall|54.9329% →33.3860%|46.0177% →22.1239%|
|Nonmixed recall|59.6937% →79.1851%|56.3910% →82.9574%|
|Trajectory BCE|0.699631 →0.803111|0.782394 →0.905347|
|Group BCE|0.749673 →0.620271|0.751339 →0.615910|
|Total joint loss|1.449304 →1.423382|1.533733 →1.521257|

Ordinary accuracy rises mainly because the head predicts nonmixed more often. Heldout majority baseline is399/512=77.9297%, above the69.5313% ordinary accuracy. Balanced accuracy remains close to50%, mixed recall declines and trajectory BCE worsens: this is not successful generalization. Joint loss decreases slightly because the group term improves more than the trajectory term worsens. Different online batches do not establish monotonic loss decrease.

The full pool contains1267/5120=24.7461% mixed groups; Top512 contains181/512=35.3516%, around1.4286 times pool prevalence. This is selection enrichment, not full-pool accuracy or QA improvement. The fresh diagnostic has113 mixed groups. Its questions stay fixed, but each periodic actor generates new G5 labels/hidden; period20 versus40 is not a fixed-label learning curve. The100-update loop takes8.24818seconds, not whole-outer time.

Human's conditional request to resume20 at1e-5 could not apply because complete20 actor/Adam/RNG files were rotated. Probe LR therefore remains1e-6; retained50 is the next review point. No automatic stop or LR change at50 is assumed.

## Normal50: actual QA, grounding fit/adoption and complete recovery state

|Same new run and shared QA contract|Outer5|Outer50|
|---|---:|---:|
|QA512 strict EM|139/512=27.1484%|190/512=37.1094%|
|QA512 F1|35.4714%|45.4629%|

Outer50 has256 NQ and256 Hotpot questions, with NQ EM97/256=37.8906% and Hotpot93/256=36.3281%. All original eight workers exited0. A CPU-only reader independently hashed/read the16 original rank-receipt/output files, checked every canonical qid/ordinal/source, and recomputed strict EM/F1 from actual prompt/response token IDs, masks and original gold using the actual scorer. It matches the producer aggregate. Full outputs stayed remote. Asynchronous output order is aligned by canonical ordinal, rather than zipped against sorted receipts. This is post hoc output verification, not model generation or a training gate. QA50 uses the same new BF16 interaction contract as this run's QA5; old broken-run/legacy scores are not silently combined with this curve. The change cannot isolate a data-selection algorithm benefit without a matched control.

The original300-question grounding procedure actually fitted boundary and commit heads against actor0050 with2-fold grouped CV: commit AUC0.838282 and boundary AUC0.741794. The recorded new interaction contract matches train/QA. All8 reward replicas actually adopted the resulting version, and the original completed51 actor receipt shows all8 reward ranks used that exact version. This now supplies runtime evidence forI3, beyond earlier static call-chain checks. These are reward-grounding heads; the correctness-selection head still updates only20/40/60…200.

The immutable50 recovery point has36 independently full-hashed files,39,240,509,750B, actual CPU load of all8 actor/rank states and periodic probe: actor1000/rank, probe200/rank, scheduler50, saved RNG, pool consumed256000/next51. Effective next grounding round50 is fitted/adopted; saved actor50 itself correctly used round0 for its preceding update. No GPU semantic reload or CUDA-RNG application is claimed.

An independent h100-10 ext4 copy started once (PID1103066/startticks1929451008),171 payloads45,615,868,686B at20MiB/s with128GiB ordinary reserve. Two source/destination-phase observations show the same birth progressing from1.263GB first-shard copy to a real3,968,658,944B target shard with638MB of its independent target full hash read. It remains pending, not a completed backup or semantic reload. No model bytes went through Mac. Existing unique intent/receipt must not be replayed.

Human asked to revisit probeLR at retained50 after complete20 proved unavailable. A choice between complete50 resume at1e-5 and continued1e-6 has been presented; no answer has been assumed. The current healthy job continues at1e-6. If that change is authorized, it must preserve actor1000/probe200, optimizer moments, scheduler50, RNG, pool and adoptedgrounding50; the old pre20/Adam0 LR5 importer cannot be used as-is. No old warmup or actor reset is permitted. Full200 and final51713, future milestones and all final closure remain pending.


## Existing probability distribution audit through actual73 (CPU-only)

The human-requested audit reads existing per-group probabilities and strict-EM metadata remotely; no tensors, model forwards, training edits or prediction rows leave h100-8. `probability_distribution_audit.py` reproduces the aggregate; `probability-distributions-through73.json` contains source hashes, probability quantiles, calibration bins/Brier/AUC and selected/candidate summaries. The training runtime remains59d; later publication is not a runtime deployment. Probe LR is now the explicitly authorized1e-5 after the full50 restoration, superseding the historical pending-choice paragraphs above.

|Post-update fixed-question heldout|20|40|60|
|---|---:|---:|---:|
|Mean trajectory p / actual correctness|15.99% /33.09%|10.24% /37.11%|25.25% /39.14%|
|Mean mixed q / actual mixing|49.90% /32.81%|35.45% /22.07%|63.51% /19.34%|
|Mixed ranking AUROC|0.59579|0.54455|0.48962|

At73, Top512 meanq is92.0574%, but111/512=21.6797% are mixed;303/512=59.1797% are allcorrect and98/512=19.1406% allwrong. Selected trajectory probabilities all lie in0.21867..0.77322 (mean0.46304), and selected q spans0.88739..0.94342. Thus high scores are largely selecting intermediate, similar correctness probabilities rather than demonstrating different eventual outcomes. The formula naturally gives15/16 at five identical p=.5; high q alone is not evidence of calibrated diversity. Heldout60's highest51 q groups average92.71% predicted mixing but only7/51=13.73% actually mix, versus99/512=19.34% over the full heldout. This is a small observed top-decile sample, not a statistical significance claim.

Direct reconstruction of every audited q from its five saved probabilities differs by <1e-14. This checks formula arithmetic, not the entire training implementation or the conditional-independence assumption. Same-question p max-minus-min averages.07148/.03722/.05239 in heldout20/40/60; this documents similar scores, not proof that prefixes/hidden are identical. Tests use new actor-specific trajectories each period, so no isolated LR causal claim or fixed-label time-series interpretation is made. No unscheduled LR/loss/sampling change or training stop follows this audit.


## Human-scoped local actor50 fitting experiment (formal3434 stopped)

The human explicitly stopped formal training to diagnose probe fitting locally. Native3434 was cancelled after complete75; original terminal inode SHA and owned birth/cgroup/GPU release were preserved. No automatic formal200 restart is authorized. `local_probe_fit50.py` is a separate native entry: freeze actor0050, take the original50 candidate5120 questions, generate one fresh complete G5 pool plus one fresh same-actor fixed512 heldout, capture native question+20 hidden once, then repeat the same train25600 trajectories for10 independently shuffled whole-G5 epochs. The old50 rollout used actor0049 and onlyTop512 labels; it is not falsely reused as actor0050 full-pool data.

The50 joint contains the last genuinely fitted correctness head40 withAdam200. Import those weights/moments/RNG, set current human-authorizedLR1e-5, and train100 new synchronizedAdam/rank each epoch, final1200. Global batches remain80x255+20x260 with completeG5. Loss remains mean owntrajectory strictEM BCE plus mean ownG5 mixed BCE, weights1/1. Epoch0 and each postupdateepoch evaluate the identical fulltrain and heldout features/labels, with ordinary/balanced groupaccuracy, recalls/confusion, trajectoryaccuracy and bothlosses. No encoder or actor forward is repeated during these10 epochs. This initial fixed-data experiment tests fitting/overfit; it alone cannot identify learning rate versus loss as the unique cause.

All checkpoints/hidden/rollouts/predictions stay remote. The local config is `reproduction/prefix_selection200/local-probe-fit50-config.json`; only this local diagnostic entry is submitted, never the200outer pipeline.


Local50 execution uses prior authorized physical8 sharing:3440 pending0time was cancelled before any work;3441 ran on96CPU/512GiB with0 reservedGPU/8 actualCUDAUUIDs, preserving foreign3439 onGPU0. All25600 actor0050 prefixes/nativehidden completed once.3441 then actually FAILED1:0 at fullcontinuation vLLM initialization onGPU0 (foreign49.28GiB; typed CUDA OOM), before any probe writes.20 original inode logs fullSHA and subsequent ownbirth/cgroup/GPU release are preserved in the source terminalproof SHA8a7faead94dd417498d57c7bd23b73ccec0cd423bbbbdce34ee07cc1bca506e9.

The isolated recovery reuses exactly those complete prefixes/hidden/states via read-only hardlinks, never regenerates them, and starts only full continuations. OnlyGPU0 resource settings reduce to fixedvLLMutil.16/maxseq256/maxbatched8192/eager; other7 retain their largebatch/autofree policy. The actual initializationOOM retry also now leaves its except block beforegc/emptycache, allowing traceback-held allocations to be released. Actorweights/inputs/seeds/headloss/LR/10epoch budget remain unchanged. These are actualOOM resource repairs, not a numerical gate. A read-only local observer follows10epochs/nativeexit and never changes training or primary state. CPU identical-input conflict analysis is prepared for posthoc use; it reads actual token contexts and labels without hidden tensors/model forwards.

# Question diversity probe 实验记录与标准差损失对照

记录日期为2026年10月15日。旧实验分析截止点为已验收的formal outer108；本记录不将历史状态视为实时进展，不宣称旧实验200轮或最终恢复已验收。用户本日授权另起仅修改loss的四卡对照，目标为200个成功outer。

六分类loss下降尚不能证明探针更擅长选出高标准差题。混合组减少也不能单独解释为退化：随机组全对比例和固定验证EM同时上升。新实验先检验监督目标对齐，容量另做比较。

## 原实验设置

| 项目 | 设置 |
| --- | --- |
| Actor | 官方Qwen2.5 3B Instruct，revision aa8e72537993ba99e69dfaafa59ed015b17504d1 |
| 资源 | HPC3 jxu957，4 GPU、48 CPU、512 GiB，emergency_acd |
| 旧运行 | job719225，冻结源码1843babdd96b387dd10d91e3207c9ea9489b6941 |
| 问题特征 | 冻结官方模型第27层，2048维 |
| Head | LayerNorm→Linear64→GELU→Linear6，135622参数 |
| 选题 | 5120候选，先均匀随机64，再从剩余题中按预测标准差选Top448 |
| 生成 | 512题各5条完整真实轨迹，不按生成后EM筛选或补题 |
| Head优化 | Adam 3e-4，batch256，每outer两epochs |
| Replay | 16outer窗口，年龄半衰期4outer |
| 教师warmup | 官方冻结actor，5120唯一题×5真实轨迹，全部先拟合head |
| 评测与checkpoint | 固定512题在5、50、100、150、200；最终原生51713题 |

K为五条严格二元EM的正确数，样本标准差为

\[
\sigma(K)=\sqrt{K(5-K)/20}.
\]

旧loss：

\[
L_{old}=CE_6(K)+0.1(\hat\mu-K/5)^2+0.1((\hat\sigma-\sigma)/\sqrt{0.3})^2.
\]

选择使用分布下的期望样本标准差，不用二分类阈值，也不用预测均值代入Bernoulli方差。原Search G1 reward、GRPO/PPO、LR warmup57、数值和行为概率检查均保留。

## Loss与分类诊断

| 时点 | Replay加权总loss |
| --- | ---: |
| warmup | 1.177557 |
| outer5 | 1.232647 |
| outer16 | 1.444584 |
| outer30 | 1.499656 |
| outer50 | 1.378034 |
| outer100 | 1.158731 |
| outer108 | 1.143829 |

Loss先随在线标签变化上升，再下降。Outer100六分类CE为1.132226，均值辅助项加权贡献约0.013766，标准差辅助项约0.012739。标准差项约占总loss数值的1.1%；这不等于梯度贡献比例。原loss确实包含标准差监督，但主任务仍是六分类，且没有直接排序目标。

| Head | Replay六分类普通accuracy | 年龄加权accuracy |
| --- | ---: | ---: |
| warmup | 66.5234% | 66.5234% |
| outer5 | 61.6797% | 59.1294% |
| outer50 | 51.7944% | 51.4472% |
| outer100 | 60.2905% | 60.7613% |

Warmup的argmax全部为K=0，accuracy等于多数类基线，不能据此证明学会选题。Outer100 replay加权混合recall为5.7833%，balanced accuracy为52.7486%。这些结果来自已有head的CPU推理，没有新增拟合。

Outer101至108的4096个预测前置样本：六分类accuracy50.6836%；先argmax再归并的二分类accuracy77.0508%；用p1+p2+p3+p4≥0.5判断时76.9043%，恒预测非混合基线77.6611%。概率和规则的混合recall为27/915，即2.9508%；预测标准差对混合的排序AUC为0.580813。

六分类正确蕴含二分类正确，仅对“argmax后归并”成立，概率和阈值是不同决策。普通accuracy与balanced accuracy也不能直接比较。同一六分类accuracy可以对应不同的跨组误分错误。

训练replay随机子组的标准差排序AUC约0.77，新随机样本约0.57；但policy、时间和选择分布不同，没有同一policy下独立QID holdout，因此不能确定过拟合或欠拟合，容量不足仍属待验证假设。

## 混合比例和随机组分解

混合定义为1≤K≤4。下表只统计实际生成标签的选中题，不推断全部5120候选的真值。

| Outer窗口 | 全部512混合率 | 随机64混合率 | Top448混合率 |
| --- | ---: | ---: | ---: |
| 1至10 | 44.9219% | 28.5938% | 47.2545% |
| 11至20 | 48.3008% | 32.3438% | 50.5804% |
| 41至50 | 36.4258% | 25.1563% | 38.0357% |
| 91至100 | 22.6953% | 16.2500% | 23.6161% |
| 101至108 | 22.3389% | 13.0859% | 23.6607% |

Outer108为全部119/512、随机10/64、Top109/448。随机预算始终64/512；原算法没有随着训练提高混合配额的调度。

| 随机组窗口 | 问题数 | K=0 | K=1至4 | K=5 | 五轨迹平均EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1至10 | 640 | 66.8750% | 28.5938% | 4.5313% | 16.5000% |
| 11至20 | 640 | 58.9063% | 32.3438% | 8.7500% | 23.4063% |
| 41至50 | 640 | 53.1250% | 25.1563% | 21.7188% | 34.2500% |
| 91至100 | 640 | 48.9063% | 16.2500% | 34.8438% | 43.0000% |
| 101至108 | 512 | 51.1719% | 13.0859% | 35.7422% | 41.9141% |

11至20与101至108相比，K=5增加26.99个百分点，K=0减少7.73个百分点。这支持作答结果分布向更多全对组移动，而非仅全错增加。不同窗口没有逐题配对，不能解释为逐题转移或因果证明。

Top448的HotpotQA占比由1至10的64.75%升至101至108的89.06%，后者全部512题中HotpotQA占84.01%。原始候选来源比例只在窄区间波动。选题确实带来训练来源偏移，但不足以证明它导致验证下降。

## 固定验证和历史mixed first G1

当前probe同一固定512题：

| Outer | 正确数 | EM | F1 |
| --- | ---: | ---: | ---: |
| 5 | 101/512 | 19.7266% | 25.8452% |
| 50 | 184/512 | 35.9375% | 43.6407% |
| 100 | 211/512 | 41.2109% | 49.0129% |

50至100增加27道正确题。混合率下降不能单独证明训练越来越差，也不证明当前选择优于基线。

历史mixed first G1来自“验证GiGPO在Search Agentic RL的效果”任务，run15bb15365346401da1d919e9cde8f813，job3364/3381，恢复源码0553adec146537e07b84377ce31cba41a54f37cd。其候选上限2048题，生成后优先填满512混合题，不足补全对。筛后混合率不能与当前随机64直接比较。

| 窗口或时点 | 历史候选混合率 | 当前随机64混合率 |
| --- | ---: | ---: |
| 早期1、2、4、5 | 29.05%至31.84% | 不推算同窗口 |
| 91至100 | 21.9287%至21.9727% | 16.2500% |
| 101至108 | 20.3247%至20.3735% | 13.0859% |
| 141至150 | 16.8555%至16.8652% | 截止点后未分析 |
| 191至200 | 16.2451%至16.2646% | 截止点后未分析 |
| outer200 | 14.1113%至14.1602% | 截止点后未分析 |

早期为真实保存计数；后期由完整6至200日志中三位小数reward均值反推整数可行区间，不是新读取的候选计数或统计置信区间。旧G1也下降，因此混合下降并非probe独有。当前随机混合率更低，仍不能排除训练偏置。

历史日志固定512 EM约为outer50 40.8%、100 46.9%、150 48.0%、200 52.7%。旧与当前验证的逐题身份尚未在此记录完成绑定，只作背景，不作为严格配对或因果比较。

## 新goal补充的准确率与混合率脱钩

用户提供的第1至35轮分析包含17920个题次，使用生成前、当轮拟合前预测；10轮滚动Pearson相关如下。这组数值作为新goal的问题陈述，尚未在本记录中独立重算。

| 相关指标 | 第1至10轮 | 第26至35轮 |
| --- | ---: | ---: |
| 混合率与二分类accuracy | +0.9985 | −0.6885 |
| 混合率与六分类accuracy | −0.8870 | −0.5809 |

这支持将验收重心转向Top448命中、真实标准差、预测误差与排序；滚动相关本身不建立因果，也不替代同policy选题效果评估。

## 新实验的标准差目标

保留六输出及135622参数，将概率归并为三个标准差类别：

| K | 样本标准差 | 主类别 |
| --- | ---: | --- |
| 0、5 | 0 | 无差异 |
| 1、4 | 0.4472136 | 有差异 |
| 2、3 | 0.5477226 | 高差异 |

\[
q_0=p_0+p_5,\quad q_1=p_1+p_4,\quad q_2=p_2+p_3,
\]

\[
\hat\sigma=\sqrt{0.2}\,q_1+\sqrt{0.3}\,q_2,
\]

\[
L_{new}=CE_3(\min(K,5-K))+1.0\left(\frac{\hat\sigma-\sigma}{\sqrt{0.3}}\right)^2+0.1CE_6(K).
\]

CE3以logits成对logsumexp实现，避免小概率下溢。去掉均值MSE，实际梯度和报告loss共用实现。权重1.0和0.1为此次待验证起点。不同时增加容量、epochs、replay或排序loss；均值与六分类误差仅作诊断。

新actor从官方权重开始，重新初始化actor/head Adam、scheduler及私有RNG。已验收教师轨迹复制到独立远端目录并逐文件校验，复用真实标签重新拟合全部5120题，不复用旧head或训练checkpoint。旧实验曾恢复5轮断点且有RNG证据限制，两次执行不构成逐比特匹配的对照。

## 新实验验收和位置

主要指标为Top448真实混合率、相对随机64的百分点差与倍数、选中题真实平均标准差、标准差MAE/MSE，以及观测组内标准差成对一致率和混合排序AUC。预测在标签前冻结。单类时AUC缺失，不伪造零值；未生成候选不宣称oracle标签或全池排序质量。以窗口评估，避免随机64单轮噪声主导结论。

同时保留5、50、100、150、200固定512验证，最终原生51713题七来源EM/F1宏平均和加权结果；五个完整paired checkpoint逐文件SHA归档，最终实际四卡恢复actor/head/Adam/scheduler/RNG/data/replay/selector并验证相等。无update201。选题指标改善不替代actor效果验收。

- 新远端根目录：`/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015`。
- 新源码分支：`codex/qprobe-sigma3-loss-20261015`。
- 入口：`reproduction/hpc3_qprobe/sigma3_experiment.py`，沿用四卡train.sh，独立claim/日志。
- 教师来源：`/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718509-20261014/teacher-warmup-cache`。
- 模型、checkpoint、轨迹和环境只放远端；本地保留代码、文档和小型审计摘要。

## 证据清单

本地旧证据目录：`/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/qprobe-warmup-hpc3`。

| 文件 | SHA256 |
| --- | --- |
| probe-fit-analysis-through108-20261015.json | a4e0a5e5a4a1f58376e692de3209a740d6b42464f28f285acaf39bb51a1fcf2d |
| mixed512-actual1to108-diagnostic-20261015.json | c1d295e6f7c6b75287284d1a332b54c82d1da0921b87948251eae7ea386a5c18 |
| mixed-first-g1-versus-qprobe-bounded-comparison-20261015.json | e709f449fdcfd62a2c2053e2c48769d78524aba306982eb7f9812242b937fe99 |
| recovery719225-actual108-all-private-prefix108-before-probe-diagnosis.json | 9aae802ae702e3effe3413e73aa8a7d6122c20d01b7e28fae7a48ac073271a36 |

同目录有诊断PNG/SVG/CSV。新提交、部署SHA、两次实际进展、失败记录保存在独立`outputs/qprobe-sigma3-loss-20261015`，按真实状态更新。排队或初始化不等于有效训练推进。

## 人工切换至新goal

用户随后明确授权停止旧探针训练，仅继续新标准差loss目标。旧job719225的人工停止指令已绑定原账号、run、源码和作业入口后执行，记录为人工取消，禁止按故障自动恢复。停止前日志最后完整update为117；本文统计诊断仍固定截止108，不将117冒称完整独立验收。旧检查点、日志及已归档证据保留。

新job721449已经一次提交，源码aca0e779e94ddc3fbd50e49c3d59e2e889fbcf12；4 GPU、48 CPU、512 GiB，emergency_acd。提交阶段实际held child wait0、教师22文件完整SHA通过；提交后仍须核验实际actor/head迭代。总预算200，不因初期检查完成而停止。

721449随后在加载actor前因共享临时目录700权限导致全项目du核验失败（FAILED1:0，formal0）。原错误与终态保留至远端failed-job-721449。目录所有者按既有共享访问授权，仅将该临时目录组读/进入权限改为750；未改loss或模型。核对具体报错、原submission、完整源SHA、无actor/optimizer工作及目录du成功后，一次提交恢复job721485，仍运行同一aca0训练源码。此失败是启动存储门禁问题，不是新loss训练结果。

## 新损失的实际warmup（2026-10-15）

恢复作业721485在ACD1-21实际分配4卡，已通过初始模型和模型/检索共存检查。新缓存复用5120题、25600条教师轨迹，预测头实际完成40次Adam更新；容量仍为hidden64、135622参数。四rank actor参数、optimizer和scheduler冻结核验通过，warmup边界四rank RNG恢复相等。

| warmup拟合后指标 | 实测值 |
| --- | ---: |
| 三标准差类CE | 0.8169189692 |
| 按sqrt(0.3)归一化的标准差MSE | 0.1708564758 |
| 六类辅助CE（未乘0.1） | 1.3284107447 |
| 实际总loss | 1.1206165552 |

总loss与CE3+标准差MSE+0.1CE6相等；mean_mse仅作诊断，权重0。warmup-ready收据SHA256为0d4f1029289d7d7a8a0936798ff1cb8a66a39005a0c7ffbc24b78d1d7550ac62，head版本qprobe:0:0:dfcc10256ce0e802。此时正式第1轮已开始采样，完成计数仍为0，不作为正式iteration验收。旧719225保持人工停止。

已发布独立的新实验归档及最终native4跟进入口followthrough_sigma3，发布源码9c34720ecd564dfe902127d0167474d27b115cfe、全部1025文件远端SHA通过；它不会启动或改动训练，按实际配对checkpoint和最终200证据推进。训练源码继续固定aca0，不热改。独立raw审计入口发布0cca743462da769667262417f8064084f93fc912，须实际首轮完成后才允许一次启动。最终200、固定里程碑、51713评分、三副本SHA和实际恢复仍待完成。

## 新损失前五轮实际结果与第六轮启动

2026-10-15实际job721485在四卡上完成连续1至5轮，四rank各100次actor Adam写入，scheduler及预测头提交通过。第6轮已实际开始。前五轮共12800条实际训练轨迹，独立CPU子进程均实际wait0，canonical strict EM、选题前预测、私有trace、actor训练行及保留token重建一致。此为初期运行检查；目标仍为200轮，不在5轮停止。

| 轮次 | Top448混合数 | Top448命中率 | 随机64混合数 | 随机命中率 | 拟合后加权loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 196 | 43.7500% | 24 | 37.5000% | 1.092822 |
| 2 | 218 | 48.6607% | 17 | 26.5625% | 1.094450 |
| 3 | 216 | 48.2143% | 13 | 20.3125% | 1.100786 |
| 4 | 217 | 48.4375% | 13 | 20.3125% | 1.108376 |
| 5 | 232 | 51.7857% | 15 | 23.4375% | 1.137957 |

五轮合计Top448命中1079/2240（48.1696%），随机82/320（25.6250%），相差22.5446个百分点，倍数1.8798。Top448真实平均样本标准差0.238529，随机组0.126848；Top448标准差MAE0.244763、MSE0.062490。只对真实生成的512题评价，未生成的4608候选不宣称具有oracle标签。

在线拟合loss未单调下降：第1轮1.0928到第5轮1.1380。每轮policy、标签分布和recency加权replay均变化，不能把该数值直接解释为同一固定样本集的学习曲线，也不能据此宣称损失设计已验证有效。以上是五轮早期观测，无匹配随机种子重复对照，需继续看50/100/150/200的选题和actor结果。

同一run、同一固定512题（NQ256＋HotpotQA256）的真实验证：初始92/512 EM17.96875%，第5轮94/512 EM18.359375%，仅增加2题（+0.390625个百分点）；F1由23.73414468%到24.32425846%。固定题身份、源和逐行聚合已由verify_pair复核，没有额外生成或重复评测。初始/第5轮收据SHA分别为f3dc8a86c6b0c7f6903d12392dcc1adf9a30b29b3f2bd0a7977be1b1b14a142c、e55cc153f33e267c93e94776ae8d59028a7c62fab3ede38530a4dae67a93610f。

第5轮配对source收据SHA eb13a22502399c9cdbe458c5cbd73e8f07ebdfd9a347f2ef55508d3224d58339，完整依赖清单39个文件、44,550,414,490字节，包含模型/Adam/scheduler/RNG、预测头/replay/data/selector及奖励探针依赖。单个checkpoint目录约38.36GB，完整配对依赖约44.55GB；两者范围不同。独立归档已启动，三副本完整SHA和最终实际native4恢复仍待验收，不能把源收据当成恢复成功。

当前远端项目实际allocated bytes为1,397,052,772,352（只读du真实wait0；非原子统计，未核实硬配额）。完整200轮及所有副本的体量仍需持续复核，当前估计可能超过2TB预算。HPC3旧重复副本204.22GB清理清单已单独列出；用户现已明确批准删除这5个副本。执行前必须重新核验保留源文件完整SHA，不能扩展删除范围；此记录时仍未删除远端数据。该清理不会被混同为停训或改变新goal。

## CPU归档失败与独立恢复（2026-10-15）

新训练实际已完成第6轮，四rank各120次actor Adam写入，下一轮7采样已完成；第6轮探针loss1.129501，Top448混合221/448、随机20/64，独立raw审计1至6实际wait0通过。四卡训练源码aca0与目标损失均未热改。

原9c CPU归档第5轮在首次完整磁盘统计门禁处实际exit1，未开始复制，载荷0字节；原follower随后按设计退出1。原代码没有保留该次du stderr，故原始具体原因未知，不能归因于权限或训练算法。完整原claim、task.log、runner/task出生标识和wait1收据保留。修复版逐次保留du完整stdout/stderr；只对目录消失ENOENT作最多3次重新完整统计，必须某一次return0且stderr空才允许复制，权限等其他错误立即失败。

恢复使用独立、显式新源码和新控制目录；仅允许已验证的这一次CPU归档复制前失败，保全原证据后才移交，原训练、原follower目录和原科学配对收据不覆盖。最终独立CPU验收增加显式recovery pin，仍要求真实200及最终native4恢复；恢复已提交或进程存活都不代表三副本归档完成。新一次独立完整du诊断正在执行，归档恢复尚未激活。

后续两次完整诊断实际保留了具体错误：共享的CoSearch临时目录先有10个、随后另有9个MOU所有者的700目录不可读；第二次还遇到一次本实验process-ledger临时文件消失。依照既有共享目录授权，仅由原所有者对这19个已观测inode加组读/进入权限700→750，无递归、内容或训练修改。两次原始诊断失败均保留，未当作有效容量测量。第三次完整du实际wait0、stderr为空，于1792054374.0381303完成，allocated bytes为1,439,764,324,352；此为当时非原子测量，不是硬配额证明。

归档恢复已从独立发布的09ccff26c5b764cc4c6a285c1e8f57a8e9b07726一次激活，新控制目录为followthrough/721485-archive5-recovery-20261015。原8个失败归档证据文件同父原子迁移并核验inode及完整SHA一致；原9c follower的6个文件保持原位。新归档另做完整存储门禁，于1792054590实测复制载荷1,677,868,851字节（包含正在复制的partial），从0持续增加；完整三副本验收仍待真实子进程返回。未来独立最终CPU验收必须显式指定此09cc源码的followthrough和recovery pin，不能使用原EFA/9c正常路径替代。

已授权五个旧副本的清理程序发布为4c043aacfcedadfc696e87ff09fcb07fd31a63e8，11项本地和11项远端CPU测试通过，独立审查未发现P0/P1。实际持有CPU子进程进行完整SHA检查，1792054331累计32,189,333,294字节，1792054590推进至262,442,412,160字节，stderr为空；此记录时尚未开始删除。这些是校验累计读取量，不是释放空间。原始5/50/100、额外archive100和原运输证明都须保留，最终释放与清理后占用另以实际终态记录。

## 实际清理、归档和第10轮验收（2026-10-15约17:11 CST）

五个已授权旧副本目录已全部按精确文件清单删除，共204,223,139,168字节、200个文件。原始5/50/100依赖、第100轮额外archive和原运输证明删除前后完整SHA复核通过，实际持有子进程wait0、两端进程均结束、stderr为空。独立retired收据SHA为b7250db2e3a3a539a1a612e8ba68243a766a143564b2bf3f06e1dfc79849057a，实际终态结果SHA为598bfedcbd45d9e122a0b00035e58c826f96b6f4f966bbbbdc5ff2d72c0a87be。旧运输证明作为历史保留，另以本次收据注明相应副本已退役。

清理后独立完整du于1792055385.3160028实际wait0、stderr为空，测得项目allocated bytes为1,326,682,820,608（约1.327TB）；包含同时新写入的第5轮源/归档/目标及训练产物。不能用全项目两次差值直接当作删除量，因期间仍有新文件写入；204.22GB是实际删除文件的总长度，当前占用另由完整du测得。该测量不确认文件系统硬配额。

新第5轮归档CPU子进程已实际exit0，完整三方校验39个依赖文件、每份44,550,414,490字节；无partial残留。运输证明SHA为39e2edbaf0f92871aacb303011671a002e3ad2530fb43ab01959df134bb87b76，archive/target重定位收据SHA分别为16dcb2d53a0d73bd61cb7bb1c14e0d4ef958bd0a981f54b9c6fbd6f5c7bf1b61、9595ca9e481634422da4557794dd9176b2e3ce77fb485d861e0e1a31d3b161ed。新follower随后又独立重读三份完整SHA并通过，继续等待第50轮。此处证明完整复制与CPU文件校验，最终200的四卡实际model/Adam/scheduler/RNG/head/replay/data/selector恢复仍待执行。

新训练721485在1792055471.9246435已完整提交1至10轮、四rank各200次Adam写入，scheduler/head提交一致，第11轮已开始。第10轮head为qprobe:10:10:f48dab288e44c9b5，在线拟合loss1.171981；Top448混合211/448（47.0982%），随机15/64（23.4375%）。独立raw审计随后在1792055560确认1至10轮均实际wait0通过。没有新的致命错误、训练重启、容量或PPO/reward修改。目标保持ACTIVE，200、50/100/150/200固定验证、native51713和最终实际四卡恢复均不在此阶段提前宣称完成。


## 第12轮进展与指标口径补充（2026-10-15约17:31 CST）

在1792056503实际完成前11轮并完成第12轮采样之后，1792056672.358824再次核对，第12轮已完整结束，四rank各240次actor Adam写入，scheduler/head提交一致，第13轮已开始；没有新的致命错误。第12轮Top448混合226/448（50.4464%），随机64混合24/64（37.5000%），差12.9464个百分点，真实平均样本标准差分别为0.25028183和0.18655053。在线拟合总loss为1.18317616。随后1792056728独立RAW审计确认1至12轮均实际held child wait0，最新结果SHA4c15b64006f7dbbef82b378e1afce9cb65a2e57a25e181a7246eed692fe9d8ac，stdout SHAde02181429e6a94201cfda60dd821a111ca8a24346c9a694d7e595d753b2f044。

`head_optimizer_updates`是**本次fit**遍历replay所执行的Adam次数，不是累计Adam步数；例如第11轮10752条replay样本、两epoch、每批256，对应本次84次更新。不得把84解读为warmup40加累计正式44。预quential标准差MSE在原始σ尺度上，而拟合loss中的标准差MSE按sqrt(0.3)归一化；汇总时必须分别命名。

按当前完整du约1.327TB、前10个真实轨迹产物的字节增量以及实际checkpoint/HF依赖体积推算，保留所有200轮及后续完整副本的总量估计约1.9944TB（观察均值）至2.0014TB（观察最大值），尚未包含其他实验增长及不确定额外产物。这不是硬配额确认或保证；在50/100/150/200里程碑重新实测。已批准的五副本清理全部完成，本次授权不扩展为其他删除。

新增独立只读指标汇总入口`summarize_sigma3_progress.py`：从RAW0cca真正完成的held子进程收据消费成功前缀，复核scalar/head完整SHA、题目K分布与标准差数学一致、loss分解，分别汇总累计和最近10轮的Top448/随机64混合率、真实σ及误差。AUC和成对一致率只取各轮同policy内的已定义统计的均值，不能将变化policy混在一起算跨轮AUC。它不读取模型、不生成轨迹、不拟合、不提交任务、不写远端证据，也不替代最终全量RAW/实际native4恢复验收。当前仍为部分结果，目标ACTIVE。


## 独立累计汇总实际验收（2026-10-15约17:36 CST）

汇总入口以独立源码d9342e618cd2aa1615d8425337f71e3ea37678c7提交并推送，全部1046源码文件远端SHA通过；训练源码aca0、RAW0cca与恢复follower09cc保持原版本。针对收据/公式/原始K数学/组计数/guard前缀的测试通过。远端实际先消费前5轮held0，精确复现1079/2240与82/320，再消费1至12轮held0，两次CPU入口均实际return0。

| 同policy选题前预测汇总 | 前12轮累计 | 最近10轮（3至12） |
| --- | ---: | ---: |
| Top448混合题次 | 2627/5376 | 2213/4480 |
| Top448混合命中率 | 48.8653% | 49.3973% |
| 随机64混合题次 | 206/768 | 165/640 |
| 随机64混合命中率 | 26.8229% | 25.7813% |
| Top减随机 | +22.0424个百分点 | +23.6161个百分点 |
| Top/随机倍数 | 1.8218 | 1.9160 |
| Top真实平均样本σ | 0.24167787 | 0.24419910 |
| 随机真实平均样本σ | 0.13278109 | 0.12770383 |
| Top原尺度σ MAE | 0.24408806 | 0.24420273 |
| Top原尺度σ MSE | 0.06257084 | 0.06258676 |
| Top逐轮同policy混合排序AUC均值 | 0.54839588 | 0.55353080 |

这是已选并生成题目的题次统计，不能当成全5120候选oracle排序、唯一问题数或新loss优于旧loss的因果对照。前12轮Top真实混合比例高于随机组，但σ误差仍需继续观察；AUC只是在各当轮Top448已选组内计算再求均值，存在筛选后的条件范围限制。

第12轮拟合loss分解为CE3 0.8786897659＋归一化σMSE 0.1654541343＋0.1×CE6 1.3903232813＝1.1831761599，浮点绝对误差6.85e-8；meanMSE诊断0.1031696051仍权重0。本次replay11264题次，head本次fit88次Adam；不作为累计optimizer state步数。实际汇总证据为sigma3-progress-summary-max5-1792056952.json和sigma3-progress-summary-all-1792056954.json，绑定原始held结果/输出/head字节SHA。全量RAW的再次读取和最终四卡恢复仍单独待完成。


## 探针每轮拟合次数和回放口径（2026-10-15）

按实际运行manifest完整SHA ef830dd3ee84fdf34838c53730183d7e08a7801ad115aad662dedec54d379678，预测头batch为256、每次拟合2个epoch，回放窗口16轮，权重半衰期4轮，学习率3e-4。每个成功outer新增512条问题级标注，每条来自5条轨迹的二元EM正确数K及样本标准差；拟合在当轮actor完整更新后执行一次。最近16轮包含8192条标注记录，因此当前一次拟合的Adam更新次数为8192÷256×2＝64。新增512条标注在本轮各遍历2遍，其余batch来自历史回放；同一问题跨policy的标注计为不同题次，不把8192称为8192个全局唯一问题。

| 阶段 | 实际/既定回放题次 | 单次拟合Adam更新数 |
| --- | ---: | ---: |
| 冻结教师预热 | 5120 | 40 |
| 正式第t轮，1≤t≤15 | 5120＋512t | 40＋4t，即44至100 |
| 正式第16轮起 | 16×512＝8192 | 64 |

第15轮实际为12800题次、100次本次fit更新；第16轮实际为8192题次、64次，符合现有窗口规则source_outer＞outer−16。第16轮选题使用拟合前的head15，第16轮拟合产生的head16从第17轮选题生效。`head_optimizer_updates`始终是本次fit次数，不是累计Adam状态步数。每轮原始轨迹、标签和审计记录仍保留在HPC3；移出训练回放窗口不等于删除原始记录。教师预热5120×5轨迹另有独立缓存，可在身份和完整SHA一致时复用。


## 第20轮独立审计与分窗口统计（2026-10-15约18:47 CST）

job721485在1792061084.3213048实际完整完成1至20轮，四rank各400次actor Adam写入，scheduler/head提交一致，第21轮已开始。第20轮Top226/448（50.4464%），随机18/64（28.1250%），head版本qprobe:20:20:95f8e781f26ca9a2。独立RAW0cca于1792061171确认1至20轮全部实际held child wait0，最新结果SHA e0de3d6c19f1fd2d13fbb007e7fb77fed3602876cc8d7da8949d7243afa6bfcc、stdout SHA d38e7de2a8f766ea6da35c678b67a9ec8d41c7ed70fb344a2f3439f300ed8663。没有新的训练错误、重启或科学配置改动。

独立只读汇总入口d934以显式max_outer20消费已接受前缀，实际CPU return0于1792061194.7541783；全20轮原held收据/输出/head标量SHA再次复核。下表仅使用每轮选题前、拟合前预测与已生成512题的真实G5标签。1至10窗口由同一验收JSON前10轮按原汇总函数计算，11至20为实际recent10；比例以题次计。

| 窗口 | Top448混合题次/总题次 | Top命中率 | 随机64混合题次/总题次 | 随机命中率 | Top减随机 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1至10 | 2179/4480 | 48.6384% | 165/640 | 25.7812% | +22.8571个百分点 |
| 11至20 | 2281/4480 | 50.9152% | 199/640 | 31.0937% | +19.8214个百分点 |
| 1至20累计 | 4460/8960 | 49.7768% | 364/1280 | 28.4375% | +21.3393个百分点 |

| Top448指标 | 第1至10轮 | 第11至20轮 |
| --- | ---: | ---: |
| 真实平均样本σ | 0.24028907 | 0.25192944 |
| 原尺度σ MAE | 0.24350039 | 0.24437960 |
| 原尺度σ MSE | 0.06200143 | 0.06393745 |
| 逐轮同policy已选Top448混合排序AUC均值 | 0.55288092 | 0.54175487 |

后10轮Top命中率比前10轮高2.2768个百分点，同时随机组也高5.3125个百分点，故相对随机的提升由22.8571缩至19.8214个百分点。σ MAE数值基本持平，已选组内排序AUC均值略降；不能把Top比例上升直接解释为新loss已优于旧loss。policy、回放及标签分布持续变化，前20轮仍是早期观察。未生成候选不宣称有oracle标签；AUC只在各轮同policy、已选Top448内部计算后求均值。下一次固定actor验证是第50轮，没有额外第20轮评测。

第20轮拟合loss1.2378621101＝CE3 0.9161176682＋归一化σMSE 0.1686498076＋0.1×CE6 1.5309479237，公式误差1.58e-7；meanMSE诊断0.1268529743仍权重0。实际回放8192题次、2个epoch、batch256、本次64次Adam更新。观察中的原始σ误差与拟合归一化MSE保持区分。证据为sigma3-progress-summary-max20-1792061194.json；本地小型派生记录sigma3-window-comparison-through20.json绑定其完整SHA。

归档跟进的原09cc进程在1792061181.6955667仍实际存活，第5轮完整三份SHA验收已通过，等待第50轮；最终native4恢复尚未执行。训练和RAW源继续固定aca0/0cca，目标保持ACTIVE：200轮、固定50/100/150/200、最终native51713、全部里程碑完整三副本SHA、实际四卡恢复和独立最终CPU验收均仍待完成。


## 第30轮独立审计与窗口变化（2026-10-15约20:17 CST）

job721485在1792066509.7976224实际完整完成1至30轮，四rank各600次actor Adam写入，scheduler/head提交一致，第31轮已开始。第30轮Top222/448（49.5536%），随机22/64（34.3750%），head版本qprobe:30:30:97df7f5cd8682df1。RAW0cca在1792066588实际确认1至30轮全部held child wait0，最新结果SHA d28c7bb385694ae7bccf6f1de006c4571c83dc991c4c20f2ee9cc3e4f5e9e6c0、stdout SHA e57ee2bb018af7d69c1810e97eb05bd5a5b1618033967009417ea9df966315f8。第28至30轮均以采样、四卡实际写入增量、完整提交和独立RAW验收依次确认，没有重启、热改或科学配置变化。

原独立只读汇总源码d934显式max_outer30于1792066604.6812909实际CPU return0，再次复核已接受前缀的原held收据/输出和scalar/head完整SHA。以下比较为两个不重叠的10轮窗口，均使用当轮选题前、拟合前预测与真实G5标签；累计按题次而非全局唯一题计。

| 窗口 | Top448混合题次/总题次 | Top命中率 | 随机64混合题次/总题次 | 随机命中率 | Top减随机 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第11至20轮 | 2281/4480 | 50.9152% | 199/640 | 31.0937% | +19.8214个百分点 |
| 第21至30轮 | 2043/4480 | 45.6027% | 195/640 | 30.4688% | +15.1339个百分点 |
| 第1至30轮累计 | 6503/13440 | 48.3854% | 559/1920 | 29.1146% | +19.2708个百分点 |

| Top448指标 | 第11至20轮 | 第21至30轮 |
| --- | ---: | ---: |
| 真实平均样本σ | 0.25192944 | 0.22433483 |
| 随机组真实平均样本σ | 0.15114796 | 0.15008037 |
| 原尺度σ MAE | 0.24437960 | 0.24337850 |
| 原尺度σ MSE | 0.06393745 | 0.06258206 |
| 逐轮同policy已选Top448混合排序AUC均值 | 0.54175487 | 0.54458396 |

最新10轮Top命中率较上一窗口下降5.3125个百分点，随机组下降0.6250个百分点；相对随机的差距由19.8214缩至15.1339个百分点。Top真实σ随之下降，σ MAE基本持平。当前证据显示选题相对增益收窄，尚不能据此判定过拟合、欠拟合或具体因果。policy和标签分布持续变化，在线拟合loss不是同一固定数据集的学习曲线；没有修改loss、回放、探针容量、PPO或奖励。未生成候选无oracle标签；AUC仅在各轮同policy、已选Top448内部计算后求均值，不是跨policy合并AUC。

第30轮拟合loss1.2211579084＝CE3 0.8964479566＋归一化σMSE 0.1704084575＋0.1×CE6 1.5430129766，公式误差1.97e-7；meanMSE诊断0.1447500587仍权重0。回放8192题次、batch256、2个epoch、本次64次Adam更新。证据sigma3-progress-summary-max30-1792066604.json完整SHA cf45299b911c1ed528138ba20ccbeebfe43b1a9bb92f3d81f76d92e82a1b3f58；本地派生记录sigma3-window-comparison-through30.json同时绑定本次和第20轮原汇总完整SHA。

归档跟进09cc原进程在1792066608.0689752实际存活，第5轮三份完整SHA已验收，仍等待第50轮；最终native4恢复尚未执行。下一次固定actor验证为第50轮，不补做第30轮评测。目标保持ACTIVE：200轮、固定50/100/150/200、最终native51713、全部里程碑三副本完整SHA、实际四卡恢复和独立最终CPU验收均仍待完成。


## 登录节点身份绑定修复（2026-10-15约21:02 CST）

第35轮训练已实际完整结束，四rank各700次actor Adam写入，第36轮继续。1792068972/9003两次原RAW检查返回UNKNOWN_NO_REPLAY：当时查询落在ACD-Login-2，找不到原PID1171759；共享state却持续更新并启动第35轮审计child。没有故障终态或退出证据，未重启、重放或热改任何训练/审计进程。诊断保留在raw721485-unknown-birth-diagnostic-1792069045.json；网关10.120.81.30对应ACD-chess，实际登录节点映射由该集群/etc/hosts和DNS核对。

通过现有StrictHostKeyChecking=yes、原HostKeyAlias和原knownhosts，明确连接Login-1（10.120.48.31），1792069300.606421实际找到RAW1171759/start75909729及归档follower2688346/start76403669，PID/SID/UID、command和cwd均与原0cca/09cc一致，状态S。该主机boot_id为a27512a3-24a0-4ca3-aa92-98d2a5256d9a、PID namespace为pid:[4026531836]。原不可变receipt保持原样，另以login1-bound-birth-diagnostic-1792069300.json保存主机绑定实证；当时共享RAW状态登记35轮完成，完整consumer仍另作复核。

独立审查指出：在另一主机上/proc缺失不能证明进程死亡。local_account_bridge新增显式固定节点白名单，默认网关行为不变；可传已观测的hostname/boot_id/pid_namespace，远端在运行请求命令前逐项匹配。主机、启动实例或PID namespace不同即返回UNKNOWN_HOST_IDENTITY_MISMATCH/exit86，不执行请求命令，也不新增或放宽SSH主机信任。后续本任务RAW、归档、指标和最终CPU进程检查均绑定原Login-1身份；连接失败或身份不符不作为重启授权。

相关本地隔离测试覆盖默认入口不变、固定节点仍严格校验、shell/stdin/退出语义、非法endpoint/账号/身份在读取凭据前拒绝，以及hostname/boot/PID namespace任一不符时禁止exec；不访问真实凭据或远端。训练ACA、RAW0cca、归档与最终恢复09cc、指标d934保持原版本；该修复只处理本地控制连接及观测范围。

修复实现4262959e12dc16be3021d23e9555a6f29a5207e1已提交推送，1792069453.8346548远端1047文件完整SHA通过；6项本地针对性测试通过，独立审查未发现新增P0/P1。独立主机绑定文件SHA为0d28e1c45c2b735e9f68d660dad7d5325043f08a79b2ea0038e05b5f3107a20c。实际带身份guard的原RAW consumer查询于1792069487确认35轮held0、1792069583确认36轮held0；最新RAW36结果SHA 051976c307a89dc39cdc65e614901283cb4866e6d2a1656ff34a9548587e1538、stdout SHA 1441c08e36d4a00acde7dc2a82b398c08b07bb1acdb4f79e1dc1aebbafd9e0b2。训练在1792069484.0813463完整完成36轮、四rank各720次Adam写入，第37轮已开始；同次身份guard下归档follower实际存活，继续等待50。此次为观测范围修复，没有训练或RAW故障重启，完整200目标及最终验收仍未完成。


## 第40轮独立审计与窗口变化（2026-10-15约21:38 CST）

job721485在1792071362.64187实际完整完成1至40轮，四rank各800次actor Adam写入，scheduler/head提交一致，第41轮已开始。第40轮Top185/448（41.2946%），随机16/64（25.0000%），head版本qprobe:40:40:ab5160886b977533。RAW0cca于1792071449确认1至40轮全部实际held child wait0，最新结果SHA 7800bcef6f12e6bae5e57c0f4b053b019e116884856eb98566bce1bb2f2e9223、stdout SHA c6d706c1f160b00b71fff425004c4a9a9073d8644bcdb0bd7dfbb3882ac34246。查询明确绑定原Login-1的hostname、boot_id和PID namespace，没有训练/审计重启或科学配置变化。

原独立只读汇总源码d934显式max_outer40于1792071463.8777702实际CPU return0；原held收据/输出及scalar/head完整SHA再次复核，并附本地主机绑定证明。下表两个10轮窗口互不重叠，均使用当轮选题前、拟合前预测与真实G5标签；仍是部分成功前缀统计，不替代最终全量RAW再读及native4恢复。

| 窗口 | Top448混合题次/总题次 | Top命中率 | 随机64混合题次/总题次 | 随机命中率 | Top减随机 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第21至30轮 | 2043/4480 | 45.6027% | 195/640 | 30.4688% | +15.1339个百分点 |
| 第31至40轮 | 1832/4480 | 40.8929% | 182/640 | 28.4375% | +12.4554个百分点 |
| 第1至40轮累计 | 8335/17920 | 46.5123% | 741/2560 | 28.9453% | +17.5670个百分点 |

| Top448指标 | 第21至30轮 | 第31至40轮 |
| --- | ---: | ---: |
| 真实平均样本σ | 0.22433483 | 0.20024313 |
| 随机组真实平均样本σ | 0.15008037 | 0.13895476 |
| 原尺度σ MAE | 0.24337850 | 0.24136516 |
| 原尺度σ MSE | 0.06258206 | 0.06148333 |
| 逐轮同policy已选Top448混合排序AUC均值 | 0.54458396 | 0.54837573 |

最新10轮Top命中率较上一窗口下降4.7098个百分点，随机组下降2.0313个百分点；相对随机的差距由15.1339缩至12.4554个百分点，Top/随机倍数由1.4967降至1.4380。Top真实平均σ下降，原尺度σ MAE仅小幅下降约0.0020，未转化为更高混合命中率。在线拟合loss由第30轮1.2212降至第40轮1.1731，但policy、标签及recency加权replay同时变化，不能将其当作固定数据集的loss下降或据此判定过拟合/欠拟合。仍无新旧loss的匹配重复因果对照；未生成候选无oracle标签，已选组内AUC也不代表全候选排序。此次仅记录观测，没有加入排序loss或改变探针容量、回放、PPO或奖励。

第40轮拟合loss1.1730874777＝CE3 0.8554528952＋归一化σMSE 0.1663975567＋0.1×CE6 1.5123704672，公式误差2.09e-8；meanMSE诊断0.1597543806仍权重0。回放8192题次、batch256、2个epoch、本次64次Adam更新。证据sigma3-progress-summary-max40-1792071463.json完整SHA 15a419e2ed2f17eb0072227e7f5f5c5cedfe2c7a0d6c849a3d1f6496b6eacf47；本地sigma3-window-comparison-through40.json同时绑定本次及第30轮汇总完整SHA。

原归档follower09cc在1792071373.3764284经同一主机身份guard确认实际存活，第5轮三份完整SHA已验收，仍等待第50轮。下一次固定actor验证、完整项目存储实测和新三副本验收均在第50轮；不额外生成第40轮固定评测。目标保持ACTIVE，完整200轮、最终native51713、所有里程碑完整三副本SHA、实际native4恢复、独立最终CPU验收及no201仍待最终证明。


## 2026-10-15 23:12 CST actual Login1 accounting outage / CPU recovery preparation

Login1 Munge socket/accounting service failed at about22:45CST. Original RAW0cca supervisor1171759/start75909729 stopped after authentic held0 audits1..49 with exact sacct CalledProcessError at1792075504.6165059. Original09cc follower2688346/start76403669 stopped with exit1 at1792075543.145685 after accepted archive5. Both original terminal result/state/claim/output remain in place; samehostname/boot/PIDnamespace proves both births dead. This is distinct from prior wrongLogin2 UNKNOWN observation. Small diagnostic munge-service-diagnostic-1792075742.json binds the actual errors and deaths; service later recovered.

Training721485 ACA continued untouched: at1792076968.4465215 complete1..51, four ranks each1020 Adam writes with matching scheduler/head;52started, nofatalerror. Step50 fixedvalidation/checkpoint executed but exactscore/fullarchive still pending. Typed once CPU continuation is implemented locally and68 targeted/regression tests passed; independent review ongoing, not yetpublished orstarted. OriginalRAW49/archive5/legacyrecovery evidence remains immutable; newRAW onlysuffix50onward, newfollower carries accepted5 and claims onlynew50+components. Futurefinalreader/holder requires explicit observer_recovery_commit plus original09cc recovery pin. Readonlyscheduler transport failures wait as UNKNOWN, while successful invalidowner/state/source still fail. No training restart, hotedit, extraeval, scientificchange, or newdeletion. GoalACTIVE; all200/finalnative4/fullSHA/finalheldCPU/no201 remain required.


Independent read-only review by existing qprobe_launch_review found no remaining confirmed P0/P1 in this typed service recovery; private cross-source bindings, explicit final pins, original49/new suffix and archive5 lineage checked. Local68 tests passed. Full positive old49/new151/archive5/newnative combination still requires actual200 final acceptance; no partial progress is substituted. local_publish now accepts explicit existing Login1 host/boot/PIDnamespace binding, preserving default behavior.


## 2026-10-15 23:14 CST CPU service recovery activated and fixed50 scored

Recovery source f979d53cdb0d348e87c5877d84403252e85cd784 pushed and1052 files remotely fullSHA verified at1792077108.0338483. NewRAW3857750/start78675791 andnewfollower3861035/start78678649 started exactlyonce onboundLogin1; old49/archive5/oldcontrollers remain unchanged. Authentic RAW50 held0 accepted by1792077223.9461803;51childrunning. Newfollower accepted retainedarchive5, claimed newarchive50 once, actualrunnerlive butpayload0 at1792077246.610543, not archiveacceptance. Native complete52/all4Adam1040 at1792077196.6291971,53started, nofatalerrors.

Fixed512 step50 exactorderedidentity/sourceSHA/run checked: EM185/512=36.1328125%, F1=44.534195188492065%; NQ96/256=37.5%, F1=48.44990079365079%; Hotpot89/256=34.765625%, F1=40.618489583333334%. OriginalledgerSHA9d14949bf46c6efa17e4f258b641364b9b3478801287930da7cbfadf2a53c02c androwSHAf4ebe095ab8ebe0aab075dc19796b1c9b9776c7b8ff7c1edfb6c50906224fadb; sourcepair50SHAc149938fe61b4ffb9ae2a0d746d921369ddf058ced65d254f4f14b8da4fe23e1. No newgeneration/modelreload performed for scoring.

Explicit recovered metricsreader fullSHA/held graph reconsumed1..50 at1792077258.6292398: latest41..50 Top1719/4480=38.3705357%, random147/640=22.96875%, difference15.4017857pp. CumulativeTop10054/22400=44.8839286%, random888/3200=27.75%. This remains partial evidence; final200/fullnative/fullRAW/finalheldCPU pending.

Step50 fullproject metadata du firstdiagnostic used systemPython3.6 incompatiblecapture_output; subsequent compatible readonlywalk timedout40s andownedmeasurementchild was killed by subprocess timeout cleanup. Both failures retained, no usage inferred. New once-only outer50 adapter preserves existing canonical helddu/source/birth/exit/SHA checks with1800s detachedmeasurement, uniqueclaimdirectory. It changes only its entry filename/controlnamespace; no deletion or trainingcontrol. Hardquota remainsunknown; last actualfullDU still1,326,682,820,608B at1792055385.3160028 until newwalk returns0.


## 2026-10-15 23:20 CST actual archive50 pre-copy failure

Independent outer50du a3 returned1 at1792077574.0235565 withfour exact Permission denied MOU21930/GID21919 mode700 cache/tmp directories. Its stdout1,404,336,775,168B is incomplete and not acceptedusage. Originalf979 archive50 actual heldchildexit1, runner/taskbothdead and payload0; retainedtask.log independentlyrecords its ownfaileddu (started1792077218.6207843, completed1792077574.0235293) with samefour errors andincomplete1,404,321,161,216B. OriginalclaimSHA4fca977abc89dea7aac22f702a3fa0e916c48b662e55f2a076303de077cbabfd andalllogs/receipts stayinplace, neverblindlyreplayed. Training53/all4Adam1060 remainshealthy; RAWf979alive.

MOUreadonly metadata at1792077659.9863226 verified exactfourinodes/device49/mode700, sameparent/sameprior explicitlyauthorizedsharedgroup scope. Newpublishedrepair onlyadds groupread/traverse700→750 viaheldfds andmatchingowner/inode, no recursion/content/trainingactions. CanonicalnewdetachedDU uniquepermission-repaired namespace willmeasurefullownerusage once; previousfailure kept. Anyarchive50 recovery separatelyrequires originalfollower actualterminal pluspreserved heldfailure/no payload; newRAW mustcontinueoriginalf979, finalreaderrequires explicit expandedlineage/all200/no201.


## 2026-10-15 23:32 CST archive50 recovery implementation and repaired usage

Exactfourpermission repair d7f61cec1c27581137584fa4646ac63e89743b7b completed1792077855.1802468; actualfullowner DU process3960561/child3960562 returnedheld0 at1792077963.997741, stderr0, allocated1,405,047,443,456B. This fresh non-atomic measurement satisfies own2TB admission; hardquota stillunknown. PreviousinvalidpartialDU numbers remain labeledunaccepted.

Typedarchive50 recovery fixes originalf979 claimSHA4fca977abc89dea7aac22f702a3fa0e916c48b662e55f2a076303de077cbabfd, exactheld1/deadcontroller+runner+task, exacttasklogSHA8d0750d76083c6e81be6dce8941e0ee4dc583bf6159fad523c191fb114816296, actualpair50 SHA c149938fe61b4ffb9ae2a0d746d921369ddf058ced65d254f4f14b8da4fe23e1, and complete absence of anypayload. Only failedarchivecontrol is atomically preserved once, oldfollower staysinplace, oldRAWf979 is untouched and mustremainhealthy. Newfollower willreuse accepted5 and claimonly50+; everyarchive stillindependently doesfreshdu+own2TBgate+triplefullSHA. Newintent/partial/unknown neverreplayed.

Finalreader/holder explicitly separates archive_recovery_commit=NEW/followthrough_commit=NEW, observer_recovery_commit=f979d53cdb0d348e87c5877d84403252e85cd784 for original49+newRAWsuffix, recovery_commit=09ccff26c5b764cc4c6a285c1e8f57a8e9b07726 for archive5. Scope/claim/command/stdout allbind the additionalpin. Legacyroutes retained; all200/native4/two readbacks/rawnormalattempts/allfivefullSHA/no201 unchanged. Nine newtests cover authentic originalsmallreceipt/log byteSHA, wrong/live/partial failure refusals, realatomicpreservation and secondstartrefusal/migrationtamper, freshheld0DU and explicitsourcechains. Fullcombined200actualacceptance remains pending, not inferred fromfixtures.


## 2026-10-15 23:39 CST second observed archive50 pre-copy permission failure

Originaldc9 archive50 claim8d79166547f70259369b37d18b80bd33c17b8177b6bda231378c3397ef496587 returnedheld1 at1792078512.3364935 beforeanypayload. Task.logSHA81276ebf2457200eb29048293af60ef2a10a79ae53c8bb61c9c66a58ec1ef12e shows freshDUfailure on two newlycreated/sharedMOU feature dirs of anotherexperiment, notthe four repairedpreviously. dc9follower4037307/start78801800 actualexit1 at1792078542.5607188. Originalclaim/logs/waits/follower remaininplace, previousf979preservation remainsunchanged. Native55/RAW55 stillhealthy, no training/source/RAWrestart.

Fresh MOU metadata1792078780.7602317 identifiesexacttwoUID21930/GID21919 mode700 directoryinodes9223372092952108439/9223372092952976648 underqprobe-teacher-warmup200-native-thread5-20261015. Preparedrepair implements existing shared-group authorization onlyviaheldfd700→750 matchinginode/owner/path; execution stillrequires publishedsource andactualreceipt. No recursivechange/content/deletion/agentmessage ispermitted. Newunique onceDU willmeasurefresh fullownerusage; prior1.405TB measurement remainshistorical, latestinvalid1.407769903104TB du isnotaccepted. Secondtypedarchive50layer mustprivatelyreconsumedc9firsthandoff andretainallpriorfailure/RAWf979/09cc archive5 lineage; currentclaims mustnotreplay.


## 2026-10-16 每轮拟合后二分类、六分类与方差诊断（outer0–58）

用户本次要求训练集逐轮准确率与方差拟合，不展示召回率。以下均为拟合后指标，区别于每轮selection的拟合前预测。训练job721485、run 70827e85fe584f0c84453aae907918a4、ACA科学源码、native4及loss/容量/回放/PPO/奖励均未改变。

原实现只保存逐轮拟合loss，完整head保存于0/5/50。通过独立CPU进程读取本次warmup model/Adam/private RNG与真实每轮features/strict EM，重算历史小探针更新：1–50每轮版本及全部loss分量一致，5/50完整model、Adam、RNG、replay、协议/计数逐tensor完全相等。单线程重放第1轮hash不一致，其结果已拒绝；使用原CPU线程数8通过（CPU线程不是GPU卡数）。51–58从原始完整head50开始，逐轮原版本和loss一致，但没有新的完整状态终点锚，图中灰区标明较弱尾部验证。未加载actor、生成新轨迹、改动或重启在线训练。

当前轮数据是本轮新512题；完整训练集是本轮实际拟合的replay，0轮5120题，1–15轮为5120+512×outer，16轮起为最近16轮共8192题次。主表为无年龄加权的准确率/误差；实际fit损失使用半衰期4轮的年龄权重，完整加权诊断也保存在JSON。

定义：二分类以sum(p1..p4)>=0.5预测K=1..4；六分类为argmax(p0..p5)。另存六类argmax后合并的二分类准确率，不与前者混用。真样本方差v=K(5-K)/20，预测v=sum(p_k v_k)；真σ=sqrt(v)，预测σ=sum(p_k sqrt(v_k))，预测方差不等于预测σ平方。RMSE越低越好；R²=1−MSE/同一数据集目标方差，0对应同一数据集常数均值基准。日志std_mse是归一化σ MSE，乘0.3只恢复原尺度σ MSE，不能当方差MSE。

主要观察：完整训练集六分类先降后升，须结合warmup退出和K分布改变判断；第58轮46.5576%高于同集多数类32.7637%。二分类70.9473%高于多数类65.4785%。完整训练集方差R²约0.15，当前512仅0.03118，仍只有有限拟合信号；当前512二分类74.2188%仅高于多数类72.6563%约1.5625个百分点。不能将accuracy或loss变化直接视为方差预测有效，也不能仅凭训练指标归因于容量或过拟合。

数据与图片：sigma3-postfit-training-metrics-outer0-58.json（SHA256 199778af022cd07a6dba24b86f88358f2dc8d91bf6368368ae6edb07b3e02aa9）；sigma3-postfit-accuracy-variance-outer0-58.png及-fit-detail.png。诊断代码61ea40f/9ccd305均已推送并远端完整SHA；1–50完整证明与51–58尾部证明保存在独立本地小审计JSON。

| outer | replay题次 | replay二分类% | replay六分类% | replay σ RMSE | replay方差RMSE | replay方差R² | 当前题二分类% | 当前题六分类% | 当前题方差RMSE | 当前题方差R² |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5120 | 69.8047 | 66.5234 | 0.226400 | 0.114743 | 0.006795 | 69.8047 | 66.5234 | 0.114743 | 0.006795 |
| 1 | 5632 | 68.9631 | 65.0746 | 0.222833 | 0.113163 | 0.059111 | 58.3984 | 50.5859 | 0.124477 | 0.041587 |
| 2 | 6144 | 69.2871 | 63.8184 | 0.220762 | 0.112281 | 0.096336 | 60.9375 | 49.8047 | 0.123524 | 0.071254 |
| 3 | 6656 | 69.7716 | 62.7103 | 0.219262 | 0.111563 | 0.120232 | 61.5234 | 49.4141 | 0.122800 | 0.040932 |
| 4 | 7168 | 70.5218 | 61.6490 | 0.218071 | 0.111094 | 0.141328 | 65.8203 | 47.6562 | 0.123209 | 0.078802 |
| 5 | 7680 | 68.5938 | 60.6120 | 0.220968 | 0.112671 | 0.131464 | 59.3750 | 45.8984 | 0.124301 | 0.079569 |
| 6 | 8192 | 71.0205 | 59.9121 | 0.216243 | 0.110400 | 0.173854 | 62.5000 | 47.2656 | 0.121395 | 0.081447 |
| 7 | 8704 | 69.0142 | 59.2716 | 0.218380 | 0.111389 | 0.159813 | 64.6484 | 48.0469 | 0.114369 | 0.115893 |
| 8 | 9216 | 71.4627 | 58.4744 | 0.215318 | 0.110048 | 0.189885 | 71.0938 | 42.7734 | 0.119772 | 0.156367 |
| 9 | 9728 | 69.9836 | 57.7611 | 0.216791 | 0.110891 | 0.184484 | 63.4766 | 43.5547 | 0.121248 | 0.118623 |
| 10 | 10240 | 68.2715 | 57.0898 | 0.219665 | 0.112215 | 0.167826 | 65.0391 | 45.5078 | 0.116947 | 0.140337 |
| 11 | 10752 | 71.2612 | 56.6406 | 0.215198 | 0.110165 | 0.203479 | 65.0391 | 41.4062 | 0.121905 | 0.115483 |
| 12 | 11264 | 70.8807 | 56.1701 | 0.215433 | 0.110385 | 0.205396 | 59.3750 | 42.5781 | 0.125351 | 0.058374 |
| 13 | 11776 | 70.6352 | 55.4942 | 0.216380 | 0.110931 | 0.201904 | 64.6484 | 43.3594 | 0.124461 | 0.086146 |
| 14 | 12288 | 70.5648 | 55.0212 | 0.215965 | 0.110738 | 0.208347 | 62.6953 | 38.4766 | 0.122032 | 0.085127 |
| 15 | 12800 | 69.1953 | 54.2109 | 0.218120 | 0.111930 | 0.193973 | 63.2812 | 39.8438 | 0.118320 | 0.144513 |
| 16 | 8192 | 67.9443 | 45.9961 | 0.223277 | 0.115099 | 0.191475 | 66.9922 | 38.6719 | 0.117509 | 0.158960 |
| 17 | 8192 | 67.1143 | 44.5679 | 0.224029 | 0.115547 | 0.186510 | 64.0625 | 37.1094 | 0.122128 | 0.099640 |
| 18 | 8192 | 67.4072 | 43.9209 | 0.223143 | 0.114987 | 0.193404 | 66.6016 | 37.8906 | 0.117299 | 0.146586 |
| 19 | 8192 | 67.5781 | 43.0542 | 0.223203 | 0.115093 | 0.191990 | 66.2109 | 39.0625 | 0.117360 | 0.128274 |
| 20 | 8192 | 66.2476 | 42.4438 | 0.223974 | 0.115459 | 0.184635 | 60.9375 | 38.0859 | 0.119508 | 0.095899 |
| 21 | 8192 | 67.1387 | 42.1143 | 0.223208 | 0.115015 | 0.190960 | 62.1094 | 37.1094 | 0.124609 | 0.076351 |
| 22 | 8192 | 66.6504 | 41.4551 | 0.224480 | 0.115667 | 0.180964 | 63.0859 | 40.8203 | 0.121208 | 0.059002 |
| 23 | 8192 | 64.5142 | 41.2842 | 0.226909 | 0.116937 | 0.166903 | 59.7656 | 41.0156 | 0.123525 | 0.051007 |
| 24 | 8192 | 66.6260 | 41.0034 | 0.224694 | 0.115662 | 0.179507 | 64.8438 | 39.0625 | 0.115720 | 0.109574 |
| 25 | 8192 | 67.4438 | 40.7593 | 0.223574 | 0.115000 | 0.186033 | 61.5234 | 36.5234 | 0.119730 | 0.089056 |
| 26 | 8192 | 66.1499 | 39.7827 | 0.225207 | 0.115792 | 0.174489 | 61.1328 | 37.1094 | 0.119969 | 0.087511 |
| 27 | 8192 | 65.9302 | 40.0269 | 0.225674 | 0.116012 | 0.168879 | 56.2500 | 34.7656 | 0.123904 | 0.042193 |
| 28 | 8192 | 65.5029 | 39.5386 | 0.226096 | 0.116187 | 0.162606 | 64.6484 | 39.4531 | 0.118177 | 0.098550 |
| 29 | 8192 | 62.2314 | 38.5986 | 0.230452 | 0.118076 | 0.130150 | 65.2344 | 36.7188 | 0.117073 | 0.108885 |
| 30 | 8192 | 66.1133 | 39.1968 | 0.224953 | 0.115222 | 0.169335 | 62.3047 | 33.9844 | 0.120544 | 0.077088 |
| 31 | 8192 | 65.9058 | 38.5010 | 0.224807 | 0.115080 | 0.168547 | 61.9141 | 31.6406 | 0.119757 | 0.076769 |
| 32 | 8192 | 66.3452 | 38.2446 | 0.224705 | 0.114979 | 0.167758 | 58.3984 | 38.8672 | 0.122979 | 0.045262 |
| 33 | 8192 | 65.2100 | 37.7197 | 0.225746 | 0.115233 | 0.157359 | 65.6250 | 41.0156 | 0.117503 | 0.045668 |
| 34 | 8192 | 66.0767 | 37.7686 | 0.224767 | 0.114750 | 0.159127 | 62.6953 | 43.1641 | 0.115835 | 0.077479 |
| 35 | 8192 | 64.3677 | 37.6587 | 0.227339 | 0.115924 | 0.138375 | 63.4766 | 38.6719 | 0.116390 | 0.084367 |
| 36 | 8192 | 67.1143 | 37.5732 | 0.224282 | 0.114369 | 0.157396 | 67.5781 | 41.9922 | 0.116039 | 0.079403 |
| 37 | 8192 | 65.6128 | 38.6230 | 0.225847 | 0.114896 | 0.141957 | 67.3828 | 46.0938 | 0.115979 | 0.073053 |
| 38 | 8192 | 66.9434 | 37.0972 | 0.225096 | 0.114497 | 0.146124 | 63.4766 | 37.3047 | 0.119239 | 0.057412 |
| 39 | 8192 | 65.9180 | 38.2568 | 0.224280 | 0.113887 | 0.150393 | 62.8906 | 38.2812 | 0.118294 | 0.043674 |
| 40 | 8192 | 66.4795 | 39.2700 | 0.224421 | 0.114040 | 0.148520 | 67.7734 | 40.8203 | 0.116368 | 0.105457 |
| 41 | 8192 | 66.3086 | 39.0015 | 0.224714 | 0.114268 | 0.143828 | 66.0156 | 39.8438 | 0.117666 | 0.102433 |
| 42 | 8192 | 66.5039 | 39.3311 | 0.224142 | 0.113981 | 0.143994 | 66.6016 | 41.7969 | 0.115791 | 0.071708 |
| 43 | 8192 | 67.7612 | 40.5029 | 0.221969 | 0.112848 | 0.155820 | 67.3828 | 41.9922 | 0.116185 | 0.080774 |
| 44 | 8192 | 67.1997 | 40.8447 | 0.222396 | 0.113058 | 0.151388 | 67.7734 | 44.7266 | 0.115988 | 0.114164 |
| 45 | 8192 | 66.6870 | 42.2241 | 0.224237 | 0.113988 | 0.132344 | 70.7031 | 43.9453 | 0.112674 | 0.081489 |
| 46 | 8192 | 67.8345 | 42.6880 | 0.221050 | 0.112398 | 0.152444 | 64.8438 | 40.2344 | 0.117041 | 0.092143 |
| 47 | 8192 | 68.7500 | 44.2017 | 0.219513 | 0.111682 | 0.160703 | 66.9922 | 43.5547 | 0.116934 | 0.087844 |
| 48 | 8192 | 69.4214 | 44.5312 | 0.218299 | 0.111079 | 0.166428 | 66.6016 | 41.9922 | 0.116166 | 0.103531 |
| 49 | 8192 | 69.6533 | 44.8975 | 0.218192 | 0.111071 | 0.169104 | 64.0625 | 43.1641 | 0.119177 | 0.066337 |
| 50 | 8192 | 68.6401 | 44.7144 | 0.219309 | 0.111741 | 0.161873 | 68.1641 | 43.1641 | 0.117890 | 0.095041 |
| 51 | 8192 | 68.7134 | 45.2393 | 0.219004 | 0.111731 | 0.158364 | 70.3125 | 46.6797 | 0.112691 | 0.066340 |
| 52 | 8192 | 68.0786 | 45.2515 | 0.221318 | 0.113000 | 0.139975 | 68.3594 | 44.5312 | 0.117741 | 0.066600 |
| 53 | 8192 | 69.5312 | 44.8853 | 0.219598 | 0.112097 | 0.153441 | 68.3594 | 47.4609 | 0.116533 | 0.060034 |
| 54 | 8192 | 68.3228 | 45.2515 | 0.221355 | 0.113103 | 0.134452 | 66.2109 | 44.1406 | 0.116476 | 0.037272 |
| 55 | 8192 | 69.9707 | 46.1304 | 0.217820 | 0.111191 | 0.157164 | 72.2656 | 48.8281 | 0.108375 | 0.065185 |
| 56 | 8192 | 68.7744 | 45.8008 | 0.220274 | 0.112535 | 0.134465 | 67.1875 | 43.9453 | 0.116985 | 0.063777 |
| 57 | 8192 | 70.3857 | 46.2402 | 0.216272 | 0.110223 | 0.162745 | 69.7266 | 45.3125 | 0.110849 | 0.090213 |
| 58 | 8192 | 70.9473 | 46.5576 | 0.217044 | 0.110613 | 0.150092 | 74.2188 | 47.6562 | 0.108912 | 0.031179 |


## 2026-10-16 archive50第二次实际故障恢复已推进到真实复制

第二次dc9 archive50实际held1在零复制前由两项新MOU mode700 feature目录触发；旧控制/日志原SHA完整保留。预备权限脚本的父目录/40位commit guard在执行前修正；第一次尝试因临时目录已rename，在chmod/intent前明确拒绝。新鲜MOU只读inode查找确认同一inode已名为token-features，已发布61ea40f精确heldfd700→750修复两项，实际receipt1792079537.1257162，无递归、内容或训练操作。

新独立DU真实wait0、stderr0，1792079646.3112707测得1,487,487,234,048B（非原子使用量实测，不是硬quota证明）。b89495c8ec1f2a7fb0d5cd70d89974c16c801cff一次恢复原第二故障到failed-archive-50-second-dc9，INTENT archive50-recovery-second-dc9。初次SSH55s超时没有重试：后续真实host/boot/PIDnamespace查询确认新supervisor58027/start78951603、原子migration与once claim已存在，结果未知状态据此解除。新claim SHA943239779ce4bf0a85ae52034763ecdbb49854a2f487b379e01f082b469cbd83；runner60118/start78953557，task64094/start78955802。

archive worker自身独立DU于1792080018.9327197真实rc0、stderr0，测得1,488,134,529,024B。实际复制载荷从1792080029的1,113,739,616B增至1792080120的19,844,995,283B；尚未完成archive50完整source/archive/target SHA，不能视为验收。原native训练56→58、四rank1160 actor写入及scheduler/head一致，无fatal；RAW f97955→57实际held0，均未重启/热改。第二最终reader/holder显式注册b894，独立second scope及dc9原first migration重消费已纳入admission/stdout/最终末尾重验；相关39测试通过，既有只读reviewer复核无新增确定性P0/P1。目标保持ACTIVE，100/150/200、exactnative51713、最终真实native4、全量RAW/finalheldCPU/no201仍必需。


## 2026-10-16 00:22 CST archive50完整验收、native61与后续容量投影

第二恢复b894的archive50真实child64094已wait0，1792080594.355055返回三方完整SHA PASS。原claim/wait/task/transport及relocated receipt小文件SHA已逐一核对；transport413ef94f42a4a360e632e69225ac2910bd06056d26da1d9673185b6f2088fe76绑定原pair50 c149938fe61b4ffb9ae2a0d746d921369ddf058ced65d254f4f14b8da4fe23e1，覆盖全部35文件，每份38,371,257,523B。原supervisor58027在1792080911.6261125又独立fresh重读源/归档/target通过，现等100；archive5仍沿用原39e2证明。实际native CUDA恢复尚未发生。

native721485在1792080433完成59轮、1792080583完成60轮、1792081018完成61轮，四rank各1220次Adam、scheduler/head一致，无fatal，62已收集真实512/G5。原RAWf979在1792081202实际held0前缀1..61；训练及RAW未重启。目标ACTIVE；固定100/150/200、final51713、全部五套三方完整SHA、最终native4两次dataequal、全量RAW/独立heldCPU/no201仍未完成。

容量投影并非新DU/quota证明：archive50复制前实际DU0=1,488,134,529,024B，两份载荷76,742,651,357B。未来100/150/200三源六副本估计345,341,317,707B，另计200两份150 encoder约27.21GB，固定产物约1.937TB，余62.57GB。最近20轮42–61仅rollout、behavior JSON、selection/audit、before/after边界文件的allocated均值0.757GB/轮，139轮按此推算最终约2.043TB，至少约42.68GB缺口；私有额外痕迹、校准/评测及其他共享目录增长尚未计入。已清理204.22GB已在基线内，不能重复扣减。fresh DU+2TB门禁不变。

下一步准备可核验远端保留/腾挪方案；未新增删除授权，未删除或移动旧原始5/50/100、旧100额外archive、当前目标任何源/副本/RAW/teacher。证据storage-projection-after-pair50-and-recent20.json及sigma3-recent20-storage-metadata-1792081009.json；只读review确认有实质容量风险，但投影不能当必然硬quota超限。


## 2026-10-16 00:41 CST 拟合诊断至65与非破坏备份准备

真实native721485与原RAWf979均完成65，四rank各1300 Adam写入、scheduler/head一致；66已完成512×5采集。未重启或修改训练。独立CPU历史诊断从原完整head50重算51–65，每轮原model version和全部fit loss一致，50–58全部指标与上一诊断逐项完全一致；65暂无完整状态终点锚。完整replay65：二分类72.6196%、六分类47.1436%、方差RMSE0.107527/R²0.161850；当前512：71.4844%/45.7031%/0.112142/0.026211。当前题56–65二分类均值70.8008%，多数类68.3984%，优势2.4023pp；1–10对应优势9.0625pp。数据sigma3-postfit-training-metrics-outer0-65.json SHA e8ca2d05daa88e509a1569dcc7fb704491359f7063f6647d71c6cdd67d8275bd，本地实验记录保留每轮完整表与静态曲线。不得将loss下降直接当方差解释能力提高或据此判定过拟合。

当前2TB预算未来容量风险仍需处理。准备HPC3→HKUST3只复制旧人类CANCELLED719225的原5/50 checkpoint及额外100 archive、恢复依赖/原receipt metadata，共118个去重文件121296682925B，计划SHA0c9a27ca36509f885ad2c0d0d5ff357a3c4f7055ce485ec5ff607ec9dc6d75f2。新工具强制source before/after与HK3 target逐文件完整SHA、单writer、RAM密码、实际held worker wait验收及未知结果只inspect。目标HK3原ssh别名、UID1002/实际boot固定，fresh可用10.24TB。当前只备份，无删除授权、源清理或释放空间；旧原件和当前目标全部产物仍保留。备份验收后才可形成精确三树115100389376B腾挪提案并请求新的范围授权。执行前须已commit/push/两端源码全SHA与真实rsync dry-run通过；7项本地scope/heldwait测试已通过。

65014c9代码已推送并在HPC3/HK3两端1074文件完整SHA通过；实际远端rsync dry-run认证/协议/118文件清单通过，无payload复制。只读审查随后发现绑定写入失败可能遗失held wait、终态摘要未重核当前target全集两处P1，以及本地SSH timeout未落盘；全部在start前修正。新实现finally收敛直接child实际wait，终态核完整manifest/前后token/各阶段wait0/当前目标118文件path-owner-size-inode-device-mtime及无额外文件，timeout保存UNKNOWN_NO_REPLAY。12项本地测试通过（包括Popen后publish失败、缺件/改inode/非零wait/异token/timeout），仍未启动备份，旧预检证据保留，待修复源码发布后重新预检。


## 2026-10-16 01:10 CST 备份真实复制与保留旧100依赖核验准备

afda9f708ad0d93312c162d3aa8452816c420f35两端1074文件全SHA发布和真实dry-run通过后，备份仅start一次，token800d9a5f728445d893d741e635b43ba1，HK3 supervisor3806261/start987513611、worker3806263/start987513616、rsync3808257/start987529760，均SID3806261/UID1002。源before实际118文件121296682925B全SHA通过且SSH child真实wait0，source-before.stdout SHA afffeea6704718f2f8302f034ae8e5972ac5e39b7f6290c7c82c070ed1dadfe9。实际复制allocated由1.659→9.631→20.826→32.886→48.793→80.051GB持续增长，stderr0。源after/目标全SHA及最终held0仍待完成；未新增删除，释放0B。

一次372个当前日志/manifest、teacher-ready、current pair5/50和保留旧source100小JSON引用扫描全部完成，无skip。当前目标不引用候选三树；旧source100明确引用旧5/50 checkpoint下complete.json两个小清单，未来必须保留其HPC3原路径，不能整树删。草案排除两文件后候选78文件115100222440逻辑B，仍无删除授权/执行入口。准备独立一次read-only旧source100完整SHA核验，固定原transport/receipt及全部恢复依赖，零tensor/model import、GPU或训练动作；真实held child wait，源别名若越过两保留清单则拒绝。5项本地完整性/原生4卡/别名/发布失败wait测试通过，执行须在最新源码推送和HPC3全SHA发布之后。

622724a44e7ea25740d6c53dd7ae1967919ea6cc已推送并HPC3全部1076源码文件SHA通过。独立旧100读取仅start一次，supervisor683803/start79390952、worker683805/start79390972、SID683803/UID14090；实际held child exit0于1792084346.9503343，完整37文件38371213245B SHA及全部metadata身份通过，complete SHA48a249ab7516d391468227cc32defa05e451ed48b6918c66f2a39815d2dd733f。两处旧5/50 complete.json依赖再次确认，当前科学训练和原备份afda源码均未改变。

为使未来腾挪前的备用副本具备明确持久化证据，另准备独立终态fsync工具：只在原afda备份真实held0、118文件全SHA/当前全集身份通过后，对同一owned target文件与目录逐项fsync，前后身份不变；不宣称另一轮全内容重读，不删除或改变HPC3源文件，不热改原worker/source。工具发布到独立HK3 durability目录，以新Git commit/单文件SHA及原afda全源码证明绑定；须实际SSH返回0和持久receipt后接受。两项本地文件身份/符号链接/实际fsync测试通过，尚未执行。


## 2026-10-16 Exact78 old-original retirement authorization

The original afda reserve has completed all 118 files / 121,296,682,925 bytes with source-before/source-after/target full SHA and actual held child exit0. Complete SHA ba51bc0dc2511117bfef6072832d3ee28f475b0828fb6ff2037d1d52b2f98739; independent owned-file/directory fsync receipt SHA 1410139a59035dd45a228fce4ac6b8bc4f67591420dd07ef4de1a25bb1b202d1. Retained old100 has separately passed all 37 files / 38,371,213,245 bytes full SHA and actual held0, complete SHA 48a249ab7516d391468227cc32defa05e451ed48b6918c66f2a39815d2dd733f.

The human now explicitly approved the exact78 proposal SHA a04121d8ddfc4d552330d666143f73f43d1a8b17a2c716f00ed298e59d8e08bb, 115,100,222,440 logical bytes. This new authorization supersedes earlier preservation only for those named files. Original old100 and both referenced old5/50 complete.json manifests remain at HPC3, and full recoverable original5/50/archive100 copies remain at HKUST3. The proposal stays unchanged; separate authorization receipt SHA e5fcfea443213760bf4af2246f3fc212ad926e5e44f0e4b425faf73e6e3d6337. The original UI timestamp was not provided; only observed receipt time is recorded.

New CPU-only exact-name retirement holds all78 candidate full-SHA FDs through directory-FD unlink, validates active native4/old human cancellation and authenticated fresh backup admission, and fully re-reads retained100 before and after. A once-only supervisor retains the actual child through wait, including binding errors; unknown or partial state is inspect-only. Empty directories remain. Six local scope/hash/mutation/replaced-name/symlink tests pass. Execution requires Git push and all-source remote SHA publication; no training/model/retrieval action is authorized by this cleanup. Latest measured native721485 completed74 with four ranks at1480 Adam writes, matching head/scheduler, no new errors.

Before any start, independent read-only review identified stale NFS metadata as a P1: the held candidate must be fully rehashed immediately before each unlink. Fixed this while preserving the full78-before-first-unlink barrier; added equal-signature/changed-byte rejection, later-candidate failure prevents first unlink, and supervisor binding-publication failure still performs actual child wait. Nine exact78 tests now pass. First publication812cb14 was read-only preflight only; no retirement control or deletions occurred with that source.

## Exact78 retirement actual completion (epoch1792086826.4614234)

The once-only2ce2a38 CPU supervisor retained worker1138254 through actual exit0. All78 candidates were fully hashed before first unlink, then each held FD was fully rehashed immediately before its exact directory-FD unlink. Exactly78 approved names /115100222440 logical bytes are absent; the unlinked inodes had115100381184 allocated bytes. This is not a filesystem-quota delta measurement. Empty directories remain. The original100 retained37 files /38371213245B were fully rehashed both before and after and have unchanged identities; old5/50 complete.json dependencies are intact. CompleteSHA3a506e71835a7083319a17fddd29ca04374b53f64720953011b84d8932e6f9cd.

Independent second inspection at1792086871 confirms original100 metadata and original held0 proof still accepted. HKUST3 originalafda backup inspection at1792086870 again accepts all118 target files/current identities, full before/after/target SHA history, actual held0 and fsync receipt. The old full backup verifier source_manifest/old100 verify-start cannot be rerun after this approved retirement because it intentionally requires retired source originals; use original target inspect and new exact78 inspect. Never replay the completed cleanup.

Training continued: native721485 complete1..76, all4ranks1520 Adam writes with matching scheduler/head, no new errors,77 running. RAWf979 actual audit accepted1..76. Existing archivefollowerb894 remains the same live birth58027/start78951603 and waits100 after independently full-SHA accepted5/50. A55s heavy follower inspection timeout is recorded as unknown observer outcome; lighter existing-birth/state inspection re-established its live wait state, no restart or hotedit. Goal remains ACTIVE; final200/native51713/all5paired archives/actualnative4recovery/independentCPU/no201 remain pending.

A short synchronous post-cleanup full experiments du query reached its40s observation deadline; its own retained measurement child exited-9 after kill, with no usage accepted and no training signal. Prepare a separately named once-only detached storage observer, byte-identical to the previously validated observer except fixed control directory/source path. It retains the actual direct du child through communicate, rejects any nonzero/stderr/partial output, and reports current allocated usage separately from quota. Retirement itself and both independent postinspections already accepted; never replay cleanup for this measurement timeout.

Detached post-exact78 full experiments du genuinely terminated with held directchild1 at1792087294.1294093 (285.1759s),766B stderr: five unreadable private tmp directories belonging to other periodic25-thread5/-r1 experiments. Partial stdout1467047866368B is explicitly NOT accepted as a complete storage total. No permission changes to other experiment trees, no scope expansion, no other-agent messages, no replay of cleanup/observer. This observation failure is separate from the already twice-accepted exact78 cleanup, preserved100 fullSHA and complete118 remote backup. Native and RAW now both77; final200 remains pending.

## Exact directory-read repair for future archive capacity admission

Native and RAW freshly accepted78/all4ranks1560Adam/head/scheduler; same b894 follower waits100. Static inspection confirms archive_completed_pair.measured_owner_storage performs its own strict successful full-owner du; five denied private tmp roots would therefore also block actual100 archival. Owner MOU inventory at1792087547 found23directories, of which exactly8existing700inodes need group read/traverse (5roots+3internal parentdirs);15existing770dirs and all regular files remain unchanged.

Apply the human's already explicit shared experiments authorization (both mou176/jxu957, requested chmod-R770) as a minimal exactFD700->750 change by originalMOU owner21930/GID21919. No recursion, group write, world bits, file bytes/modes, scientific source, training/RAW/follower restart, cache/model/retrieval modification or other-agent messages. Reuse the previously applied archive50 directory repair implementation with immutable8target inventory, exclusiveintent and code/target fullSHA publication. A separate existing-template once-only duobserver revalidates the actually failed observation; never replay prior failed claims. Current run continues unchanged during this preparation.

The repaired fullscan genuinely failed1 again at1792087858.3827248 after69.4636s: a newly-created r2/torchelastic_ego2f2yy root denied read, plus one process-ledger.json.tmp ENOENT race. Original8repair is intact. No partial1468781359104B total accepted. OwnerMOU actualr2inventory1792088062 found7dirs/0files, exactly2mode700(parent/run),5mode770. Prepare the same exactFD700->750 shared-access repair for only those2newinodes, fixedseparateintent; old8notreplayed.

Fresh observer uses canonical archive policy: atmost3complete scans only for pure within-experiments ENOENT, retains each directchild/process/wait/output receipt, never retries or accepts permission/mixed failures and never accepts partial stdout. Final inspection reconsumes each perattempt SHA/process/timeline and the successful finaldu0. Four meaningful simulated directchild/error-path tests passed (ENOENT->0,permission stops,3ENOENT bound,outsidescope/mixed refusal). Healthytraining/RAW/b894follower and scientificACA source unchanged; realnative79complete,80has512x5collection and continues.

The R2 fullscan genuinely failed1 at1792088402.8222516 after68.1227s for another newly-created torchelastic_wfz3wjm8 root under the same-r2/tmp; previous10fixedinodes stilltraversable. Partial1470302076928B is not accepted. Instead of publishing another guessed whole-tree permission change, prepare a helper that consumes an owned terminal published DU failure with exactSHA, validates complete stderr and dead process/child births, and selects only actualreported permission roots matching the fixed periodic25-20261016/tmp pattern. OriginalMOUowner21930/GID21919 may change only actualmode700directoryFDs to750 below those roots, after everyFDguard passes. Exclusive control is keyed byfailureSHA; no replay. File bytes/modes/model/cache/retrieval/training untouched.

New storage observer keeps the proven3pureENOENT retry/fullheldproof logic and uses bounded numeric fresh slots(1..16), each with separate originalclaim/process/results; no source rewrites are needed for a genuinely failed later slot. Dynamic repair has strict model/cache/path/mixederror/liveDU refusal tests; together with prior retry tests8PASS. Actualexecution stillrequirespushed/fullSHA source and actualpriorfailureSHA; no new training/control action is permitted.


## 2026-10-16 02:33 CST：授权清理与清理后全量空间复核完成

本次批准的78个旧文件已删除，合计115,100,222,440逻辑字节（约115.1GB）；原操作的实际子进程退出0，完成凭证SHA为3a506e71835a7083319a17fddd29ca04374b53f64720953011b84d8932e6f9cd。旧第100轮原检查点37文件在删除前、删除后均完整SHA通过；两份被引用的旧5/50 complete.json仍在HPC3。HKUST3完整备份118文件121,296,682,925字节仍保留，源前后及目标完整SHA、实际退出0、文件/目录fsync均已通过。02:29再次独立读取清理、备份与保留100的凭证通过。此清理已终态完成，禁止重放。

清理后的全experiments扫描此前被其他periodic25实验新建的mode700临时目录阻断，部分du输出从未当作总量。按用户此前两账号共享experiments授权，原MOU属主仅对实际报错目录及其私有子目录做精确FD700→750，JXU随后逐项确认同一inode可读可遍历。固定8目录、R2固定2目录后，另对实际新R2失败修复2目录，对新R3失败修复4目录；所有失败凭证、once intent和逐目录结果均保留。未改普通文件、模型、缓存、检索或训练；未联系其他任务。

通用修复和有界扫描槽位源码a2e4934b7375e8202eae1a1e46f3c326d116c0b0已推送，HPC3全部1094文件SHA通过（归档SHA abd4c2c507f9c55d3b90e40fa50faef3f338fb65789d1994e0388a8411f0c48d）。已有只读审查无确认P0/P1，8项范围/失败/重试测试通过。修复仅消费已终止、已发布、完整错误和全结果SHA匹配的实际失败，不允许接受部分du或重放未知intent。

新独立slot3完整扫描于1792089159.771583真实du退出0、stderr0，测得experiments allocated总量1,472,658,509,824字节（约1.473TB）。结果SHA ecf0b797ff64ec96661e6f339427cca8669af3afd928e09a5577e9fd07671bf3；两次终态检查1792089180/9203重新消费实际子进程、claim及各attempt证明通过。按2TB预算当前余527,341,490,176字节；这是活跃目录非原子实测，不是存储硬quota证明。该已完成slot3同样禁止重放。

以此新鲜基线，加未来100/150/200三源六副本约345.34GB、200复制所需encoder约27.21GB，以及42–61实测保留RAW均值0.757GB/轮估计剩余118轮，预计最终约1.935TB、余约65.44GB。该估计未计其他实验后续增长及额外评测产物；不能替代每次archive的真实全du和2TB门禁。115.1GB清理已体现在新鲜基线中，不重复扣减。详细估计保存在storage-projection-after-exact78-and-full-du0-20261016.json。

当前721485四卡完成82/200，四rank各1640 Adam写入，scheduler/head一致，无新fatal；83已完成512×5采集。原RAWf979已实际验收1..82；原b894同一进程等100，5/50完整三方SHA通过。科学源码ACA和训练/RAW/follower均未重启或热改。目标保持ACTIVE；固定100/150/200、最终exactnative51713、全部五套三方SHA、最终真实native4恢复与两次dataequal、全RAW/独立最终CPU/no201仍待完成。


## 2026-10-16 02:36 CST：native83完成与选题验收到82

当前四卡721485实际完成83/200，四rank各1660 Adam写入，scheduler/head均与成功outer83一致；83拟合总loss0.9893484711647034，第84轮已开始。对照前一次82轮/1640写入的实测，确认一个完整iteration正向完成，未见新fatal。原RAWf979在83成功返回后已启动同一次真实CPU原始轨迹审计，尚未将其当作held0；5/50归档已全SHA通过，原b894同一进程等100。未重启、热改或新增训练。

用已发布源码e6a22a52ad87f0d51b3ff5d8bc7d4d91feb6aa5a纯CPU重消费实际RAW1..82 held-child/claim/result/stdout两次，并新鲜hash每轮head/scalar，生成只读选题汇总。无模型加载、轨迹生成、参数更新、远端证据写入。该汇总不代替最终200原始轨迹重解码或真实native4恢复，也不是postfit二/六分类准确率或方差拟合。

| 范围 | Top448混合数/题次 | Top448混合命中率 | 随机64混合数/题次 | 随机混合命中率 | Top448提升 | Top448真实平均样本标准差 | 随机真实平均样本标准差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1–82累计 | 14652/36736 | 39.8846% | 1335/5248 | 25.4383% | +14.4463pp | 0.196306 | 0.124891 |
| 73–82最近10轮 | 1378/4480 | 30.7589% | 135/640 | 21.0938% | +9.6652pp | 0.151199 | 0.103757 |

最近10轮全512生成前标准差预测MAE0.209036、MSE0.052766；全512每轮内部mixed-AUC均值0.550431、成对concordance均值0.546335。Top448对应0.528189/0.525786；这些排名指标仅各自同一policy的轮内计算再取算术均值，禁止跨policy混池。选题仍相对随机有正提升，但最近10轮Top448与随机绝对混合率、真实标准差均低于累计值；不能因此改动当前科学配置或将loss下降当作选择性能已通过最终验收。

小型完整逐轮数据sigma3-audited-selection-metrics-outer1-82.json为217997字节，SHA674bc9fda6d89b7f0cdc86eeac6012f5bf7cf51941ee0807d46706485ef03055；原审计sigma3-progress-summary-max82-service-1792089362.json保留完整路径/SHA/held证明。该数据不含tensor/rollout，符合本地小审计限制。目标保持ACTIVE，固定100/150/200、finalnative51713、五套三方SHA、最终native4两次dataequal/RAW/独立CPU/no201仍未验收。


## 2026-10-16 03:22 CST：四卡native90里程碑与实际审计等待

同一HPC3/jxu957作业721485保持4GPU/ACD1-21/emergency_acd，科学源码aca0e779e94ddc3fbd50e49c3d59e2e889fbcf12未改动。连续83→90成功迭代均核对全部4rank的每轮20次实际Adam写入、scheduler及head提交；90成功前缀1..90、每rank累计1800写入在1792092106.7232976通过，head qprobe:90:90:4242349d5f84f05e，fitloss0.9442072510719299。第91轮已开始，不能将未完成91提前计入。当前观察无新fatal，未重启、热改或额外提交训练。

原RAWf979 supervisor3857750/start78675791实际heldCPU0验收前缀1..89；第90轮同一直接child actualbirth live，尚未退出验收，不把LIVE当成功或故障。原b894 follower58027/start78951603在1792092108仍实际live等100，5/50原三方完整SHA证明保持已验收；light观测不宣称新SHA。原审计、归档进程均未重启。任何观察延迟均只重读同一handle，未据此恢复。

| 成功outer | 每rank累计Adam写入 | 原始head fitloss | native实测证据 |
| --- | --- | --- | --- |
| 84 | 1680 | 0.9893519282 | recovery721485-progress-1792089906.json |
| 85 | 1700 | 0.9899488091 | recovery721485-progress-1792090173.json |
| 86 | 1720 | 0.9909700751 | recovery721485-progress-1792090569.json |
| 87 | 1740 | 0.9713534713 | recovery721485-progress-1792090985.json |
| 88 | 1760 | 0.9614369273 | recovery721485-progress-1792091366.json |
| 89 | 1780 | 0.9486702085 | recovery721485-progress-1792091736.json |
| 90 | 1800 | 0.9442072511 | recovery721485-progress-1792092114.json |

每个回合均至少两次实测，并等一个完整native迭代返回或记录同一具体审计child的实际verified wait。原RAW87更慢的child2054890/start80056686在1792091072已terminal/stdout6360B/stderr0/result2663B，1792091081由原supervisor重消费通过；没有把延迟当作故障。RAW88/89分别1792091433/1792091804实际接受。最新RAW service-raw-inspect-1792092107.json、follower archive50-existing-follower-light-1792092108.json，完整小型运行审计保存在本地outputs/qprobe-sigma3-loss-20261015。

当前90轮Top448生成前选题的实际混合率122/448=27.2321%，随机组11/64=17.1875%；真实平均样本标准差分别0.132779/0.083147。这是单轮prequential观测，不是训练后二/六分类准确率或方差拟合；最近10轮的已完整重消费汇总仍截至82，待100完整状态锚后再更新，不用单轮推断算法优劣或据此修改科学配置。

已授权exact78清理、HKUST3完整118文件备份、保留旧100/两份小清单及postcleanup DUslot3两次验收均保持完成；不得重放清理、DU或已完成failureSHA权限intent。最后完整空间实测1792089159为1,472,658,509,824B，当前活跃目录继续增长，不能把旧数值当新鲜quota/归档容量证明。未来每次archive仍独立strict全du/2TB门禁。

目标保持ACTIVE，不能完成于90或Slurm RUNNING。固定100/150/200、最终exactnative51713与七源EM/F1 macro+weighted、全部五套三方完整SHA、实际native4全状态恢复与两次dataequal、完整RAW/独立finalCPU/no201均仍待最终实际验收。RAW90 exactheldchild如下，下一步只consume同一child：
{"pid": 2283511, "sid": 3857750, "start_ticks": 80168080, "uid": 14090}


### 2026-10-16 第100轮固定验证与归档100故障

实际四卡721485已完成102轮，原f979独立RAW读回held0至102；第102轮Top448混合126/448（28.125%），随机15/64（23.4375%）；最近93–102轮分别1192/4480（26.6071%）和129/640（20.15625%）。固定同一512题第100轮EM206/512（40.234375%），F1=48.6545926372%；第50轮EM185/512（36.1328125%），提升4.1015625个百分点。

B894 CPU follower与原archive100实际退出1：全量du遇到新R6临时目录Permission denied，在复制前拒绝部分空间结果。原100pair和所有失败记录保留，payload根实际缺失；原5/50已通过的归档保持原claim/source/controller及transport SHA。R6、R7的实际DU错误分别触发已发布a2最小FD目录权限修复，并由JXU按原inode读回；不修改正常训练、RAW或其他任务内容。

新增recover_sigma3_archive100：固定原100pair/claim/wait/log SHA和三个死亡进程身份；排他intent后只原子保全失败control，保留旧follower原位；必须消费明确slot的真实held0完整DU，再由标准归档worker独立执行完整DU和2TB准入。继续原5/50记录，后续100/150/200及最终native4仍走原完整验收。真实字节fixture覆盖篡改、活动进程、空payload目录/悬挂链接、错slot、原50身份及未知intent不重放。恢复尚未启动，最终200未验收。


The archive100 recovery implementation is published as4194ad2c21c5e10ae5ee3b0212fb6fbf3e02f41b; remote1097 source files matched at1792097689.1649547. The independent final reader/holder now use a separate archive100_recovery_commit pin while retaining B894 as archive50, dc9 as its prior handoff, f979 as original RAW/service and09cc as original archive5. The exact original50 claim, held0 and transport remain inherited; later archives and native submit bind only the new100 follower. Five new protocol tests plus51 existing recovery/reader/holder/transport tests passed locally; these are compatibility tests, not real200 acceptance.

The first read-only100 preflight exceeded the former55-second SSH observation window; this is UNKNOWN, not a terminal training/recovery failure. The local transport now preserves its55-second default and permits an explicitly bounded1..1800-second observation for this longer already-authorized SHA validation. Both SSH and HK3 observation deadlines remain finite, strict endpoint/kernel binding and RAM/stdin credentials are unchanged, and timeout never permits replay. No100 recovery start has yet been sent.

### 2026-10-16 05:28 CST：第100轮第二次归档恢复准备

上述C1恢复4194ad2已实际执行一次：原B894失败control完整字节及inode经原子移动保存在failed-archive-100-b894，原follower保留原位，新C1 follower实际启动。随后C1归档worker再次因其他实验新R8临时目录Permission denied，在独立全du准入阶段真实退出1（1792098910.566765）；C1 follower退出1于1792098924.3780577。三个固定出生身份均已死亡，payload根实际缺失，task-result不存在。部分du数值1532322361344B不接受为使用量，失败证据完整保留；不能重放已有C1 intent或start。

实际DUslot10只报告当时仍存在的两个R8 torchelastic根，已发布A2 helper由原MOU属主对4个实际mode700目录FD执行750，JXU按同一inode和14目录遍历读回通过。后续全量DUslot11实际退出0、stderr空，完成于1792099186.5770264，allocated1532964118528B，两次终态读回通过；此值为活跃目录非原子实测，不能代替未来归档worker的独立全du或硬quota证明。所有完成修复/DU intent禁止重放。

新增recover_sigma3_archive100_second固定消费C1实际claim/wait/log/六条R8错误及三个死亡出生身份；同一验证阶段只consume一次原C1完整链，原5/50及f979/09cc身份保持不变。新排他scope只原子保全这次失败control为failed-archive-100-second-c1，拒绝任何payload根、活动或未知进程及已有intent，仍需明确新鲜全DU0并保留标准worker独立2TB门禁。已有本地只读审查未发现实质问题，11项新协议测试及32项相关恢复/归档回归测试通过。源码必须提交、推送、全SHA发布后才允许执行；当前尚未执行第二次恢复，也未接受100归档。

同一原四卡训练已实际完成108轮，全部4rank各2160 Adam写入，head/scheduler连续，109正在进行；未重启或热改。原RAW held0至107，第108轮实际CPU child正在审计。最新108轮Top448混合131/448（29.2411%），随机11/64（17.1875%）；已重消费的98–107最近10轮为1198/4480（26.7411%）和117/640（18.28125%）。目标仍ACTIVE，200及所有最终归档/原始轨迹/实际native4两次读回/no201门禁尚未完成。

第二100恢复源码23ab8b0c8af403ec830644ff4fa04eeef7f55927已推送，1101个远端源码文件完整SHA于1792099711.0783858通过（归档SHA2f86cc472019a9bf0fa18da6640834851e8f18d3230437ab414453d033945b31）。独立最终reader/holder只注册C1和这一确切C5 pin，两条路线使用不同模块、follower和排他holder scope，不接受任意commit回退。C5将当前100恢复与原C1恢复分别输出；入口和最终边界各fresh消费一次C1并验证原B894/原50继承/RAW/5一致，stdout绑定两条100证明。五套完整归档held0、原50身份、真实native4两次读回、final200和no201保持全部要求。本地77项相关协议测试通过，新增线路测试覆盖未知pin/两scope互换、C1缺失或变动、原50继承变化、missing200早拒绝，以及所有5/50/100/150/200/native-submit held0与源身份。这些为协议兼容验证，不是实际200终验；第二100恢复尚未执行。

### 2026-10-16 05:47 CST：C5实际失败保留及阶段校验去重

C5 start实际执行一次，PID4104475/start80968623持续读取并于1792100415.9465575创建新follower4178254/start81002039。原C1失败control已原子保全；独立读回8文件10628B同SHA、同inode，migration SHA ea83848538cd49f09f293f697ef12c8e56dd01d46be5030893297f24bb9a6f6a。原生四卡110、111均完整提交，原RAW110已held0；本地观察会话中断没有触发任何重放或训练改动。

新归档claim SHA4b330576cb220fb38ecf8c41d96a0ee1d621784603d47a16d82eeee9f1368912，runner1351/start81009899，直接child2878/start81010738。独立全du于1792100592.8158503实际失败1，新的R9两torchelastic根Permission denied及一个临时ledger ENOENT；部分1536853307392B拒绝。child held1完成1792100593.4124832、follower退出1完成1792100618.2825127，三个出生身份均死亡，payload根实际缺失。完整原始小证据（归档8文件9717B、follower6文件47471B）已保存。C5原失败control仍在原位；禁止重放它的intent/start。

独立DUslot14也真实失败，full result SHA2e252492ac944919e680413175591114e4e1e3b180d68a725318af73a268e93d。已发布A2 helper按实际错误只修复原MOU属主4个目录FD700→750，JXU同4inode/14目录遍历读回通过；无普通文件、训练或检索操作。修复intent SHA b3e55874ac45f5659e10214482df4947e70865eead6ea4b1c13f66476b1c6ecb，result SHA f5e40cd2cec62bb7679e5bd671bcf3edb67e28e7fe167cff0271236fee2f2c19，全部已完成操作不重放。

实际读取量和只读调用图审查指出，C1的隐式inspect/consume让一次C5 failure_evidence派生约4次B894、16次dc9、48次f979（每次49个原RAW收据）。新增recovery_lineage_stage仅在一个恢复边界内使用私有导入实例，把完整验证过的同一原consume证明作为DAG节点复用；seal时每个原consumer再次完整执行，所有来源重新完整SHA，并对保留证据再次完整hash。禁止缓存host、PID/live、Slurm或payload查询，禁止共享全局替换及跨边界cache；每个敏感边界必须创建新Stage。7项协议测试覆盖原嵌套证明等价、同长度/恢复mtime篡改、末次源校验后篡改、来源变化、host/live变化、新依赖及跨边界不复用。仍需发布后对真实C5原消费者做逐字证明等价对照；尚未实现或启动第三100恢复，200全部最终门禁未通过。

阶段优化初版a1e6575已完整1103源SHA发布，真实只读C5原消费与Stage消费证明逐字相同，SHA均546e5b276b241f752bdc297f7f7fe48b7d3909b28606e23c1137c1bd7e053e2a，耗时90.4069s/18.6713s。此为静态状态等价诊断，非最终接受：只读审查发现P1，普通control.read及observer.read_json尚未全部进入seal最后重哈集合。现修复为保留原scope检查，JSON返回对象与记录SHA从同一次完整稳定读取获取；不缓存活跃ledger、PID、host、Slurm或payload。新增普通JSON/observer JSON末次publication期间篡改拒绝，以及实际原50五项held文件经共享验证器接受后逐项同长度篡改拒绝。修复版需重新发布和真实等价验证，不用已知缺口的初版启动恢复。

第三100恢复实现绑定C5实际claim/wait/log/两R9拒绝及ledger ENOENT、原follower/runner/child死亡身份及缺失payload根，沿用排他intent、完整相对文件集/字节/inode原子保全和独立全du/2TB门禁。每个敏感failure_evidence边界新建阶段，完整消费原C5证明；输出分开保存原C1及原C5，原B89450/RAW/5保持原身份。31项相关本地测试通过。第三恢复尚未执行；还需精确新source pin注册到最终reader/holder并发布，真实200、全部归档与native4两次恢复等最终要求仍完整保留。

P1修复及第三100实现f8a332a72257f54bf0c01a01fce592e2515d7937已推送，1106个远端源文件于1792101296.5155344完整SHA通过（归档SHA3fb54d91428fea7f52add07136a18746bd88233ba72589ea52c78c7c4cb7d026），只读复核确认P1闭合且无新P0/P1。修复版真实原C5/Stage等价对照已执行到独立CPU观察进程，结果仍待实际返回，旧a1e诊断不替代它。四卡112已完整all4Adam2240/head/scheduler，113采集完成但未提前接受。

独立reader/holder只新增精确f8a第三route/scope，入口及最终边界重新完整Stage消费原C5，明确输出并绑定原C5和原C1两份prior证明，原B894/dc9/f979/09cc保持身份。原5/50只继承真实held0且全SHA的原claim；100/150/200/native-submit仅接受第三follower的真实来源与控制进程。102项相关本地协议测试通过，包括三scope互换拒绝、priorC5/C1/原50变化拒绝、missing200先于source/scope/spawn拒绝，以及所有五套归档与native-submit实际held0门；最终200/native4两次完整恢复/no201未放宽。第三100仍未执行，目标ACTIVE。

### 2026-10-16：F8失败保留与同一归档worker等待修复

上述F8后来实际执行一次：原C5八文件9717B原子保全后独立同SHA/inode读回。独立归档全DU因新R10的torchelastic_oirmwa6a权限失败，在复制前held1退出；follower也实际退出1。三出生身份死亡，原100pair及14份原始小文件字节保留，payload根缺失。原四卡115完整、116采集中，原RAW未改动。

新增第四typed F8交接精确绑定原claim/wait/log SHA、三dead birth及pair，分别保留C1/C5/F8三份prior和原B89450/RAW/5。外部额外DU0是实现自加的重复检查，移除；唯一复制准入仍在原archive-storage.lock内执行完整DU、真实held wait、完整零错误输出及2TB预算判断，随后才创建payload。权限失败保留完整DU receipt，同一归档worker进入WAITING_STORAGE_REPAIR；同源MOU精确FD修复记录及JXU当前inode/权限/遍历复核通过，才启动新编号完整DU。部分总量、未知或已有claim均不接受或重放，主训练/RAW不动。

旧slot1真实DU1报告R10和R11临时根；A2 helper遇到一个已经消失的R11根，在intent和任何权限修改前退出1，未冒称成功。新版helper单独记录确实已消失的报告根，只改变仍存在的实际报告树内700目录FD为750；JXU复核消失根仍不存在，复制仍需下一次完整DU0。没有删除、递归chmod或主动扫描修复其他实验。

审查发现正常与异常路径都必须持有并回收observer Popen，现由私有core捕获真实handle，正常wait后再消费DU；Popen后start失败、inspect或receipt写入异常也真实wait，超时明确记录cleanup_pending而不冒称终态。真实本地子进程测试覆盖正常回收、异常回收、pending及permission期间零payload/同parent修复后再测。F8内部依赖只对私有模块绑定同一Stage，seal继续完整原consumer/sourceSHA/保留字节复核。实现尚未执行，须发布及精确第四route注册后再一次启动；最终200全部门禁保留。


### 2026-10-16 06:08 CST：114轮真实指标与F8归档失败

修复版Stage对真实C5消费者的完整证明等价复核通过，SHA546e5b276b241f752bdc297f7f7fe48b7d3909b28606e23c1137c1bd7e053e2a，原消费者79.4092秒、Stage19.1183秒。F8恢复实际执行一次，原C5八文件9717字节同SHA/inode原子保全后独立读回通过，新follower277215/start81134743启动。归档runner281967/start81137069、child286742/start81138876因新的R10 tmp/torchelastic_oirmwa6a目录Permission denied，独立全DU真实退出1；部分1540205047808字节未接受，复制前失败。child于1792101874.5357316 held1，follower于1792101889.941727退出1，三出生身份死亡且payload根缺失。原始归档8文件、follower6文件的小字节完整副本和SHA已保存；禁止重放F8 start/intent。主训练及原RAW未改动。

原四卡721485完整1..114，全部4rank各2280 Adam写入/head/scheduler连续，原RAW实际held0至114；115正在进行，无实际fatal。114轮Top448混合111/448=24.7768%，随机13/64=20.3125%；105–114轮1199/4480=26.7634%，随机120/640=18.75%，差8.0134个百分点。最终200及归档/恢复验收未完成。

旧通用观察器slot1已核对NOT_STARTED后启动一次（368375/start81175784，du368376/start81175792），保留同一进程观察，不因超时重放。新增独立测量namespace入口复用原完整DU worker的私有模块实例，槽位1..64各有排他claim；同时验证入口和复用core的发布SHA，不改原观察器文件或已有槽位。失败部分总量拒绝、最多3次纯ENOENT、permission不重试及原生held-child完整检查全部沿用。只有代码推送并完整发布后才执行新入口。

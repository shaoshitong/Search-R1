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

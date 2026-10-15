# Question diversity probe：本地静态代码与证据核对

2026-10-16。只读本地源码、小型JSON与配置；未SSH、未读模型tensor/原rollout/大log、未运行模型/训练/独立测试，也未联系其他线程。最新独立新512题测试状态由负责线程报告；本笔记不虚构其结果。

## 核对身份

现实现目录：`/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix`。当前本地关键8文件SHA逐个匹配已保存publication中的`source_sha256`，其source commit为`fcb77204f48d5cb8ec21c68df7d4a2c937b3557a`。这是本地持久证据核对，不是新的远端实时核验。

[publication记录](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-selection-metrics-fix-source-publication.json:1)、[独立source审查](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-selection-metrics-fix-independent-source-review.json:1)。所查模型/selection/feature/runtime源码均匹配，不是凭目录名认定线上版本。

正式作业小摘要绑定run `509c391b7e5c4174bfb55f724f41a3d6`，remote base `/data/shaoshitong/Search-G1-a800-fresh200-e9c4c584107b4624bcc9787aaf24b7a2`。原warmup200与formalinitial的文件SHA不同，但已保存CPU实际比较receipt证明14个model tensor、78,076,418参数、FP32、optimizer及private RNG一致，均finite；formalinitial并未替换成另一个随机head。[身份实读receipt](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-new512-cpu-weight-compare-20261016.json:4)。本次仅阅读该receipt，没有重读权重。

## 当前训练实际学的是什么

### 输入及可训练结构

问句原文经固定官方 `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`编码；缓存保留**layer27输出的所有非padding问句token**，无truncation，hidden2048、全局max244。hook是该decoder layer的forward输出，不是layer35输入，也不是实时actor当前隐藏状态。[缓存spec](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_token_features.py:33)、[实际抽取hook](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_token_features.py:72)。输入编码question-only/add_special_tokens，不含正确答案、未来5条响应或实际检索结果。

Head复制官方layer35作为可训练block，接全244×2048 flatten后Linear2。block、readout都训练；输入detach，梯度不进冻结featureencoder/actor。两个输出是binary mixed logit和sigmoid×sqrt(.3)的bounded sample-std。[复制和trainable](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_qwen_block_model.py:109)、[forward及两个输出](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_qwen_block_model.py:150)。因此不能描述为“actor最后一层接小分类头”；它是独立约7808万参数的 copied-block predictor。

**可检验结构风险，非已证根因**：官方layer35原本接layer34输出，现在接layer27输出，跳过28–34层，预训练输入分布可能不同。复制参数字节正确与warmup/formal一致，不证明这条跨层接线具有预训练语义。flatten还保留绝对位置与问句长度，允许记住训练题/长度模式，是否有害须按去重题holdout及轻量模型对照验证，不能仅凭参数多认定过拟合。

### 标签、损失、预算

每题五条严格EM取K∈{0,…,5}，binary标签Z=1[0<K<5]，sample std目标`sqrt(K*(5-K)/20)`。训练loss为`BCEWithLogits(z,Z)+.3*((predicted_std-target)/sqrt(.3))²`，即分类项加一个原单位std-MSE；没有count分布likelihood、rank loss或classreweighting。[目标源码](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_token_model.py:106)、[recipe](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/reproduction/common8/qprobe-native-block-defaults.json:24)。

Warmup固定5120题及其已采样G5标签，重复10epochs、batch256/micro32、Adam LR3e−4、200更新；代码只有finite checks，没有gradnormclip调用或显式weightdecay设置。每epoch总结再次在这同5120训练集上前向；源码明确`training_only_not_independent_holdout`。[训练循环](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_qwen_block_model.py:279)、[训练summary范围](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_qwen_block_model.py:243)。不可把99.88%称为新题或fresh-G5预测准确率。

Formal按完成outer后更新head：保留当前+前7完整outer，age权`2*2^(−age)`，均值为`mean(w_i*loss_i)`而非除sumw；每outer2epochs，Adam状态继续。warmup标签在adoption后完全排除formal replay。[年龄权](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_qwen_block_model.py:185)、[fit](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_qwen_block_online.py:139)、[adoption无refit且清warmup replay](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_qwen_block_online.py:203)。8个outer每题数量相同时，raw权均值从首轮2变为满窗约.498；loss规模随窗口变，不能直接把跨outer weightedloss下降当相同目标变好。

每批预测冻结在rollout/fit之前，prequential指标随后对实际G5算，完成actor outer后才提交head更新；这点代码顺序正确。[冻结与prequential](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_runtime.py:350)、[完成后更新](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_runtime.py:393)。replay记录policy snapshot，固定缓存却不随actor变化；不同snapshot的标签会被汇总到同一question表示，形成近期加权目标，未显式输入actor状态。[replay路径](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_token_online.py:285)。这可能追踪滞后，未证明已经导致当前失败。

## 已证观察与不能推到的结论

1. **原题、原标签拟合可复现**：独立head200原5120 inference重现5114/5120=99.8828125%，TP1843/TN3271/FP4/FN2。分类logit范围−112.2654到138.3607，p精确0有2、精确1有91。[control](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-head200-control-read-1792083940623678000.json:31)。这支持“保存/接入后原题分类未坏”，不证明泛化或概率校准，也不能单凭高训练准确率证明记忆化。
2. **正式首轮是selected Top512 precision**：小摘要记录261/512=50.9765625% true mixed，全部selected预测≥.5；不是随机512题的全域accuracy。[formal小摘要](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-selection-metrics-fix-native-progress4.json:1)。其候选5120中176题见过warmup，selected53见过；这53原全mixed，fresh G5仅37仍mixed、16翻转，K变化39。其余459新题有224 mixed。[同题fresh-G5审计](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-same-question-G5-label-and-std-audit-20261016.json:6)。不能把训练99.88与Top51250.98直接相减当域泛化gap；题池、选择条件和G5标签实现都变。也不能拿warmup总体36.04%作这新候选池的随机基线。
3. **STD支路失败是独立事实**：原5120的predstd min=max=0，精确0有5120；target variance平均.0884375，control regressionMSE及contribution约.0884375。[std审计](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-same-question-G5-label-and-std-audit-20261016.json:25)、[control loss](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-head200-control-read-1792083940623678000.json:56)。分类正确率不能覆盖该项。当前forward是sigmoid，不是clamp；极负rawlogit、数值下溢、共享特征优化失衡等是待检假说，缺该支路rawlogit/grad，不能确认哪一个。随机G5标签本身不能解释为何全dataset的回归输出皆0。
**std支路的时间线补核**：已保存小metadata的`observation.files["warmup/warmup-metrics.json"].metadata.epoch_history`含11个epoch统计：epoch0/Adam0的std-MSE=.15825710585340858、MAE=.33612060546875；epoch1/Adam20变为std-MSE=.0884374997811392、MAE=.17760460078716278；epoch2–10这两个值和regressioncontribution逐项完全相同。分类accuracy同期从epoch1的.3751953125提升到epoch10的.998828125。说明训练调用了joint loss并实际更新200次，但分类成功掩盖了std支路长期无改善；并不能说未fit整个head。结合原head200全0实读，std从首epoch起处在与零预测吻合的平台，**没有中间输出数组时不宣称epoch1已逐题exact0，也无法定位是哪次Adam或确认sigmoid underflow**。[warmup小metadata原receipt，JSONpath如上](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-json-binding-repair-completion-observation.json:1)，其远端warmup-metrics原文件SHA=`95e878a16b095af5f605534eaa020ea7746ddca7110b1c95b0169ae57b7b8557`；此本地receipt SHA=`8b2acf4c75e6fa14cb48b949f4d6e9db0a58c34e823e2b534ca13e0b8b5832cf`。没有std rawlogit/gradient字段。

4. **排序饱和已证**：首轮候选只有8个p=1，selected分数范围.8874–1，边界尚可分；一次online fit后第二轮候选1412个p精确1，selected512全部1，与unselected顶分相同。[排序审计](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-native-new512-existing-ranking-overlap-20261016.json:7)。实际selector按p降序、qid SHA打平局，std不参与、exploration=0。[selector](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_selection.py:225)。因此第二轮topset在这1412中由hash定，不能称严格概率排名。用rawlogit排序可避免sigmoid舍入平局，但不自动修好logit失准、过拟合或目标；必须新副本明示这种变更，不能热改。
5. **首轮formal fit后的损失/排序健康警报需正视**：保存摘要postfit replay accuracy .552734、weightedBCE13.9965，headAdam200→204；分类accuracy相比同批prefit .509766实际略升，不能把它称为准确率恶化。高BCE和后续饱和是健康检查线索，但缺同批同权重pre/post完整loss不能单凭这一个数认定优化发散。这是选中512题上2epochs的近期weighted训练summary，不是独立holdout。新的随机512测试尚待负责线程实测结果，不能在此预判。[formal小摘要](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-selection-metrics-fix-native-progress4.json:1)。

## 训练目标该如何校正：建议供人工决定，未执行

### 优先识别可预测对象

问题本身不是永久mixed。定义固定actor版本、检索服务、生成预算/temperature/seed分布下

`m(q,theta)=Pr(新一组G5出现0<K<5 | q,theta,protocol)`。

当前Z是这个事件的一次伯努利观测，BCE在population上可以学m；反复训练同一Z不能把其不确定性消掉。若五次独立同分布、每条正确率p，则`m=1−p^5−(1−p)^5`；检索/采样相关性存在时该式不是定理。回归的population最优目标为E[observed sample std]，通常不等于sqrt(E[sample variance])，不能把估p后的sqrt(p(1−p))冒称完全相同目标。对固定actor每题重复独立G5取组事件均值，BCE可用其softlabel；或者收集正确次数，用Binomial likelihood估p再推m，同时明确i.i.d.假设。比较更灵活的6-count categorical likelihood（pi_K，mixed=sum pi_1..4、E[sample std]=sum pi_K*sigma_K）可避免两输出不一致，但单次G5训练六类也可能更难，不能未验便认定优于BCE。

### 最小分阶段路线

1. **先等当前fixed-actor/new512测试完整验收**，绑定同actor/protocol/features/head，不拿初始化或首条progress当完成。按题身份/规范化内容去重，训练/校准/最终holdout分开。新题holdout与同题fresh-G5复测分开报告，避免一项混合解释全部。
2. **保持fixedactor，不先扩大模型和epochs**：用原cached features和新增标签，在独立诊断目录比较常数prior、轻量BCE、count/Binomial基线与现nativeblock。先记录binary rawlogit、std rawlogit、范围/梯度/参数更新，定位全0std和分类饱和。LR/clip/weightdecay/冻结block等如需改变，预先给最小单因素配置；现代码LR3e−4不等于已证明学习率根因。
3. **如果采集预算允许，先增加每题独立G5重复而非重复epoch**：固定选定训练/校准题，重复seed组估m及其噪声，actor/protocol不变。同题重复共享题ID，split不能随机拆成行。当前53是selected原mixed条件样本，16/53翻转不是全题噪声率或Bayesaccuracy上限；用无选择随机/分层题集重估。
4. **以实际selection效益选模型**：在同一冻结候选池上比较Top512 mixed precision及对随机512的uplift、group label整体BCE/Brier/可靠性、分层PR-AUC、count/STDMSE、tie数量；不要以训练accuracy作checkpoint selection。检验数据集/长度/难度区间的校准；采集随机探索题才可能估全池泛化，否则selected-only replay形成反馈偏差。
5. **在线阶段另验actor漂移**：fixedactor通过后，按actor版本分组监测、先预测再采集再fit。用少量随机探索保持label覆盖，记录选择概率；近期labels同时混用时明确age权和分母。actor变化是目标变化，不是冻结feature本身失效的充分证据。重放window8/half-life1与丢弃warmup属当前设计选择，不能无对照称最优。

计划层的核心排序是：**目标噪声与heldout识别 → 数值/双头健康 → 轻量可校准基线 → 最后才更大模型/online replay**。训练标签拟合、teacher/head字节一致、fixedactor随机性、跨题泛化是四个不同验收对象，须分别验证。

## 持久小证据SHA

- head200 control JSON `a8c662a310523201a2a589681e5e95fbdcba10106ca23476aa10218c6e3e7766`
- 同题G5/std审计 `09ed73a94962252ccec3e91918352c0daf7cd6bbf159b5446a6fffe2bb974497`
- 排名饱和审计 `0abe2cdc9e7b3460e9977f2a520017d71876ad63a08a9b1df0635ecddcd1b49e`
- head/Adam/RNG相等审计 `163c9a47a47c2a488e1f11a6fc9c4a79aee003e87798b9818f27bed7c357f899`
- formalprogress4小摘要 `d1479d5a95d2058444108fe0ab09a7faf4fa33b51a4b21a552854d3155f16667`

未改任何实验实现、配置、checkpoint或运行状态。本文件供主任务统一审核/提交/推送；不把本地写notes冒充已发布或训练修复。

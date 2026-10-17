# Search-G1 训练／评测不一致与低分结果：合并事故报告

日期：2026-10-17。对象：当前 Gi Prefix20 actor-hidden 实验，并合并 G1 Prefix5、Co Prefix20 与成功 Easy Selection＋Soft Reward 的交叉审查。本文记录已发现的问题，不声称已排除所有潜在错误；修复效果尚未通过受控重跑验证。

**结论：当前实验不能作为“只改 data selection，其余完全相同”的对照。此前这样描述不准确；实现重写时没有守住原训练契约，审计与结果汇报也没有及时暴露这些差异，这是本线程的实现和核验责任。不能把当前低分直接归因于选题算法。**

**第0轮 grounding 探针确实训练过，当前复用了真实拟合结果；漏掉的是后续50／100／150轮的 grounding 重拟合流程。** 这与每20轮更新的选题正确性探针是两套模型，不能混为一谈。

本次只读源码、现有日志与现成轨迹，在远端CPU核对小文件和汇总；没有修改训练、评分器或正式分数，没有启动新推理，也没有把完整轨迹、模型或optimizer搬到Mac。用户要求整理文档不等于停训指令，本次没有取消3430。

## 1. 合并来源、实验身份及正式结果

本报告合并以下两条证据链，保留原报告，不用一份报告的计数替换另一份的不同口径计数：

- 用户指定的[跨实验审查原文](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/REPORT.md)，下含G1、Gi、Co子报告。
- 本线程[非选题实现差异审计](NONSELECTION-AUDIT-20261017.md)，以及远端CPU读取的QA5／50、actor归约、round0拟合证据。

|对象|逻辑run|实际运行源码|定位|
|---|---|---|---|
|成功Easy Selection＋Soft Reward|`15bb15365346401da1d919e9cde8f813`|`0553adec146537e07b84377ce31cba41a54f37cd`|本次效果对照；最终正式训练为BF16／FP32 master|
|Gi Prefix20／probe LR1e-5|`a008d7ce983347bbb6e81051b25a7c4e`|`c22dbf86664bcc4da86aecd702fbaaea9c381551`|本线程，3430／h100-8|
|G1 Prefix5／P10／2560候选|`387d3f6e228a488f90c418e2ba47e28e`|`0de9565c9b1c735635744f7e80a2526a2852e568`|用户指定报告中的另一实验|
|Co Prefix20＋virtual0|`e4cf46d47ae24434a177931df5530751`|`478f46315147ebb0d4d1c556fcdd356c658e117e`|用户指定报告中的另一实验|

更早纯Soft Reward源码`fe59dc164490098054879141d5421e0b418b4056`／run`db474b6abf654ad9892b02ae67ab6f86`用于补充静态核对；其旧快照未达到50，**不能拿它代替成功Easy的209／512结果**。当前checkout后来发布的文档和审计代码也不是训练runtime更新。

固定验证集均为NQ256＋Hotpot256，文件SHA256为`3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb`。

|第50轮|选中训练轨迹strict EM|固定512题正式EM|固定512题正式F1|比成功Easy少答对|
|---|---:|---:|---:|---:|
|成功Easy＋Soft Reward|约49.60%|209／512＝40.8203%|47.9996%|—|
|G1 Prefix5|1354／2560＝52.8906%|157／512＝30.6641%|37.4141%|52题|
|**Gi Prefix20，本线程**|**2053／2560＝80.1953%**|**134／512＝26.1719%**|**32.3751%**|**75题**|
|Co Prefix20＋virtual0|1257／2560＝49.1016%|199／512＝38.8672%|46.5439%|10题|

成功Easy第50轮记录来自实际`logs/validation-identities.jsonl`，该条记录SHA为`849eacd666efb979a0281e5956ee2161649fd9d39b508e05e0af891338dad14d`，见[同轮原始摘要](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/exact-iter50-comparison.json)。其训练200轮完成，但最终另有`OWNED_CLEANUP_FAILED`记录；效果对照可引用，不能写成全部收尾验收成功。

这些是不同实验的已发生结果，**不是单因素受控对照**。选中训练集是随机G5且已被筛选，固定QA是未参与筛选的greedy单轨迹，80.20%与26.17%不能直接相减当作泛化损失或实现错误证明。下面的实现和逐题输出证据才支持故障判断。

## 2. 问题清单与影响边界

|编号|已发现的问题／差异|证据级别|影响范围与不能声称的结论|
|---|---|---|---|
|I1|训练保护information外壳并采用新context预算；QA却回退旧raw500截断|源码、真实输入和输出均确认|QA5／50已出现大量格式拒绝；未量化修复后能恢复多少EM|
|I2|actor从旧micro2等权masked mean变成local16整体token mean|源码、实际binding与同批权重统计确认|从首轮起改变长短轨迹权重；尚无梯度／EM因果量化|
|I3|奖励grounding探针固定round0，缺50／100／150刷新链路|源码、实际config与reward版本绑定确认|第50轮后的reward生命周期不一致；**不能用它解释QA50之前已出现的差距**|
|I4|custom actor绕过旧before／after-minibatch KL与负优势tail monitor调用|调用路径确认；旧manifest声明启用|DualClip／gradclip／finite仍在；旧实际逐步监控和触发历史未全查，不能说已证明它导致下降|
|I5|成功Easy live QA为BF16路径，Gi独立QA为FP32；train／QA精度不同|运行manifest／QA contract确认|FP32回退来自用户要求，不能冒称偷偷改动；其分数贡献未测|
|I6|新rollout流式遇首个完整action停止，QA等待native生成结束再裁action|源码确认|是执行路径差异；尚未证明造成动作／采样分布的具体偏差|
|I7|G1／Co使用virtual0，改变全对组GRPO优势|其他实验源码和报告确认|**当前Gi没有virtual0**；不能把他组变化算成Gi错误或用其成绩直接归因选题|
|R1|曾把上述运行说成“只换选题，其余不变”|本次审计已否定|撤回干净对照与算法单因果解释；保留真实结果及异常标记|
|R2|混合比例、分类accuracy、平衡accuracy、QA EM被不充分地区分；QA50未及时完整汇报|已发生的汇报缺口|不能以高mixed比例或高训练EM代替QA有效性；本文统一口径|
|R3|第20轮只有轨迹／探针，完整actor／Adam已被轮转清理，无法按请求完整resume20|此前实际恢复核对确认|后来从完整5重走是用户明确选择；不能冒称恢复了20或拼接未保存更新|

I4要结合用户明确要求减少无端风险gate的时间线理解：减少额外检查不等于改变PPO目标，也不能拿“保留DualClip”声称整个旧monitor路径原样保留。本报告不据此要求给健康训练添加新gate。

## 3. 第0轮到底拟合了吗？——已查清，确实拟合

2026-10-17 UTC05:06远端实际读回原parent和迁移后两份manifest、`calibration.log`、`calibration-report.json`、`calibration-label-proof.json`，并对两份head文件完整SHA核对；未加载模型、未重新生成。小证据：[round0拟合来源核对](grounding-zero-fit-provenance-audit.json)。

|项目|实际证据|
|---|---|
|原始拟合run／round|`15bb15365346401da1d919e9cde8f813`／0|
|发布时刻|2026-10-11 UTC04:56:38，manifest中的实际published时间|
|冻结encoder|官方Qwen2.5-3B-Instruct，`aa8e725`官方资产目录，manifest含权重／tokenizer哈希|
|校准规模|300个不同问题、G5＝1500条原始轨迹；commit拟合使用944条searched-valid轨迹|
|commit／flip head|`Xa`，layer27，MLP256；按题分组2-fold CV AUC＝0.9051980364|
|boundary head|`Xp`，layer27，MLP64；CV AUC＝0.7774344406|
|类别计数|commit `[269,675]`；boundary `[263,37]`|
|原始拟合日志|明确记录BEST commit、boundary候选CV以及`[save] commit(...) + boundary(...)`|
|两份head|commit 6,746,246B、boundary 1,629,483B；原目录和迁移目录的实际SHA均与原manifest一致|

原manifest位于远端：

```text
/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011/artifacts/
round_0-attempt-22c55c34f646454f9c1f158ebdeb59c2/version-manifest.json
SHA a3eaa7213f63553bf09316063b031b43adc825f5555e9af16b711af7eab1095d
```

当前pin的迁移manifest位于同后缀的`search-g1-speed-resume5-0553ade-20261012/artifacts/...`，SHA为`924e5487d19cf667dee56ae0a76eba7143cd6eb6325f1024989bac5d05efbb61`。其`restoration_origin`明确给出原路径／原SHA，且`new_calibration_generated=false`。

两份head实际SHA：

```text
commit_probe.pkl   c48f4337a5dc78506febaf8fdf745b1b155f5bab39efd44e8d4170b55b4f1654
boundary_probe.pkl c42564a7ab4b81d86551c4aaccf4bba172478f698fb28499fcba65bd7772fe42
```

因此正确说法是：**parent实际做过0轮拟合 → 0553恢复时导入 → 当前a008再次复用；当前run没有重新拟合0轮，但并非拿未训练head计算奖励。** CV AUC不是当前QA分数，也不证明对后续actor始终有效；本次核对亦不冒称逐tensor重新验证encoder或重做GPU推理。

### 真正缺失的是后续生命周期

旧common8规定rounds为`(0,50,100,150)`。checkpoint后调用`refresh_after_checkpoint`，暂停旧服务，基于相应actor重新校准／拟合两个grounding heads，再将`reward_fn.expected_probe_version`切换到新版本。因此round50用于后续51起，不用于已经完成的1–50；200结束后无需拟合一个不会使用的新奖励版本。

当前新launch只从config加载固定manifest，每轮EcoRewardManager继续要求同一个`15bb...:0:a72f...`版本；没有旧的refit／adopt分支。每20轮训练的是**选题正确性probe**，补不上grounding reward probe的50轮刷新。

用户指定报告还核对了Co：第48–50轮实际用round0，第51–55轮实际用同run的round50；存在`BOTH_PROBES_FITTED`、`REFIT_ADOPTED`及版本账本。这是实际发生的刷新证据。成功Easy自身所有刷新轮的逐次日志闭环尚未在本合并审计中全部复查，不能用“旧代码有调用”替代其每次成功的证明。

## 4. QA低分：已经定位到的具体故障链

训练采用`context_budget.observation_tokens`，截断正文但保留`<information>…</information>`。QA按照回退后的旧规则只取原始前500 token，经常切掉闭标签。模型随后续写尚未闭合的检索文本，补`</information>`，再回答；原strict解析器把模型自己输出的环境保留标签视为非法前缀，拒绝最终action。

**解析器本身与旧源码相同，不是本次新改评分器。问题是新训练输入已变，QA输入却留在旧协议。** 应修复并对齐交互输入，不能放宽评分正则，把诊断答案加回正式分数。

### 本线程Gi的完整512题分解

|指标|QA5|QA50|
|---|---:|---:|
|正式答对|121|134|
|有合法答案但错误|240|222|
|strict解析无合法最终答案|151|156|
|上述拒绝中有完整非空answer标签|136|149|
|检索observation缺闭标签：次数|175／542|166／527|
|任一observation未闭合：题数|157|154|
|这些题中strict拒绝|145／157|132／154|
|其他题中strict拒绝|6／355|24／358|

QA50的156条拒绝中，149条已经包含answer，只有7条没有非空完整answer。149条的直接拒绝规则：116条reasoning前缀有保留协议标签，33条最终policy span不恰好含一个action。

58条被拒绝输出的最后answer标签内容匹配gold，但其中可含多action，不能直接判合法。更严格的定位：115条拒绝输出在模型补完最后一个information闭标签后，剩余部分能被**原解析器**解析成一个answer；其中43条匹配gold。这是对现成输出的诊断，**不是重跑结果，不能将134改写成177或192**。

训练50的2559次information observation缺闭标签数为0；855次发生裁剪但外壳保留。同轮训练2560轨迹中仅1条没有strict合法answer。训练／QA的格式行为显著不同，且QA5已存在，不能说50轮才突然坏掉。

### 合并报告中的131／119与本线程154／115不冲突

用户指定Gi报告统计“**最后policy前一条观察**未闭合”是131；本线程统计“**任一观察**未闭合”是154。其119条是“前一观察未闭合＋模型补闭标签并给answer＋strict拒绝”；本线程115条进一步要求移除环境续写后能严格解析成一个合法answer。最后policy含闭标签的124条与本线程限定拒绝规则下的115条也不是同一集合。

不同口径不能取平均、互相加总或把119当额外正确题。两条独立扫描共同确认QA50：134正确、156严格拒绝，其中149含answer；训练2053正确、仅1条无合法answer、855次裁剪保留外壳。

### G1／Co交叉证据

G1 QA50正式157正确、153严格拒绝，其中139有完整answer。诊断抽出最后answer后有52条匹配gold，其中50条同时满足“未闭合观察＋模型补闭标签＋单answer＋无search”。其训练可解析answer为2551／2560，QA为359／512。**诊断209不是正式成绩，也不能宣称修好必然回到旧Easy的209。**

Co训练／QA共用新预算和外壳保护；本次没有找到与G1／Gi相同的内部格式错配。Co仍有相对旧Easy的协议和virtual0变化，不能因为其更接近旧分数就当成控制变量实验。

以上已证明具体输入缺陷与拒绝机制；难度、长度等也可能共同变化。尚未对同一checkpoint做控制其他参数的协议交叉生成，故**没有把全部75题差额归因于外壳，也没有证明算法无影响或必然有效**。

## 5. actor数学目标确实变了

旧actor对本地16条轨迹拆成8个二行micro，各micro内部先对有效policy token取mean，再对8个mean等权平均。新custom actor把本地16条的policy token整体取mean；OOM拆micro时再按token质量加权，继续保持这个新目标。

令第b个旧micro有T_b个有效token，逐token PPO项为ell_bt：

```text
L_old = (1/8) * Σ_b [ (Σ_t ell_bt) / T_b ]
L_new = (Σ_b Σ_t ell_bt) / (Σ_b T_b)
```

T_b不同，两者就不等价。物理batch增大并不要求改变数学权重：一次forward16条后仍可计算8个二行masked mean再等权汇总，**不需要为保持旧目标增加前向或梯度累积**。用户要求提高GPU利用率不构成这次目标变化的理由。

实际outer50的2560条轨迹、5119turn和8rank排布已CPU核对：每条policy有效token均值71.86、范围42–166；不存在空旧二行micro。在同一当前排布上，新／旧单token归约系数比为0.72436–1.67320，权重分布总变差均值0.05080。这量化了权重改变，**不是实际梯度夹角、参数差或EM贡献**。

首个实际actor更新8rank的token加权clip比例约0.0046414，LR为8.59649e-7，global gradnorm约2.027。未见支持“首步采样概率全面错位”的大clip现象；这亦不是全程逐token概率一致性证明。

## 6. 其他必须分别记录的执行差异

### KL／负优势tail监控路径

旧`DataParallelPPOActor._run_minibatch`在启用时调用before／after `_monitor_old_policy`；新custom循环直接forward、PPO loss、backward、gradclip、Adam，绕过此路径。DualClip3、gradclip1、finite检查保留，不能把它写成完全没有数值保护。

成功Easy真实manifest声明`policy_safety_required=true`、`policy_safety_mode=inline_first_five`；尚未逐条复核全部原监控执行、阈值与触发历史，以及后续减少风险gate授权对各阶段的范围。因此这里确认**调用差异**，不写“丢监控已经导致低分”，也不擅自恢复会阻塞训练的额外检查。

### 精度、预算与action终止

旧成功训练早已BF16 policy／FP32 master，不能把BF16误报为新算法改动；其live QA也走BF16。Gi QA按用户要求回退FP32，当前train BF16／QA FP32、engine4096／4596是实际差异，尚未测其分数贡献。

新训练还预留500答案token，必要时提前final；旧QA无预留，最多4个search＋最后final。**outer50实际提前final触发0次**，故不能拿这个静态差异解释该轮已观察到的大量低分。新stream动作完成即停与QA native结束后裁切是另一执行差异，是否引入分布偏差仍待受控核对。

### 他组的virtual0，不属于当前Gi

G1／Co对全对组加入虚拟reward0，改变组优势；五个相同正reward原来归一化优势为0，含虚拟0后真实五条可获得正优势。这可以单列实验，但不是“只改selection”。当前Gi没有virtual0，不应为对齐别组临时加上或把其效果当Gi预期收益。

## 7. 当前选题探针的真实效果：也不能用高mixed比例遮盖问题

混合标签为m＝1[0<K<5]，包含K＝1；q＝1−prod(p_j)−prod(1−p_j)。平衡准确率BA＝(mixed recall＋nonmixed recall)／2，与QA EM和轨迹二分类accuracy不同。

在现成1–92轮选中512组的选择时预测中，所有q都大于0.5，因此全部被判mixed：mixed recall＝100%、nonmixed recall＝0%、**每轮BA恰好50%**。普通accuracy此时恰好等于真实mixed比例。这不证明全5120的排序等于随机，也不能把这一个已筛选子集的50% BA当无条件探针性能。

|outer|选择时平均q|选中真实mixed|全错／全对|
|---|---:|---:|---:|
|20|89.02%|251／512＝49.02%|47／214|
|50|85.57%|153／512＝29.88%|40／319|
|80|92.72%|133／512＝25.98%|101／278|

第50轮80.20%的**单轨迹训练EM**并不等于80%的题组混合率；当轮319个全对组、真实mixed只有153组。预测q偏高和组分布变化是实际观测，尚不能据此断言选择方法相对同轮随机抽样完全无增益。第80轮全池mixed为942／5120＝18.40%，Top512为25.98%，富集约1.412倍，说明“存在富集”与“概率校准／分类较弱”可以同时成立。

周期更新的同份rollout、同份actor hidden、更新前后配对结果如下。训练每周期5120组；heldout始终是固定512道题的**当前actor重新生成G5**，不是旧teacher轨迹；不参与训练，无qid／内容交集。跨周期actor和真实标签会变，不能称固定标签学习曲线。

|周期|训练BA：前→后|heldout BA：前→后|训练总loss：前→后|heldout总loss：前→后|
|---|---:|---:|---:|---:|
|20|61.26%→69.19%|54.60%→58.80%|1.33595→1.23134|1.44954→1.49669|
|40|55.38%→64.86%|54.59%→51.20%|1.56999→1.31826|1.67346→1.55984|
|60|53.66%→63.71%|52.10%→52.75%|1.50394→1.45747|1.56865→1.64397|
|80|54.54%→61.06%|51.94%→54.10%|1.69442→1.39101|1.80804→1.64684|

每周期training BA上升，但heldout并不持续上升。第80轮heldout更新后普通accuracy77.54%，低于当轮多数类基线81.45%；mixed recall16.84%，nonmixed recall91.37%。同一heldout的trajectory BCE从0.72205恶化为1.08376，group BCE从1.08599改善为0.56308。**总loss或group loss下降不代表逐轨迹概率校准也改善。**

100个Adam的逐步loss来自不同mini，不能只拿step1和step100比较声称同份数据loss下降；上表使用的是更新前后same-head eval、同一整份数据的配对复测。曲线保留在[选中组指标](curves-through92/selected512-accuracy.png)、[100步loss](curves-through92/probe-adam-loss-curves.png)、[配对复测](curves-through92/probe-paired-fit-evaluation.png)。

## 8. 改动来源与为什么没有守住一致性

|提交／北京时间|实际改动|本次判断|
|---|---|---|
|`d00e4f8811cb4a2119c7d088280f7a97cd149d7e`，10-14 20:38|context预算、保护外壳、answer reserve、提前final|后来的训练入口继承；回退QA时没有统一回退训练|
|`1f75a186eea7b769c37a919096ea0b54372e5160`，10-14 23:41|异步生成用于QA|用户回退方案保留，不能算未授权差异|
|`a8c311685af6bf2d71ba6d977515106ff164f127`，10-15 01:21|H100 BF16配置|用户要求回退QA；成功训练实际BF16另有真实manifest支持|
|`29d81af65`，10-16 19:33|新增prefix rollout|继承新context规则，形成独立训练路径|
|`4e480eb6c2c74c53d2a5ea05b44d32eb05c144b0`，10-16 19:36|custom actor与独立QA|重写actor后没有保持旧二行mean目标及完整原调用路径|
|`af3b7f399872976d5ebe8d7327779876bc570af9`，10-16 19:45|按token质量处理micro拆分|显式保持的是新local16 token mean，不是旧目标|
|`e6cc9670555b6986b9a53e89a4123cf017efeece`，10-16 19:47|整合新loop、固定grounding服务|复制了reward公式，却漏接refit／adopt生命周期|

核心失误是把“同一PPO函数／同一奖励类／同样mini与Adam次数”当成训练整体等价的充分证据；实际还取决于归约权重、交互token、终止条件、grounding版本切换及评测执行契约。实现分叉后没有逐项核对这些条件，随后又过早把运行正常当成实验正确。

CP20不能恢复的问题亦是保留策略与后续需求不匹配；有轨迹或probe文件不等于完整actor／Adam checkpoint。此前已明确告知并由用户选择从完整5重走，不能把它算成已完成20点恢复。旧3423／3424因storage假设退出等工程问题有独立终态／恢复记录；它们造成耗时，不是本次QA低分的已证实机制。

## 9. 已核对保持一致、以及仍未查清的边界

已确认：核心`core_algos.py`、`signals.py`、`qa_em.py`与所核旧源码一致；EcoRewardManager类AST一致；G5组内sample-std标准化、actor LR1e-6／57完整outer warmup、mini128／20实际Adam、epsilon0.2／DualClip3／gradclip1、refKL与entropy系数0保持。旧成功运行也使用sampling-journal PPO分母，不能误报成旧HF重算、新vLLM分母。

当前未发现：QA50加载初始权重代替checkpoint50、probe LR1e-5误传给actor、固定512身份被换、以F1冒充EM、prefix token／其采样logprob被整体丢弃。没有找到不等于证明所有相关细节逐项无误。

仍未完成：同checkpoint协议交叉推理；旧／新归约实际梯度差；动态grounding版本对reward／优势的影响量；旧monitor逐步触发核对；精度／终止差异的受控贡献；早先训练前512基线差异的全部单因素归因。不能把这些未完成项写成已定位根因，也不能声称当前已经穷尽所有错误。

尤其不能得出：“高mixed保证QA更好”“所有低分都是parser”“补grounding refit能追回QA50差距”“修改解析器就算修好了”“修后续训能抹去之前优化目标的影响”。

## 10. 修复与结果处置边界（尚未执行）

1. **先固定比较基准。** 若目标是仅改选题，优先复用成功Easy的actor、reward生命周期和评测入口。物理batch可以放大，但数学mask／归约权重不能随之改变。
2. **对齐train／QA交互契约。** 对既定旧协议应统一执行；若决定改为保护外壳的新协议，则明确是第二项变化，并重测对应baseline。不得只给新模型改QA后宣称与旧成绩公平。
3. **恢复grounding round0／50／100／150的拟合、版本adopt及恢复绑定。** 当前round0可复用真实资产；后续应随actor checkpoint刷新。选题probe周期20保持为独立配置。
4. **同一个已保存checkpoint做小范围受控协议验证，再复核完整512。** 输出合法率、未闭合观察、模型续写环境文本、action数与strict EM并列；现有正式成绩原样保留，诊断宽提取值不得替代。
5. **保留问题运行，标记结果受实现差异污染。** 已有actor包含历史更新影响，修后续训不是fresh干净对照；是否重新训练应由用户另行决定。本次没有擅自修复／重启或新增GPU测试。

本报告是一份可复核的问题清单及证据合并，不是“算法失败”结论，也不是“已修复成功”声明。

## 附：证据索引

- [合并输入原文及SHA清单](consolidated-audit-sources/manifest.json)：保存用户指定报告、本线程旧报告和G1／Gi／Co子报告的未改写副本及原路径／SHA。
- [成功Easy与三组iter50原始摘要](consolidated-audit-sources/exact-iter50-comparison.json)。
- [round0原始拟合／迁移／head SHA实际读回](grounding-zero-fit-provenance-audit.json)；审计工具：[grounding_initialization_audit.py](../../reproduction/prefix_selection200/grounding_initialization_audit.py)。
- [实际outer50归约权重与reward binding](nonselection-actual50-cpu-audit.json)。
- [QA5／50完整格式统计](nonselection-qa5-50-shape-audit.json)、[严格拒绝原因及gold匹配诊断](nonselection-qa5-50-rejected-answer-audit.json)。
- [全部1–92轮选择时指标及周期配对结果](selected-accuracy-all-complete.json)、[heldout当前actor来源证明](periodic-heldout-current-actor-provenance.json)。
- [跨实验Gi现成轨迹扫描](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/gi/remote-qa-train-behavior.json)、[Co实际round50重拟合与51–55采用证据](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/co/reward-refresh-actual.json)。

完整rollout、token、模型、Adam和全量预测仍在远端；上述本地文件仅代码、文档及小型证据摘要。默认`ssh mgmt-host`及retained18路由已实际成功完成本次核对，未改SSH／auth／代理；本次没有把观察连接问题当作训练退出。

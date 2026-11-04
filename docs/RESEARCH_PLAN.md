# AI 内容平台与创作者博弈课程研究计划

本项目按课程研究项目规划，目标是完成一个明确的新建模问题、可复查的均衡推导和一组有边界的数据分析。文献检索截止日期为 2026 年 11 月 4 日。计划现已执行：理论、最低内容约束与整数账号扩展、远端元数据采集、描述分析和机制回放均完成，英文报告为 AI_CREATOR_GAME_REPORT.pdf；未占用 GPU。

## 选题

**中文题目：AI 降低生产成本后，按创作者限制数量奖励能否控制内容洪泛？**

**英文题目：Can Creator-Level Exposure Rules Control AI Content Flooding When Creators Can Split Identities?**

核心问题是：平台压缩同账号追加内容的回报，创作者会减少生产，还是把资源转向更多账号？在什么生产成本、身份成本和曝光收益下，平台能够同时实现数量上限与不拆号？

该项目使用既有 Nash equilibrium、Stackelberg mechanism design、Tullock contest 和 false-name / Sybil 框架。课程贡献定位为一个具体应用问题及其可执行边界；新颖性不建立在“首次发现拆号”上。

参考图一集中于学术发表与审稿激励，可用于理解稀缺评估容量和制度激励。参考图二给出的少玩家、明确收益、可求均衡原则用于控制本项目规模。

## 理论准备

只需集中掌握四件事：

1. 用玩家、策略、信息和收益定义博弈。
2. 用最优反应求 Nash equilibrium；有严格占优策略时直接确定结果。
3. 平台先承诺规则、创作者后行动时，用逆向归纳求子博弈与平台设计。
4. 区分个体均衡、平台目标和社会福利；创作者都选最优反应，并不表示共同收益最高。

入门资料采用 [MIT 的 Nash 均衡讲义](https://www.mit.edu/~6.7980/nfgs_nash.html)及 [MIT 14.126 Game Theory](https://ocw.mit.edu/courses/14-126-game-theory-spring-2016/)。内容创作 contest 的直接基线为下表第一篇。

## 已核实的近邻文献

这是一轮定向检索，不是完整系统综述。标题、年份和发表状态尽量以原始页面为准；摘要核验与全文核验分别标明。

| 文献 | 状态与本轮阅读范围 | 对选题的影响 |
|---|---|---|
| Yao, Li, Nekipelov, Wang & Xu, [Human vs. Generative AI in Content Creation Competition: Symbiosis or Conflict?](https://proceedings.mlr.press/v235/yao24b.html), 2024 | ICML 正式论文；模型部分 | 已有 Tullock 类注意力竞争、人机竞争与 AI 采用，作为基线 |
| Keinan & Ben-Porat, [Strategic Content Creation … To Share or Not to Share?](https://arxiv.org/abs/2505.16358), 2025，2026 修订 | 原始页面标明 WWW 2026 接收；模型部分 | 已有平台先分成、创作者选质量与训练共享；避免把训练授权当全新问题 |
| Jungbauer, Li, Wu & Xiao, [The Disruption of Attention Platforms by Generative AI](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6521778), 2026 | SSRN 工作论文；原始摘要，全文访问受限 | 已有 AI 进入、质量发现池拥堵、上传费与来源相关报酬，最接近“洪泛—筛选”方向 |
| Moosavi, Malekian & Makhdoumi, [Content Generation: Human Creators, AI Learning, and Platform Design](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7301478), 2026 | SSRN 工作论文；原始摘要 | 已有推荐权重、创作者质量投入与补偿设计 |
| Montero-Porras, Smets & Lenaerts, [Enshittification of Algorithmic Recommendation](https://journals.sagepub.com/doi/10.3233/FAIA260512), 2026 | HHAI 2026 会议论文；全文 | 已有平台质量政策与创作者努力的双稳态，不能只重做“质量合作与低质陷阱” |
| Mazorra & Della Penna, [The Cost of Sybils, Credible Commitments, and False-Name Proof Mechanisms](https://arxiv.org/abs/2301.12813), 2023 | 预印本；原始摘要及相关模型 | 有限账号成本和身份选择已有直接理论，是必要引用 |
| Pan, Mazorra, Schlegel & Mamageishvili, [On Sybil-proof Mechanisms](https://arxiv.org/abs/2407.14485), 2024，2026 修订 | 预印本；原始摘要 | 不同信息和激励条件下存在防身份操纵的机制限制，不能把特定模型结论推广成一般不可能定理 |
| Burnat & Davidson, [Quotient Semivalues for False-Name-Resistant Data Attribution](https://arxiv.org/abs/2605.07663), 2026 | 预印本；原始摘要 | 按归因簇抵御重复、近重复和身份操纵已有思路；语义聚类仅作为可比较机制 |
| Wang et al., [AI Copilot for Content Creation: Resolving the Quantity-Quality Trade-Off on User-Generated Content Platforms](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6840959), 2026 | 工作论文；原始摘要 | 摘要报告数量和质量代理均可改善，应允许 AI 增强创作，不能把 AI 定义成低质量 |

补充经典推荐文献：[Brill et al., False-Name-Proof Recommendations in Social Networks, AAMAS 2016](https://www.cs.toronto.edu/~nisarg/papers/FNP_RecSys.aamas16.pdf)。它研究用户伪账号操纵推荐，需与本项目的创作者生产选择区分。

参考图一的 [AI and Publication Commons](https://www.insead.edu/faculty-research/publications/working-papers/ai-and-publication-commons) 和 [Buying the Right to Monitor](https://arxiv.org/abs/2604.23645) 已核实存在，均为 2026 年工作论文或预印本；[More Versus Better](https://doi.org/10.1287/orsc.2026.ed.v37.n3) 为 Organization Science 2026 文章。它们提供学术评估制度的比较案例，不替代内容平台的核心 related work。

## 最小模型与已有初步分析

一个平台先承诺规则，两位创作者随后选择总内容强度 n_i 和账号数 k_i。各账号积分为其内容强度的 alpha 次方，总分与收益为

\[
s_i=k_i^{1-\alpha}n_i^\alpha,\qquad
u_i=R\frac{s_i}{s_i+s_j}-cn_i-\tau(k_i-1).
\]

AI 通过降低 c 进入；基准质量固定，避免混同 AI 使用与低质量。n 是标准化强度，k 是连续松弛。连续模型均衡为

\[
n^*=\frac{R\alpha}{4c},\qquad
k^*=\max\{1,R(1-\alpha)/(4\tau)\}.
\]

平台同时要求 n* <= n_bar 且不拆号的条件为

\[
c\bar n+\tau\ge R/4.
\]

因此成本下降会压缩这种数量治理的可执行空间。固定规则时，成本下降并不改变该齐次基准的账号数，不能写成“AI 必然增加拆号”。

另保留严格整数的课堂展示：每人固定两件作品，选择一个或两个账号。拆分净优势为

\[
\frac{R(2-2^\alpha)}{2(2+2^\alpha)}-\tau.
\]

当额外账号成本低于该阈值，拆号成为双方的严格占优策略，但双方都付出额外成本。这能直观展示个体激励与共同收益的冲突。

完整定义、证明、平台验证成本目标和适用边界见同目录 DERIVATION_PACKAGE.md。

## 数据选择与当前可用性

优先研究 Civitai，因为生成参数、作者和发布时间贴近 AI 创作。现行 [官方 Images API](https://github.com/civitai/civitai-developer-docs/blob/main/site/reference/images.md) 提供这些字段以及互动计数；普通浏览请求应使用 cursor，不能假设 page 一定生效。接口不提供曝光、真实工时、实际收益或经济主体身份。

实际采用 [thefcraft/civitai-stable-diffusion-337k](https://huggingface.co/datasets/thefcraft/civitai-stable-diffusion-337k) 的公开非 gated 历史元数据，固定源 commit 为758d4d72c87c5177c349230f38f2e8e8e3b6a23c。HKUST3经公开镜像直接下载164,627,773字节，SHA256与LFS值严格一致。没有下载图像，也没有把原始数据保存在Mac。

原始327,138行，按image ID保留首条后得到325,599件独立作品。SFW标记185,004件；要求非空作者后，主要分析池为184,072件、12,910个账号，覆盖2022-11-09至2023-04-12。另保存固定seed42的10,000件随机样本和1,000名作者、8,730件作品的分层样本。完整数据路径与哈希记录在采集审计中，原始数据留在hkust3:/data/shaoshitong/ai_creator_game_20261104。

直连接口失败后已通过镜像完成固定版本采集，未使用Stack Exchange替代主研究对象。ImageGem因需同意账号资料共享而未获取。当前数据卡未列明许可，最终交付只含代码、汇总与图，不再分发原始记录。

## 小规模分析设计

原定5,000–10,000条的采样目标已完成；描述分析和固定池回放进一步使用全部184,072条合格元数据。账号记录数只是历史快照内的作品数，不当作完整产量。

分析单位与字段：

| 单位 | 变量 | 用途 |
|---|---|---|
| 作品 | 时间、作者、prompt、模型、互动快照、内容标记 | 描述样本、缺失情况与类别差异 |
| 作者—周 | 在明确覆盖窗口内的作品数、文本重复度、模型组合 | 观察数量分布与差异化；覆盖不完整时只报样本内统计 |
| 机制回放 | 按件、按账号凹积分、理想身份归并 | 比较固定内容池下的份额集中度和模拟拆分收益 |

依次完成：

1. 描述样本日期、抽样来源、每账号记录数、缺失和截断。
2. 对提示词进行Unicode、大小写和空白标准化，保留词汇，计算精确重复；不当作质量或身份标签。
3. 在同一作品池上回放 alpha=1 与若干 alpha<1 的规则，报告作者份额集中度及固定作品拆分的模拟增益。
4. 将 R 归一为 1，在 c/R、tau/R 上做参数扫描。公开数据不能识别这两个成本比率或真实 alpha，因此不强行拟合点估计。
5. 互动缺乏采集时刻且多数为零，因此仅报告稀疏性，不估计数量对质量或收益的影响。

已输出样本内记录数分布、提示词重复度、推荐机制回放、理论约束均衡和整数账号差异。固定池回放中，alpha从1降至0.5，前10%账号份额由67.33%降至32.49%；有至少两件作品的账号单边拆号，中位毛份额增益为原份额的41.41%。这是对假设规则的计算，不是实际推荐效果或观测到的账号操纵。

不做“2025 年 7 月 15 日 YouTube 全新 AI 禁令”的 DID，因为 [YouTube 官方解释](https://support.google.com/youtube/blog/356734251/response-to-creator-questions-about-ypp-policies-july-2025?hl=en-CZ) 将其说明为既有规则的澄清与更名。Stack Exchange 禁令 DID 亦已有很近的 [2026 年研究](https://arxiv.org/abs/2607.04601)，不作为本项目的主要创新。

## 执行顺序与交付

| 阶段 | 工作 | 完成标准 |
|---|---|---|
| 1 文献与问题冻结 | 已完成原始文献对照与新颖性压力测试 | 明确课程应用定位，不将Tullock重参数化称新理论 |
| 2 模型求解 | 已完成连续、最低内容约束、整数账号和平台成本最小化 | 推导与100组数值案例均可复查 |
| 3 数据试点 | 已完成100条先导、完整元数据下载及两种样本 | 实际字节增长、完整SHA、去重口径均记录 |
| 4 分析与回放 | 已完成CPU统计与三幅结果图 | 描述、回放、均衡模拟分开报告 |
| 5 汇报 | 已形成英文课程报告与中文讲解 | 最终PDF经逐页渲染检查 |

执行产物包括报告、推导、可复现代码、数据字典、允许保留的小型汇总和结果图。只有在文本相似性确有需要时才使用一张 GPU；不训练新模型。

本轮只读库存显示 HKUST3 的八张 24GB RTX4090 均为空闲观测，HKUST1、HKUST2 也有空闲卡；此记录不是资源预留。执行时重新选择实际空闲资源。原始数据、模型和大规模中间结果留在远端或共享存储；Mac 仅保留代码、必要文档和小型结果。实验实现与分析代码按用户规则及时提交并推送；启动后核实真实数据/分析进展，不以 PID 代表完成。

DeepSeek Harness 已通过直接 CLI 完成短模型审查，退出码为 0，无 stderr。首次较长调用曾在 360 秒后超时。返回的短审查中有一处连续模型链式求导错误，已据独立复核、正确一阶条件及 48 组参数的 48,000 次单边偏离检查排除；其整数切片核验与本推导一致。不把模型意见当作数学证明。所有文献、模型和数据分析均需保留 AI 辅助使用说明。

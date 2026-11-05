# V7：新数据、识别边界与近期原始文献

核验日期：2026-11-05。目标是检验“旧作品持续曝光＋账号积累＋生产／身份选择”，不再扩大普通 prompt 库。本轮仅公开网页、官方文档、论文及匿名公开下载；没有登录私人账号、发送消息、改动平台、下载媒体或接受 gated 条件。

## 结论先行

**已实际取得一个新的、非快手的小型纵向曝光样本，并按主任务授权扩展到公开清单全部 7 个频道入口。** 其中 6 个有观测、1 个只有空表；总计 12,157 个作品日、1,208 个频道日，2,952,758 字节。Open Channel Stats 提供逐作品日 `impressions`、发布日及频道订阅状态，适合独立的动态描述／预测补充；样本是主动公开的少数频道，不能识别账号资本的因果作用、同 owner 拆号收益或 AI 成本冲击。它不应与 KuaiRec 混成同一候选池。

另外两项强主源是：2026 年 Tube-Census 的长期订阅轨迹，以及 Stack Exchange 官方跨站账号关联与声望事件接口。前者能补账号状态的长期变化，后者能补有制度定义的跨站关联／部分声望携带；两者都缺本项目需要的完整推荐曝光及随机身份选择。Mastodon 的迁移制度是“账号资本并非全部不可携带”的明确反例，但本轮没有取得其公开迁移面板文件。

在已核验来源中，**尚未取得同时含随机 AI 成本变化、同一经济主体多个账号和真实曝光的数据**。公开 AI 上线事件可以支持特定比较设计，但不能直接当作固定质量下的纯生产成本工具变量。

## 1. 已验收样本：Open Channel Stats

### 来源、取得与文件

[来源自己的公开数据说明](https://openchannelstats.com/data)列明：频道主主动选择公开后，无登录 GET 可下载 CSV，并提供 SHA256 sidecar；私有或撤回的频道返回 404。本轮先读取[公开 manifest](https://openchannelstats.com/channels-manifest.json)，确认所取频道在该公开清单中，再取得该频道的[数据页面所列文件](https://openchannelstats.com/@theartharrison/data)。这是第三方频道主自公开的原始记录，**不是 YouTube 官方发布的研究数据集，也未由我们独立核验上游采集完整性**。

- 远端原始目录：`hkust3:/data/shaoshitong/ai_creator_game_20261104/v7/source_discovery/raw/openchannelstats/theartharrison/`。
- 小型审计：远端 `v7/source_discovery/audit/openchannelstats_pilot.json`；本地备份 `docs/V7_AUDIT/openchannelstats_pilot.json`。
- 可复现采集源码：`src/v7/discovery/collect_openchannelstats_pilot.py`。仅一个 manifest、五个纯元数据文件及其 SHA；每文件限制 5 MB，不下载数据库、媒体或其他频道。
- 实际运行：2026-11-05 11:43:24—11:43:50 UTC；进程退出 0。五文件均 HTTP 200 且逐文件源 SHA 一致；原始和 sidecar 合计 **398,768 字节**。Mac 仅保留源码及聚合审计，原始 CSV／目录记录全留 HKUST3。
- 两次进展实测分别看到前三文件完成、后两文件完成；随后独立读取审计核对源码 SHA 与本地备份相同。没有把启动成功当成下载验收。

初始样本完成后，主任务授权在来源很小时扩到全部 opt-in 频道。公开 manifest 共 7 项，带 `youtubeChannelId` 与 `contributorId`；CSV 的 handle 可在远端连到这些平台频道标识，**它们不提供经济 owner 映射**。余下 6 个频道按同样方式取得，不探测 manifest 之外的地址。

| 公开频道 handle | 作品日／不同作品 | 频道日 | 频道日报日期范围 | impressions 缺失作品日 |
|---|---:|---:|---|---:|
| `theartharrison` | 1,486／52 | 53 | 2026-09-11—11-02 | 14 |
| `a-dventures-and-travels` | 4,150／55 | 316 | 2025-05-06—2026-11-02 | 530 |
| `art.harrison` | 40／5 | 14 | 2026-10-20—11-02 | 3 |
| `braindelayclips` | 11／3 | 86 | 2026-08-06—11-05 | 5 |
| `breakwayyt` | 3,731／32 | 249 | 2025-10-18—2026-11-02 | 284 |
| `koura_yt_mfun` | 2,739／7 | 490 | 2025-07-01—2026-11-02 | 2,446 |
| `men-r3c` | 0／0 | 0 | 无观测，只有表头 | 0 |

六个非空频道具有 **12 个真正共有的观测日期**，最早 2026-10-20、最晚 11-02；该范围跨 14 天，不能把范围误写成 14 个连续共有日。七个入口的共同日期数为零，是因为其中一个为空。日期缺口、各频道报告截止和 `recorded_through` 均需按来源实际记录处理，不能扩成平衡面板并补零；后两个稀疏频道也不能冒充数百个完整 impressions 日。

全池采集汇总：`v7/source_discovery/audit/openchannelstats_public_pool.json`，其中 `channels[].root` 指向每个频道的独立目录；每频道原始文件在该 root 下 `raw/openchannelstats/<handle>/`。首频道沿用初始 root，其他频道 root 为 `v7/source_discovery/public_pool/<handle>/`。独立验收 `openchannelstats_public_pool_acceptance.json` 再次核对 **35 个源文件 SHA** 并以日期集合求交，11:58:26 UTC 退出 0、`accepted=true`。两份小型审计均已备份到本地 `docs/V7_AUDIT/`。

首次全池汇总因空频道没有日期范围而退出 1；所有文件此前已下载并通过各自校验。修复只让汇总跳过空日期，并读取已经取得的文件，不重新下载、不删掉空入口，也不改变任何分析样本结果。本轮没有拟合，因此不存在按拟合效果改模型。完整采集入口及独立验收源码为 `collect_openchannelstats_public_pool.py` 与 `verify_openchannelstats_public_pool.py`，与初始采集器同在 `src/v7/discovery/`。

| 文件 | 实测范围 | 直接可用字段 |
|---|---|---|
| `videos.csv` | 53 个作品；发布日 2026-09-11—11-04 | `video_id, sequence_number, published_at`；累计 views／impressions、订阅转化等。累计字段不能回填到过去训练时点。 |
| `video_daily.csv` | 1,486 行，52 个作品，2026-09-11—11-02 | `channel_handle, recorded_through, video_id, date, days_since_publish, views, engaged_views, watch_time_minutes, impressions, ctr, subscribers_gained/lost`。 |
| `channel_daily.csv` | 53 天，2026-09-11—11-02 | `videos_published_that_day, total_views, total_impressions, subs_gained/lost/net, subs_total_eod`。 |
| `README.md, SCHEMA.md` | 已留存源版本；schema 87 | 指标字典、截断日期、报告延迟与数据质量说明。字典称 `subs_total_eod` 为日末订阅数；其是否完全由逐日原始快照观测、是否包含重构，本轮未独立确定。 |

视频日表的 impressions 缺失 **14 行**、为零 **325 行**；CTR 缺失 **339 行**，平均观看时长／比例各缺 **1,151 行**。缺失不等于零。未见 `days_since_publish<0` 行，但这不证明全部作品日完整。目录包含 11-04 发布的新作，而日报仅到 11-02；不可把两天报告延迟补成零，也不可将没有日报的新作当成退出。

### 测量与许可

YouTube 官方将 [thumbnail impressions](https://support.google.com/youtube/answer/9314486)限定为满足可见比例和时长条件的缩略图展示，并排除部分站外、通知及播放器内流量。这与 views、全平台所有曝光或推荐流量预算不同。来源 CSV 的 `impressions` 仍应按来源字典解释，不能仅凭列名声称它是完整推荐分母。官方 [Analytics 指标定义](https://developers.google.com/youtube/analytics/metrics)还区分频道整体与逐视频 watch-page 订阅转化，不能机械把逐视频 `subscribers_gained` 相加冒充所有订阅来源。

[来源条款](https://openchannelstats.com/terms)区分两类权利：衍生表／文字 CC0，YouTube 原始数值受其 API 服务条款及频道主公开授权约束，不在 CC0 内。源 SHA 验证的是取得文件与来源一致，不证明数据无误或具有全平台代表性。README 的 `Privacy mode: owner` 不能被当成我们使用了 owner 登录：本轮请求无 token、cookie 或私人账号，且频道先由公开 manifest 确认。

### 可接续的小分析，尚未由本代理拟合

这一样本可比较旧作日曝光的持续与衰减、发布新作前后旧作曝光的关联。应先冻结独立的小协议，只用当时已发布、已有记录的目录；无记录／缺失保留缺失，逐视频日报不能被累计目录汇总泄露到过去。

**单频道条件份额还有一个精确的代数限制。** 若同日同一频道所有作品的权重都乘公共账号项 `w(H_at)`，则

\[
p_{v\mid a,t}=\frac{w(H_{at})g_{vt}}{\sum_{u\in a}w(H_{at})g_{ut}}
=\frac{g_{vt}}{\sum_{u\in a}g_{ut}}.
\]

该账号项完全约掉，不能从这种日内条件份额估 `beta_H`，也不能估新旧账号权重 `r`。如果改预测频道总 impressions，虽不再代数约掉，却仍混淆全站需求、趋势、质量与频道状态。多频道有日历重合只使公共项不再在所有账号之间一起消去，**不会自动创造因果识别、同 owner 关系或完整市场预算**。七个 opt-in 频道不是共同随机候选池，不能把其总和当平台给定的完整 R。

后续分析由主任务／数据代理决定；建议先只做旧作持续与发布关联，不把本轮取得样本写成已识别账号资本或均衡证据。

## 2. 至少三项强主源及获取成本

| 主源与日期 | 字段／取得方式 | 可检验什么 | 不能识别什么 |
|---|---|---|---|
| **Open Channel Stats**；网站条款更新 2026-06-20，本轮源 build 2026-11-05 | 逐作品／频道日报，匿名公开 GET＋SHA；初始0.40 MB，公开清单全部7入口合计2.95 MB，6非空。 | 同作品曝光持续、新发布与旧作流量的描述关联；在明确对象和信息集下的条件预测。 | 随机数量压缩回报、H 的因果反馈、owner、AI成本。单频道份额中的公共w(H)精确约掉；多频道也不是完整市场。 |
| **Tube-Census**，Eggleston、Handler、Pacheco，arXiv 2026-05-07；[原文](https://arxiv.org/html/2605.06999v1)、[Zenodo 数据](https://zenodo.org/records/18267682)、[作者源码](https://github.com/blast-cu/tubecensus) | 2006—2023 归档的频道标识及订阅计数时间序列；论文明确无 video-level 数据。CC BY 4.0。全发布 16.4 GB，`cohorts.csv` 851.9 MB、最小完整 census 文本 83.1 MB；本轮未下载这些包。 | 账号增长、饱和与不同时间尺度的预测持久性；为账号状态选取外部长期候选样本。 | impressions、投稿流／旧作曝光、多个频道同 owner、迁移或 AI 成本。归档密度／知名频道覆盖有选择，别把缺快照当零。2026 是论文年，不是观测年。 |
| **Stack Exchange 官方关联账号与声望接口**；2026-11-05 核官方文档 | [`account_id`→associated accounts](https://api.stackexchange.com/docs/associated-users)，返回各站 `user_id, creation_date, reputation, question_count, answer_count`；[事件类型](https://api.stackexchange.com/docs/types/reputation-history)含日期／post_id／声望变化；少量公开 API 无需读取私人账户，bulk 则另走官方 dump／SEDE。本轮没有再下载全 dump。 | 同一网络账号的跨站技术关联；账号年龄、帖子生产及声望事件；制度上的部分资本携带。 | 不是同平台 Sybil 全集，不保证法律／经济 owner；声望不是曝光或收益。部分事件私有／被清理，事件累计不必等于当前声望；投稿删失及跨站需求差异仍在。 |

Stack Exchange 的[官方声望规则](https://stackoverflow.com/help/whats-reputation)明确：满足资历门槛时，其他站点可得一次 +100 association bonus；其他声望主要由本地帖子／投票积累。这是“技术关联可观测、资本部分携带而非全有全无”的制度例子，不是随机迁移实验。API 中 `account_id` 与 `user_id` 必须区别，不能拿观看者标识替换创作者身份。

### 有用但未取得可用行级面板的来源

- **Mastodon 迁移**：[官方规则](https://docs.joinmastodon.org/user/moving/)（页面更新 2025-11-27）允许迁移 followers，旧 posts 不迁移；[Account 实体](https://docs.joinmastodon.org/entities/Account/)公开 `moved, created_at, followers_count, statuses_count` 等定义。真实双向 alias／Move 能证明一定的技术控制关系，仍不等于经济 owner。2024 年 IMC 原文 [Fediverse Migrations](https://tysong.github.io/files/IMC24-Switch.pdf)研究 8,745 名迁移用户，但所查全文未见可直接取的行级数据下载入口。本轮未取得该面板；当前官方 lookup 文档列用户 token 权限，不将其笼统写成保证匿名可抓，不注册／登录账号。即使能取公共 profile，也无历史 impressions，迁移选择／服务器变化不外生。
- **Global YouTube Trending 2022—2025**：[作者原始论文](https://arxiv.org/abs/2510.23645)于 2025-10-24 提交；[Illinois Data Bank](https://databank.illinois.edu/datasets/IDB-9307654)是论文数据入口。原文列出 snapshot、country、rank、video／channel ID、publication date、views／comments。它可补跨作品重复观测，但只覆盖 Trending 入选者，没有未入榜分母／真实 impressions。本轮数据页直接访问 403，未下载，不冒称已验收数据。
- **AI 上线事件的公开复现**：Burtch、Lee、Chen，2024-05-06，Scientific Reports，[原文与 Data availability](https://www.nature.com/articles/s41598-024-61221-0)。原文指向 [OSF qs6b3](https://osf.io/qs6b3/)的 Stack Overflow／Reddit 数据及脚本；SimilarWeb 访客数据明确不公开。OSF 本轮浏览 403，文件清单／字段未验，不把论文的公开声明当成本轮取得。ChatGPT 上线同时改变需求替代和供给工具，不能当只改变 c 的纯冲击。

## 3. 对 V7 四类理论命题的识别判断

| 理论对象 | 新来源可以推进的验证 | 仍须另有的设计或数据 |
|---|---|---|
| 账号内数量压缩的边际回报 α | Open Channel Stats 可看新发作品数、旧作与总 thumbnail impressions 的时间关系；有真实零值和缺失区分。 | 投稿内生、质量／时间趋势与受众状态共变。需随机发布槽／可辩护的外生规则变化或结构限制；一个预测系数不能识别平台政策 α。 |
| 同 owner 多账号／资本可携带 | Stack Exchange 关联账号和 Mastodon Move 提供有制度依据的技术关联；优于显示名／prompt 相似。 | 跨站点环境变化、可迁移受众和本地资本各自测量、迁移时点外生性；技术控制不等于经济 owner，尚无同平台未披露 Sybil 的完整映射。 |
| AI 外生成本冲击 c | 公开 AI 上线论文提供真实前后事件与比较思路。 | 必须区分本人降本、竞争者进入、需求替代和质量改善；本轮没有取得只改边际成本且固定质量的随机平台数据，不校准 c。 |
| B／H 动态状态 | 新样本分别观察作品日轨迹、发布流及频道订阅状态；Tube-Census 提供长时间尺度。 | 单频道日内条件份额的公共H项完全约掉。不能由持久性识别 `H'=δH+λp` 因果反馈；订阅者不必等于平台隐含资本，日末值不可作为同日事前状态，折旧、需求和质量仍混淆。 |

V7 当前两期模型的状态转移是**明示假设**。数据能约束“是否存在旧作持续流量／是否必须考虑账号状态”这类描述，不能直接证明特定 stock-and-flow 方程、均衡或参数的结构含义。

## 4. 2024—2026 近邻原始文献与创新边界

| 原始来源／日期／状态 | 已核实内容与对本项目的约束 |
|---|---|
| **Kim, Sueyoul; Jin, Ginger Zhe; Lee, Eungik. Does Generative AI Crowd Out Human Creators? Evidence from Pixiv.** NBER Working Paper 34733，2026-01；[主页面](https://www.nber.org/papers/w34733)，DOI 10.3386/w34733。工作论文，不标正式期刊发表。 | 主摘要研究图像 AI 上线后非主要 AI 采用者的投稿、bookmarks 和题材转移。它已覆盖真实 AI 进入与创作者战略反应，不能将泛泛的“AI 使人类减产／改变题材”当新命题。bookmarks 不是 impressions；摘要不能证明纯降本机制。本轮 PDF 403，未据此声称全文审查或公开原始数据已可下载。 |
| **Sun, Yushi; Sun, Bo. From exposure to followers: A stock-and-flow closed-loop framework of creator dynamics.** Information Processing & Management 63(5), 104677 (2026)；[出版者原文预览](https://www.sciencedirect.com/science/article/pii/S0306457326000695)，DOI 10.1016/j.ipm.2026.104677。 | 原预览给出作品 views 到 follower stock、转化／流失／饱和，及 1,015 名 YouTube Shorts 创作者的发帖与日粉丝验证。与曝光→账号积累主线高度相关，**不能声称首次把 follower stock 与曝光闭环**。其 exposure 的操作化包含 post watch counts，不能直接当实际 thumbnail impressions。Crossref 的 published 为 2026-07，created 2026-02-09 仅为 DOI 记录建立日，不能冒称 online 发表日。本轮只核公开预览与登记，未取得付费全文或原始面板链接。 |
| **Yao, Yashu; Zhang, Zhihan; Chen, Qi; Huo, Jiazhen. Dynamic control of pure content quality and recommendation intensity on UGC platform considering reference effect.** [出版者页面](https://www.tandfonline.com/doi/full/10.1080/01605682.2025.2544870)，在线 2025-08-21；JORS 77(10), 2648–2674（2026卷期）。DOI 10.1080/01605682.2025.2544870。 | 出版者摘要明确动态质量、推荐强度、参考质量不确定性及微分博弈，故“平台与创作者动态互动”已有原始模型。与 V7 的身份成本／多账号生产状态不同，不凭摘要断言完全同构或完全不重叠。当前未读付费全文或核实公开数据。 |
| **Ohayon, Sagi; Taitler, Boaz; Ben-Porat, Omer. Content Creation with Spillovers: An Incentive Design Approach.** [arXiv:2603.14372v3](https://arxiv.org/html/2603.14372v3)，2026-03-15 首发，07-22 修订。预印本。 | 已读 §2：创作者选择努力，质量有跨创作者溢出，平台按质量分配注意力；标准 contest 可无纯均衡，提出 provisional allocation。当前题名不是旧称 Contests with Spillovers。其文中 dynamic programming 指求机制的算法，不是跨时期状态博弈。没有由所读模型发现 V7 的 B/H 两期身份选择，但不能因此声称 V7 已证明出版原创性。 |
| **Burtch, Gordon; Lee, Dokyun; Chen, Zhichen. The consequences of generative AI for online knowledge communities.** Scientific Reports 14, 10413，2024-05-06；[原文](https://www.nature.com/articles/s41598-024-61221-0)，DOI 10.1038/s41598-024-61221-0。 | 已核原始 Methods／Data availability：公开 SEDE 活动、Reddit posting 与不公开 SimilarWeb，研究提问量及账号资历变化。它提示 AI 采用既影响供给又替代平台需求；不是只有 c 变动的固定质量平台实验。 |

**建议的诚实贡献定位：** 本项目把已有 contest／Sybil 思路放入“有效旧内容存量与账号状态分开、每号最低内容、有限时期身份与生产联合选择”的课程应用问题，并计算其条件结果。旧作持续曝光、粉丝闭环与动态平台策略各有近邻原始研究；本轮检索没有证明这一组合的出版原创优先权，也不以新增状态变量本身主张新基础理论。

## 5. 交接与完成状态

小样本远端路径、字段、缺失和报告截止已直接发送数据代理；其既有 KuaiRec 协议不为此新样本改动。下一步若使用新数据，应由主任务另外冻结单频道补充协议，不在读过数据后称独立事前注册。

本轮实际完成的是 source discovery、一个公开 metadata pilot、传输 SHA／字段审计与原始文献核验；**没有新增拟合、回归、因果估计或真人／LLM 行为实验**。采集代码与小型审计已纳入本地源码备份，原始数据全在 HKUST3；Git 提交推送由主任务统一处理，本代理未提交。

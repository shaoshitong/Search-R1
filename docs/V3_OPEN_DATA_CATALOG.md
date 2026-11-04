# V3公开数据与采集来源目录

核验日期：2026-11-04。这里区分“文件公开可取”“本轮已经取得”“需要另行接受条件或授权”。数据卡公开、论文开放获取、代码开源，都不自动代表原始数据无门禁或具有同一许可。本轮仅处理元数据和汇总，不展示或下载图像。

## 1. 本轮最有用的数据：补曝光和稳定账号标识

| 来源与获取状态 | 可观测变量与适用问题 | 识别边界与许可 |
|---|---|---|
| **[KuaiRand-Pure官方目录](https://kuairand.com/)**；公开、非gated；本轮已取得官方47,432,272字节压缩包并通过官方MD5校验 | 2022-04-08至05-08；随机曝光1,186,059条，常规曝光1,436,609条，候选池7,583个视频。基础表有`video_id/author_id/upload_dt`；逐曝光日志有时间、`is_rand/tab`、观看时长和互动。适合比较同一候选池内随机与常规流量的作者曝光集中度、每曝光反馈 | `author_id`是平台创作者账号，`user_id`是观看者，不能混用或推断共同真人owner。Pure只保留候选视频，不是作者完整生产史，也不是完整观看序列。没有AI标签、生产成本、真实拆号或政策冲击。官方仓库[LICENSE](https://raw.githubusercontent.com/chongminggao/KuaiRand/main/LICENSE)为CC BY-SA 4.0；Zenodo没有另填独立License字段 |
| **[KuaiRec官方目录](https://github.com/chongminggao/KuaiRec)**；公开、非gated；本轮已取得85.9 MB的`item_daily_features.csv`镜像 | 每行是视频×日期，含`author_id/upload_dt/date/show_cnt/show_user_num`及播放、完播、点赞等每日统计。可以形成选中视频的真实日曝光面板，区分曝光量与反馈率 | 2020年数据，无AI成本冲击。底层small matrix是1,411名观看者×3,327视频的近乎完整观察矩阵，不能解释成全平台自然流量。作者日表仍只覆盖选中视频。官方仓库[LICENSE](https://raw.githubusercontent.com/chongminggao/KuaiRec/main/LICENSE)为CC BY-SA 4.0 |
| **[Civitai模型版本趋势库](https://huggingface.co/datasets/Shiba-2-shiba/civitai-model-version-trends)**；公开、非gated；V2已取得，V3补审计 | 2026-11-03模型供给快照。原始JSON里的稳定`userId`覆盖72,949/73,020个模型；124,653/124,835个版本可关联到9,681个账号。121个`userId`对应多个显示用户名，可检验用户名不稳定的问题 | 对象是模型/LoRA版本，不是发布图像；不能混作模型中同一种产量。没有共同owner、推荐曝光、成本或完整动态面板。数据卡未列明许可；不再分发行级原始数据 |

KuaiRand精确文件：[官方压缩包](https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz)，MD5 `0820331067a3784d9691136f772b35a7`。解压后的三份日志是`log_random_4_22_to_5_08_pure.csv`、`log_standard_4_08_to_4_21_pure.csv`和`log_standard_4_22_to_5_08_pure.csv`；作者表是`video_features_basic_pure.csv`。`video_features_statistic_pure.csv`是跨日期/场景平均后的统计，不能当每日面板。

根据[KuaiRand论文§3.2](https://arxiv.org/html/2208.08696v1)，候选池经过抽样和广告/spam过滤；初始随机抽取20万余名观看者，最终仅保留随机曝光至少10次的27,285人。随机干预发生在推荐列表内，候选视频均匀抽取，但准确干预概率未披露。比较随机与常规流量应使用共同的2022-04-22至05-08期间，并考虑`tab`场景差异；15个`tab`不是15次规则变化。`is_click`在单列自动播放界面表示有效观看，不能统一称为实际点击。

KuaiRec：[官方完整ZIP](https://zenodo.org/records/18164998/files/KuaiRec.zip)为432 MB，MD5 `261550d472c48eff4990fb13c0e5bcf7`。本轮用[固定版本公开镜像CSV](https://huggingface.co/hiiamkik/miniOneRec-kuairec/resolve/c644945dfa71b5d6f67ae9c7aff0eae40caf06b8/item_daily_features.csv)，SHA256 `45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b`。镜像文件85,855,213字节，CRC32 `12e08ed8`，两者与官方ZIP中央目录中的同名member一致；这不是完整官方ZIP的MD5验收，CRC32也不是密码学同一性证明。Zenodo记录标CC BY 4.0，官方GitHub仓库LICENSE标CC BY-SA 4.0；两处声明有差异，本轮不再分发行级原始数据。

## 2. 已取得的AIGC历史元数据：供给描述与口径稳健性

| 来源 | 本轮已有规模与主要字段 | 能做什么／缺什么 |
|---|---|---|
| [Civitai 337k](https://huggingface.co/datasets/thefcraft/civitai-stable-diffusion-337k) | 实际327,138行；图像ID、显示用户名、创建日期、prompt、反应计数、NSFW标记；2022-11至2023-04 | 作品数量、文本重复和反应分布；无真实曝光、稳定owner、成本。卡未声明许可 |
| [Civitai 2.5m扩展快照](https://huggingface.co/datasets/hanruijiang/civitai-stable-diffusion-2.5m) | 2,548,726个不同图像ID；用户名、创建时间、prompt、反应等；至2023-11-14；卡声明Apache-2.0 | 与V1有297,088张重合，可审计同作品字段和计数变化；缺逐条抓取时刻，不能估每日互动增长或政策DID；不是独立平台复现 |
| [DiffusionDB 2M](https://huggingface.co/datasets/poloclub/diffusiondb) | 2,000,000个生成图像记录、Discord账号标识、时间、prompt与生成参数、NSFW预测分数；2022-08；卡声明CC0-1.0 | 独立图库的生产记录、图片与近似请求口径敏感性；不是Civitai发帖，更不包含曝光、生产成本或经济owner |

这些文件均公开、非gated，已在HKUST3取得并核验源LFS SHA256。详见[第二轮数据与验证报告](/Users/shaoshitong/Documents/Codex/2026-11-04/cha-k/outputs/V2_DATA_AND_VALIDATION.md)。源NSFW标记和预测阈值不等于人工确认安全；只分析筛选后的元数据和汇总。

## 3. 公开可取的备选：本轮未重复取得

| 来源与精确入口 | 内容、访问与限制 |
|---|---|
| [KuaiSAR-small官网](https://kuaisar.github.io/)／[488.1 MB ZIP](https://zenodo.org/records/8181109/files/KuaiSAR.zip) | `item_features.csv`有`author_id/upload_time`；`rec_inter.csv`有推荐分发时间、观看时长及正负反馈，`src_inter.csv`有搜索展示及点击。2023-05-22至05-31，25,877名同时用过搜索与推荐的观看者。不是随机曝光或全平台样本；不保证完整平台曝光总体。[字段文档](https://kuaisar.github.io/detailed_statistics.html)；官网明确CC BY-NC-SA 4.0 |
| [MIND-small官方说明](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news)／[约30 MB验证集](https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip) | `behaviors.tsv`含真实展示列表的clicked/nonclicked标记和时间；50,000观看者抽样。新闻表没有创作者账号ID，用户ID是读者；不能直接研究作者产量与拆号。使用[Microsoft Research条款](https://msnews.github.io/)，不是CC许可 |
| [Adressa官方页](https://reclab.idi.ntnu.no/dataset/) | light版最小仍为1.4 GB；[官方字段文档](https://reclab.idi.ntnu.no/dataset/cxdataset_v5.pdf)列`author/publishTime`及用户页面访问。2017年点击/访问记录，缺未点击曝光分母，作者字符串也不是稳定经济身份。CC BY-NC-SA 4.0 |
| [Netzschleuder的KuaiRand转换版](https://networks.skewed.de/net/kuairand)／[Pure CSV ZIP](https://networks.skewed.de/net/kuairand/files/pure.csv.zip) | 2026-10-28加入的37.1 MiB转换版；2,622,668条曝光边，节点保留`author_id/upload_time`，边保留时间、随机标记与反馈。可作下载备用，但不是原始六CSV或原包hash，不与官方源混称；本轮官方采集推进后未重复下载 |
| [Civistash公开卡](https://huggingface.co/datasets/Hyphonical/Civistash) | 近期按日归档热门Civitai内容，有账号名、发帖时间、反应及抓取时刻，但媒体和JSON混装；没有可用的独立元数据parquet转换。本轮不下载大包。跨日期按ID去重，不能直接当同作品反复观测面板；按热门反应采样，也不是全平台。数据受Civitai及逐作品条款约束；工具MIT许可不等于数据MIT |

HF的[numberbeat6/kuairand](https://huggingface.co/datasets/numberbeat6/kuairand/tree/main)只见1K版日志和用户表，缺所需基础作者表；[salmon1802/KuaiRand](https://huggingface.co/datasets/salmon1802/KuaiRand)是已处理训练表，不能当完整Pure原始镜像。

## 4. 数据卡公开，但文件有门禁：本轮没有取得

| 来源 | 当前状态与用途 |
|---|---|
| [ImageGem-images](https://huggingface.co/datasets/MAPS-research/ImageGem-images)及[models](https://huggingface.co/datasets/MAPS-research/ImageGem-models) | 文件需接受账号资料共享条件；未同意、未下载、未借镜像绕过。卡声明CC BY-NC 4.0；图像元数据卡列4,840,400行及`userHash/createdAt/prompt`和反应计数。匿名用户名hash不等于真人owner，卡列字段也不是我们已验收的行级字段；未找到无门禁小样本 |
| [Civiverse-6M](https://huggingface.co/datasets/latentcanon/Civiverse-6M) | 公开API标记manual gated；CC BY-NC 4.0。未申请、未取得；不把公开卡或预览视为文件授权 |
| [Pixiv-AI-generation-metadata](https://huggingface.co/datasets/Wei-Yiluo/Pixiv-AI-generation-metadata) | [2026论文](https://arxiv.org/abs/2607.10538)所链接数据；auto gated，需同意共享HF资料。卡称2023-01-01至2026-01-01，`artworks.feather`与`images.feather`，CC BY-NC-ND 4.0。本轮未下载，不能宣称已核验曝光、稳定作者ID或重复观察结构 |

## 5. 论文有相关变量，但尚未找到开放数据文件

- [Scale over Preference（2026，v2）](https://arxiv.org/html/2604.01690v2)：主feed真实impressions、AI标签、创作者生产与31日跟踪，2024-06至2025-05，约1.18亿视频。全文/附录未找到data/code或下载入口；属于重要外部经验依据，不能作为本轮已可下载的原始数据。论文许可不等于平台数据许可。
- [Fong、Lam与Natan（2026）](https://ssrn.com/abstract=6885980)：YouTube纵向观看数据与AI披露实验；三位作者公开页面未找到该篇复现数据。公开的[AsPredicted预注册](https://aspredicted.org/qf4vu7.pdf)是设计，不是实验结果。全文访问403，故结论限于“尚未找到”，不是证明不存在。
- [Wei等MM2024论文](https://tysong.github.io/files/MM24-Civitai.pdf)：描述REST API＋Selenium采集，并分析views/tips；所查论文、作者页面及GitHub没有公开数据或专用采集器入口。views不能自动等同推荐impressions。

## 6. 官方接口与采集器：可设计面板，但本轮尚不可达

[Civitai官方开发文档](https://github.com/civitai/civitai-developer-docs/tree/7ae23923c333688ad70a7a4b3ee49afd62e60f82/site/reference)固定在commit `7ae23923c333688ad70a7a4b3ee49afd62e60f82`：

- [images](https://github.com/civitai/civitai-developer-docs/blob/7ae23923c333688ad70a7a4b3ee49afd62e60f82/site/reference/images.md)：匿名列表、cursor、作品ID、创建时间、生成参数和反应计数；文档未列推荐impressions。
- [posts](https://github.com/civitai/civitai-developer-docs/blob/7ae23923c333688ad70a7a4b3ee49afd62e60f82/site/reference/posts.md)：公开帖子可提供稳定`user.id`及用户名；响应可缓存300秒。
- [users](https://github.com/civitai/civitai-developer-docs/blob/7ae23923c333688ad70a7a4b3ee49afd62e60f82/site/reference/users.md)：公开`ids/query`查找；不使用需认证的`/me`。
- [官方CLI](https://github.com/civitai/cli)：Apache-2.0，公开读取有`--json --anon`；本轮未安装或调用。第三方[CivitScraper](https://github.com/moonwhaler/CivitScraper)偏向本地模型hash信息获取，并非首选图片面板工具；代码许可不能代替访问条件。

本轮HKUST3的robots与匿名SFW API探测均连接超时，Mac均TLS失败，HTTP状态为000；未取得robots内容或首批live记录，没有二快照或watcher。这是网络失败，不是已收到HTTP拒绝，也不是已确认robots允许。没有登录、使用凭据、处理CAPTCHA或更换出口继续探测。若之后接口正常，固定作品/帖子ID、稳定账号ID、请求UTC、缓存头和反应计数才能构成可审计重复快照；它仍不能补出未公开的曝光分母。

## 对当前博弈模型的证据作用

KuaiRand与KuaiRec补上了“真实曝光”这个缺口；Shiba原始JSON补上模型发布者的稳定账号ID。它们允许检验描述性的曝光集中、观看反馈和身份标签稳健性，但不能把这些结果称为已经验证了AI成本下降、跨账号owner拆分或纳什均衡。当前没有一个已取得的开放源同时包含真实owner映射、明确成本或外生成本冲击、完整产量/账号选择，以及平台随机化规则。直接检验实施边界仍需要已知身份和成本的激励实验或平台合作日志。

## 补充：AI平台浏览计数与Pixiv的views口径

[SeaArt-HQ](https://huggingface.co/datasets/lehduong/seaart-hq)是公开、非gated的纯元数据候选。固定commit `4321a586faf084ddfa6ab415caae4073af13aa67`的8个Parquet合计16,904,683字节，字段含作品`id`、`created_at`、`num_of_view/num_of_like/num_of_collection/num_of_task`、`nsfw/rating`、prompt和媒体URL；不含嵌入图像。本轮只读取ID、时间、数值与编码列，不访问媒体URL。卡没有声明许可，也没有作者ID或逐作品AI/HGC标签；平台来源不能代替独立验证的AI标签。完整下载与汇总状态见V3审计。

`num_of_view`只能称来源的浏览计数，没有文档证明它是推荐impressions；`created_at`按Unix毫秒解释，也没有足够文档区分上传和抓取时间。NSFW编码不作SFW解释。views与likes可能有不同计数范围或更新时间，尤其likes大于views的记录不能直接计算为“点赞概率”。本轮只汇总计数与异常口径，不作like/view概率或回归。

[Pixiv官方views说明](https://www.pixiv.help/hc/en-us/articles/34139670210713-What-are-view-counts)将详情页打开、Home竖直列表达到一定停留时间，以及Following页面达到一定显示时间计为views；不同后台页面还采用不同合计范围。因此Pixiv公开views也不自动等于所有缩略图展示或推荐机会的分母。[公开高赞Pixiv元数据](https://huggingface.co/datasets/sdtana/pixiv_2023-2024_over_10k_likes)虽然有稳定`user.id`、创建日期与书签数，有限JSON检查未见views或明确AI标签；它又按高赞截断，本轮没有将其作为曝光数据取得。

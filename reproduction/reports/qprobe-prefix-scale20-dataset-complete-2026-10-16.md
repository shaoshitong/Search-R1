# Search-G1 扩大前缀数据集完成验收（2026-10-16）

**数据集与冻结特征已完成，独立最终CPU审计通过。** 训练集102,400道唯一题、每题5条轨迹，共512,000条，200个完整chunk；固定测试512题/2,560条保持隔离并复用。旧5,120题/25,600条沿用原产物，实际新增97,280题/486,400条。actor与Head optimizer更新均为0，原正式训练goal仍PAUSED。

模型为官方Qwen/Qwen2.5-3B-Instruct revision aa8e72537993ba99e69dfaafa59ed015b17504d1。共同169,028题训练产物与manifest SHA `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`保持原划分；题ID、题文本哈希及归一化题文本与固定测试均不重叠。前缀取真实prompt与至多5个真实policy token，保留原position/mask和binary EM；本报告不含原题、token或轨迹全文。标签正负比例未在当前最终审计记录，保持unknown。

四个冻结encoder实际完成120,320、120,320、122,880、122,880条，共486,400条新增layer27特征/190个唯一shard；旧训练25,600条及测试2,560条特征复用、未重编码。四worker的完整冻结参数before/after一致，state SHA `9f79ee1a24699df7924b6b602254ad6430cd1602bdd1c582338c245ad8a54a96`，对应proof SHA `71e3a020843e93df288f9891cc57a14d2ba3e6a65fea83f7cb33f267cae4146f`。

| 实际阶段 | 记录 |
|---|---|
| 全部数据与特征结束 | 2026-10-16 15:05:37 Shanghai |
| launch到全部结束 | 45404.592438秒，12h36m44s（完整秒） |
| 新97,280题teacher生成 | 40345.95190649秒 |
| teacher阶段elapsed | 44058.52404052秒 |
| teacher退出回执到全部特征结束 | 937.217044秒，约937秒，包含阶段管理开销 |

这些是本次实际墙钟和生成计数，没有独立batch提速因果对照。

最终CPU审计读取1,609,834字节metadata，复核producer此前完成的full-SHA证明，检查2,537个封存文件的stat，共303,324,081,357字节。本次未重新散列全部大特征字节、未打开NPY/tensor，也未新增GPU测试。9个唯一owned出生身份、4个绑定UID/SID均已退出，无会话幸存者或owned CUDA；3个他人compute进程保持不动。该范围下accepted=true。

| 验收根 | SHA256 |
|---|---|
| dataset complete | `ea1aecb682d8faf184f7b4c32d52bb48b88aeb1bc483242993cf349c51a85fcf` |
| chunk manifest | `24e691cd359ed926f846f450c9a9abfbac8d2ef6eeecdf7e291b92adbdb26717` |
| compact manifest | `04741c9363866c1026d1469a1c63d798fc2b21da0b056aab8f98eaf2bc68c8d3` |
| composite features | `dab5e85c079619bf2ca55b2cfa9e91da392618591426efce74dad1c8db26a7dd` |
| 最终metadata manifest集合 | `6a660cc9534850bfdf3c9b2fe0fcb19fa7d329939f5d81389e534a94107ee522` |
| 独立最终CPU审计回执 | `d70a925b67eeb00d29a977612fa5d88d955d3455c4236b1396b7afc2db5b21b2` |

远端产物保留于 `/data/shaoshitong/Search-G1-a800-fresh200-0850db2977bd4801b9ced300b0ab387f`。运行源码0277、runtimefcb未热改；最终审计与HF追加实现已发布ba3a2b9850894ff83ce1f0e111989ca891c5bf35，fresh6 Git blob已核验。本地仅代码/文档/scalar/hash，大产物留远端。

HF追加由独立任务推进：Sutoonq/Search-G1-prefix20-dataset-probe-epoch68-20261016-111a2d保持private、原epoch68 archive不变，新增selfcontained typedprefix5 train512000/test2560，不上传feature/model/optimizer、不重新生成或评分。截至本报告冻结尚未宣称上传完成。此前独立小HF数据集COMPLETE不等于本次追加已完成；原正式goal不因数据验收标记complete，monitor实际PAUSED且未修改。

证据：`preparation/qprobe-prefix-scale20-final-audit-1792141000881306000.json`、`preparation/qprobe-prefix-scale20-native-observation-1792140512182782000.json`；同名JSON保存精确scalar/hash。本报告冻结待root与HF结果统一发布。

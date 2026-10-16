# Question + prefix5 联合监督探针200epoch（2026-10-16）

## 最新人工配置

本会话最新明确要求“只要prefix5训练，20包括5”。这取代上一版本question+20的数据补齐选择：参考20-token事件仅截取前5个真实policy token，本方prefix5记录直接使用，原采样事件与最终strict EM保留，**不新增teacher生成或重新评分**。

其余配置按用户给出的参考会话：question表示与同轨迹前5个policy表示，原生官方Qwen完整last block35加masked flatten和新Linear，逐轨迹正确性BCE与同事件G5混合BCE权重各1，200完整合并数据epoch，测试EMA衰减0.99。此次是独立probe warmup，不启动旧actor/PPO任务，也不修改参考会话的源码、权重或进程。

## 数据与事件

- 参考HF epoch68归档：348160组／1740800条；包含原teacher5120组／25600条。
- A800 scale20：102400题／512000条，其中原teacher同一事件已经包含在参考数据中，只计一次；新增97280组／486400条保留。
- 预计合并445440个G5事件／2227200条训练轨迹，固定随机测试512组／2560条。最终必须经过真实source转换、SHA、完整G5和测试题隔离检查，不能仅凭预计数启动。
- 同题不同rollout属于不同事件，不能按qid或文本去掉；每组必须是同次采样的原五条EM，不能拼为G10。
- 固定测试原回答不重新采样，题ID与规范化文本均隔离。参考canonical共同train169028保持原划分与字节身份。

当前实际数据来源为HF仓库 `Sutoonq/Search-G1-prefix20-dataset-probe-epoch68-20261016-111a2d`，固定revision `d794759ab9f28cf184128b227fb38f40b59fcedc`。原参考归档八part由HF CDN直接下载到A800，**没有数据集字节经Mac或H100中转**。8,389,298,946字节已完整SHA验收，实测191.392秒、平均43.8MB/s，实际观测窗口48.1MB/s；拼接SHA为 `085b039051359f341bcc123a952e2ac9500fd890331dc4f7cd2e367c81d39ced`。A800在CPU准备阶段核对归档内部成员与源身份，再裁成question+前5个真实policy token，保留原事件、EM和positions。原归档可能包含的完整回答仅作为已有输入留远端，不进入新的prefix5训练输入或发布产物；不搬参考probe权重或原feature20，不重新生成或评分。旧H100中转路线、旧slim等待wrapper均已明确被取代，旧源码、日志、已验收分片和partial保留，禁止恢复旧路线或重启旧handle。

## 模型与损失

官方 `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`。冻结完整官方编码器，读取index27 post-block detach BF16表示；编码仍使用轨迹自己的真实causal request/context，选择所有初始question request token与前5个policy token，保留原位置与turn segments。无decode-retokenize，无未来token，无截断，真实短1–4 token保留有效mask。全数据实际Qmax扫描后固定为question槽宽，尾部固定5个policy槽，padding清零再展平。

探针完整复制官方block35（2048 hidden、16Q、2KV、11008 SwiGLU、原生RoPE/RMSNorm/residual），block全参数可训练，最终Linear/Adam/私有RNG全新，不继承参考epoch68已训练Head。FP32 master、BF16 autocast。固定编码器只编码一次；合格A800既有layer27特征按实际IDs/positions/context及SHA匹配引用，不复制约300GB，不匹配的轨迹才一次新编码。新特征用远端packed BF16位存储，不保存padding；200epoch从mmap读取，不重复冻结模型前向。

对每组五条logits `z[i,j]`、原EM `c[i,j]`：

```
p[i,j] = sigmoid(z[i,j])
m[i] = 1(0 < sum_j c[i,j] < 5)       # K=1也属于mixed
h[i] = 1 - product_j p[i,j] - product_j(1-p[i,j])
L = mean_trajectory BCEWithLogits(z,c) + mean_group BCE(h,m)
```

实际使用32个Bernoulli模式的logsumexp计算等价mixed-event logit，无概率clamp、floor或detach group。没有标准差、六分类、排序损失、独立mixed head或类别重权。保持自然事件分布；“balanced accuracy”是两类正确率均值的报告指标，不是有放回均衡采样。

## 200个完整epoch与EMA

每epoch无放回遍历全部445440组，重新洗牌；global完整G5批次循环51、51、51、51、52组（平均256条轨迹），每epoch8700次真实Adam，200epoch共1740000次，四rank副本只计一次。4卡各rank完整组分配，head micro6组／30条；micro与尾批不得拆G5、重复或丢弃。DDP按实际组数归约。

Adam、weight decay0，LR按真实成功更新余弦3e-5到3e-6，无旧100epoch或5120题source批次冒充完整合并epoch。冻结feature encoder使用micro128、token budget65536，真实长度分桶且不截断；参数只影响执行infra。

初始化顺序是fresh raw探针→DDP同步→Adam→EMA精确复制。每次真实成功Adam后一次：`EMA=.99*EMA+.01*raw`。训练反向只使用raw，EMA/Adam实际计数严格相等。每个完整epoch后用同一EMA版本完整计算train与fixed-test的G5混合balanced accuracy、普通accuracy、两项BCE及混淆计数；raw在线loss单独记录，不能冒充训练后同权重指标。测试标签从不进入训练或采样。EMA是平滑方法，效果及过拟合程度由真实train/test曲线判断。

## 执行入口与资源

源码包 `reproduction/question5_ema_warmup` 包含prepare、data、encoder、packed cache、model、联合loss、EMA、完整G5更新、checkpoint与train入口。

```
python -m reproduction.question5_ema_warmup.prepare --config BASE/preparation/source-plan.json --output BASE/data
python -m reproduction.question5_ema_warmup.cache --action build-cache --input-manifest BASE/data/composite-manifest.json --output-dir BASE/features --model-dir OFFICIAL_MODEL --worker-index R --world-size 4 --physical-index GPU --micro-batch-size 128 --token-budget 65536
python -m reproduction.question5_ema_warmup.cache --action finalize-cache --output-dir BASE/features --world-size 4
torchrun --standalone --nproc_per_node=4 -m reproduction.question5_ema_warmup.train --config BASE/preparation/fit-config.json
```

当前实际使用 `reproduction/operations/a800_question5_hf_direct_warmup_control.py` 独立一次性pipeline：等待HF直接下载的原八part完整验收→A800 CPU解包、裁成prefix5并准备完整数据→现场准入四张空闲训练卡（排除辅助5/6及任何被占卡）→四encoder worker→CPU缓存合并→200epoch训练。source-plan使用 `reference_archive.parts`，不使用旧 `prepared_reference_archive` 中转输入。选中UUID后固定，父wrapper持同UUID lease贯穿缓存与训练，child通过继承FD校验，不相互抢锁。没有retriever/actor，不用5/6，不停止他人，不改SSH/VPN/auth/knownhosts。当前一次launch intent已消费；上面的阶段命令仅说明调用关系，不能再手动启动第二条流水线。后续只观察下节当前handle，SSH超时不能当作退出或重启依据。

全量数据、特征、模型、Adam、checkpoint和预测留A800远端。Mac仅代码/配置/文档/小型审计。源码Git提交push、GitHub文件SHA独立核对、A800完整Git archive文件SHA对齐后才允许启动。stage和launch回执不等于训练推进；必须检查两次真实进展及首完整epoch。

## 产物与完成条件

真实rank更新写 `train/executions/EX/updates-rank-N.jsonl`，当前进度 `progress-rank-N.json`。每完整epoch保存 `train/latest.pt`／`latest.json`，同时保留 `train/epochs/epoch-NNNN/{metrics,checkpoint,complete}.json`；每10epoch不可变保存 `head-epoch-NNNN.pt`，最终200保存完整状态。raw/EMA、原生nonpersistent RoPE buffers、Adam、四rank RNG、数据清单SHA、真实LR时钟和cursor一并持久化。

恢复只用本新run完整合格状态，不继承旧Head、不fresh0冒充resume、不拼未保存更新。失败真实partial保留，不伪回滚。观察入口只读已有小型JSON、计数、PID birth、错误类与GPU采样，不加载tensor/model、不加评测或review。

只有200个完整合并epoch、最终EMA train/test指标、完整finite/SHA状态及仅自有资源收尾实际通过，才能宣称本warmup完成；原actor200 goal保持独立，不由本次probe完成而标记完成。

## 2026-10-16 20:10（上海）当前实际执行

当前唯一pipeline handle为 `preparation/question5-hf-direct-deployment-20261016.json`，run `d7dd61bb36454e1e80c2306b5fb7d23a`／execution `694d67a5f1a04849b9f254c4a6408a03`，源码commit `0fde8e8bb59fc3bac0c6247704b2fd1bd063abf6`，924个源码文件。远端base为 `/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-d7dd61bb36454e1e80c2306b5fb7d23a`；pipeline PID1374205／birth623197341。

下载完整验收记录在 `preparation/question5-ema-joint200-slim-deployment-20261016-hf-direct-observe-1792152603445590000.json`；20:10实际流水线元数据在 `preparation/question5-slim-progress-observation-1792152604107462000.json`。当次CPU prepare PID1375129／birth623203356真实存活，未记录错误；Head更新数尚未知，未确认开始训练，不能把CPU准备或下载完成当作已完成epoch或已有成绩。

当前primary仅沿同handle有界只读观察，不重新launch：

```
python -m reproduction.operations.a800_question5_slim_progress_observer --handle preparation/question5-hf-direct-deployment-20261016.json
```

本节是该时间点的实测记录；后续进度以同run/execution最新实际回执、CPU准备产物、四rank ledger和错误/退出状态为准。

## 历史部署与源端封存记录（已被HF直下流程取代）

源端138cf9bc实际完成68批参考数据的question+5裁剪：训练1,740,800条、固定测试2,560条。派生归档4分片、490,027,548字节；原8,389,298,946字节归档保持原样。原producer在写完不可变完成记录后，最后一次进度打印因重复`checked_unix`报TypeError/exit1。这是已确认的记录错误，不能改写为原producer exit0。

修复与独立CPU finalizer发布于`10ce6b98e0479bebf8632be424e0d501e1f0e653`。finalizer实际返回0，核对69个scan、所有分片完整SHA及拼接SHA，保留原失败日志与完成记录，不重复转换。拼接SHA为`21e6e49a0864ccf3f0b8383c521eccbbe188248cf375e7c1029d14091f257595`，资格记录SHA为`0fee6fbf90eade07545f4865c0ed6d4693acc0df076b09ff71ca7adfe862a20c`。

已被取代的A800 slim流水线绑定run `f4aab3c7ec084f908ecce3f49c6b552b`／execution `449b362dafac40c69de61a68e7766f83`，runtime138cf9bc共920个源码文件已逐文件校验；source端finalizer属于独立操作源码，未热改这个runtime。该流水线原计划搬运较小派生包，等待阶段已被当前HF直下流水线技术替换；不得恢复其relay或旧waiting wrapper。更早db413等待整包的wrapper从未编码或训练，已技术替换并确认退出。旧目录、源码、日志、已验收slim分片和partial保留，禁止重启这些旧handle。

这些记录只证明数据封存与部署；不证明已有探针Adam、完整epoch或训练/测试成绩。实际进度以当前handle和远端ledger为准。

## 已完成的小型验证

2026-10-16必要CPU检查通过：联合损失/梯度数学一致、EMA、完整G5分组预算、两rank不等micro与末空rank归约、policy5布局、checkpoint恢复；真实compact与reference20截5的合成转换、part/concat/member哈希及安全解压；BF16 packed-cache/orderedID/位置/mask/标签/segments/尾padding往返和已验收复用路径零复制。未进行额外GPU验收、LLM概率扫描、新rollout或独立评测。真实数据/缓存/训练仍以远端原件与回执为准。

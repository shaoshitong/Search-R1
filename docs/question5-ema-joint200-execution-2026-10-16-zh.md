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

八个参考数据归档part共8389298946字节。通过既有H100→Mac内存→A800 SSH路径进行有界传输，Mac不保存数据；目的part支持按真实字节offset续传，完整SHA后才rename并出具验收回执。并发限制4、每流1MiB缓冲；失败保留partial与原来源。仅搬dataset归档，禁止混入原probe权重归档。解压只取canonical、原token scan和必要prefix/label文件，不提取完整回答或旧模型。

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

实际使用 `reproduction/operations/a800_question5_warmup_control.py` 独立一次性pipeline：等待八part验收→CPU准备完整数据→现场准入四张空闲训练卡（排除辅助5/6及任何被占卡）→四encoder worker→CPU缓存合并→200epoch训练。选中UUID后固定，父wrapper持同UUID lease贯穿缓存与训练，child通过继承FD校验，不相互抢锁。没有retriever/actor，不用5/6，不停止他人，不改SSH/VPN/auth/knownhosts。一次launch intent已消费时只能观察同handle，SSH超时不能当作退出或重启依据。

全量数据、特征、模型、Adam、checkpoint和预测留A800远端。Mac仅代码/配置/文档/小型审计。源码Git提交push、GitHub文件SHA独立核对、A800完整Git archive文件SHA对齐后才允许启动。stage和launch回执不等于训练推进；必须检查两次真实进展及首完整epoch。

## 产物与完成条件

真实rank更新写 `train/executions/EX/updates-rank-N.jsonl`，当前进度 `progress-rank-N.json`。每完整epoch保存 `train/latest.pt`／`latest.json`，同时保留 `train/epochs/epoch-NNNN/{metrics,checkpoint,complete}.json`；每10epoch不可变保存 `head-epoch-NNNN.pt`，最终200保存完整状态。raw/EMA、原生nonpersistent RoPE buffers、Adam、四rank RNG、数据清单SHA、真实LR时钟和cursor一并持久化。

恢复只用本新run完整合格状态，不继承旧Head、不fresh0冒充resume、不拼未保存更新。失败真实partial保留，不伪回滚。观察入口只读已有小型JSON、计数、PID birth、错误类与GPU采样，不加载tensor/model、不加评测或review。

只有200个完整合并epoch、最终EMA train/test指标、完整finite/SHA状态及仅自有资源收尾实际通过，才能宣称本warmup完成；原actor200 goal保持独立，不由本次probe完成而标记完成。

## 已完成的小型验证

2026-10-16必要CPU检查通过：联合损失/梯度数学一致、EMA、完整G5分组预算、两rank不等micro与末空rank归约、policy5布局、checkpoint恢复；真实compact与reference20截5的合成转换、part/concat/member哈希及安全解压；BF16 packed-cache/orderedID/位置/mask/标签/segments/尾padding往返和已验收复用路径零复制。未进行额外GPU验收、LLM概率扫描、新rollout或独立评测。真实数据/缓存/训练仍以远端原件与回执为准。

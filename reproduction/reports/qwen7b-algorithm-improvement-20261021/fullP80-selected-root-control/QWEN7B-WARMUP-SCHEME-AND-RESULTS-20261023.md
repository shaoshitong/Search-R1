# Qwen2.5 7B探针warmup方案与独立验证结果

[在线文档](https://chatgpt.com/space/page_f490892718108191af1ee8a691568b4a)

当前已验收方案是 **全量169028题、自然随机G5、L15、Prefix80、整组方差MSE预测**。两份独立训练模型在两个未参与选型的2048题池上，选512题的真实Mixed比例为 **64.65%～67.38%**，四次评测等权平均 **66.21%**。结果支持本方案在本次实验条件下明显改善选题能力，**不构成任意新题池都达到67%的保证**。

结果日期为2026年10月23日。这里的Mixed@512是探针选题指标；标签由五条完整rollout的strict EM计算。

# 实测效果与67%的含义

| 独立候选池 | 候选池本身Mixed比例 | 原E1 EMA | 新模型seed1 | 新模型seed2 | 两seed指标均值 |
|---|---:|---:|---:|---:|---:|
| confirmation2048 | 686/2048＝33.50% | 209/512＝40.82% | 341/512＝66.60% | 331/512＝64.65% | 65.63% |
| stability2048 | 675/2048＝32.96% | 221/512＝43.16% | 339/512＝66.21% | 345/512＝67.38% | 66.80% |

每个候选池固定2048道问题，每题固定五条真实轨迹。基线和两份新模型对同一批问题、同一组轨迹打分，然后各自选分数最高的512题。两seed均值是两个独立模型指标的算术平均，不是将两份分数融合后筛选。

严格达到67%需要至少344/512道Mixed；本次四个模型与题池组合中，345/512这一项达到该门槛。因而准确表述是“本次独立评测约65%～67%”，不能写成“保证至少67%”。

两池平均相对E1分别提升 **24.80和23.63个百分点**；相应同题配对95% bootstrap区间为 **[20.12,29.20]和[19.53,28.32]个百分点**。四个seed与题池组合均有正向点差，两池区间下界均大于零，满足事先固定的主要验收规则。区间使用2000次重采样、种子20261023，描述配对问题的不确定性，不是对任意新题池的绝对比例保底，也不是训练seed总体的置信区间。

# 方案的关键设计

| 设计 | 当前实际实现 | 对选题任务的作用 |
|---|---|---|
| 整组预测 | 一个question加五条轨迹的前缀，输出一个组分数 | 直接学习这组回答的正确性多样性 |
| 方差监督 | 用五条真实strict EM的总体方差作为目标 | 让全对和全错都对应零，Mixed对应正值 |
| 联合输入 | question只出现一次，五个Prefix80依次打包 | 模型可利用各前缀之间的信息，减少重复question的prefill |
| 中间层表征 | 冻结actor前15个block，另训练索引15的Qwen block及聚合头 | 在固定官方7B表征上训练组级判别能力 |
| 全量自然采样 | 169028题各有10条真实轨迹，每epoch每题只出现一次并随机选5条 | 增加实际G5组合覆盖，保留自然的困难负例 |
| 完整状态与独立复现 | raw训练、EMA跟随，保存Adam和RNG，运行两个独立seed | 可以复核训练状态，并观察随机性下的结果差异 |
| 按选题目标选checkpoint | 只按seed1开发集Mixed@512选epoch及raw或EMA | checkpoint选择直接对应最终使用方式 |

本次结果验证的是上述完整配置相对原E1的整体收益。由于输入层、前缀、结构、采样和训练预算等因素同时不同，不能把全部提升单独归因于某一种损失、某一层或Prefix80。

# 输入和探针结构

每道问题形成一个组输入，顺序为：

```text
question prompt
→ trajectory separator + trajectory 1的前80个policy token
→ trajectory separator + trajectory 2的前80个policy token
→ …
→ trajectory separator + trajectory 5的前80个policy token
```

分隔符固定为换行加“### trajectory”再换行。不足80个token的轨迹补151643，并将补齐位置的attention mask设为false；有效token的位置连续计数。question保留一次，question、prefix、separator分别有token type 0、1、2。当前head的位置容量为2048个有效位置。EM、标准答案和真实Mixed标签不进入模型可见输入；只使用实际生成轨迹的截断前缀。若生成回答本身很短，其完整内容可能自然位于前80个token中。

**层编号以代码为准：L15表示零起始索引15，即第16个Qwen block。** 冻结actor保留索引0～14，其输出是第15个block的输出；随后送入从官方模型索引15复制并训练的Qwen block。

聚合头的实际结构为：

```text
Qwen hidden size 3584
→ Linear 3584→256
→ 加入3类token type embedding及2048位置embedding
→ 加入learned CLS
→ 两层小型TransformerEncoder
   每层8个attention head、FFN 1024、GELU、dropout 0、norm_first
→ CLS读出与LayerNorm
→ Linear 256→1
```

可训练部分包含一个Qwen block和整个聚合头，约2.36亿参数；其余actor冻结。聚合头能双向整合有效位置，padding不参与注意力。训练时在远端现场prefill并detach冻结骨干的表征，没有为这份最终实验持久缓存hidden tensor。

# 预测目标和损失

设第i题实际抽出的五条轨迹strict EM为y_i1到y_i5，每条取0或1：

```text
c_i = sum_j y_ij
mu_i = c_i / 5
v_i = (1/5) * sum_j (y_ij - mu_i)^2 = mu_i * (1 - mu_i)
Mixed_i = 1 if 0 < c_i < 5 else 0
s_i = sigmoid(probe_logit_i)
v_hat_i = s_i / 4
loss = mean_i (s_i - 4*v_i)^2
```

| 五条中正确条数 | 真实组别 | 总体方差v | 实际MSE目标4v |
|---|---|---:|---:|
| 0或5 | 全错或全对 | 0 | 0 |
| 1或4 | Mixed | 0.16 | 0.64 |
| 2或3 | Mixed | 0.24 | 0.96 |

这份最终方案只有上述整组方差MSE，没有单轨迹BCE、直接Mixed BCE、排序辅助损失或标准化相关损失。它也不再计算五个单轨迹概率的“1减概率乘积减反概率乘积”作为组分数。

筛选时按s或等价的预测方差v_hat降序取Top512，不需要分类阈值。s是归一化方差分数，**不能解释成Mixed发生概率**。报告BA时另使用固定方差阈值0.08，即s阈值0.32；这个阈值没有用最终池拟合。

# 数据和训练配置

| 项目 | 已运行配置 |
|---|---|
| 官方冻结actor | Qwen2.5 7B Instruct，revision a09a35458c702b33eeacc393d103063234e8bc28 |
| 数据规模 | 169028道训练题，每题10条真实轨迹，共1690280条来源轨迹 |
| 每epoch采样 | 每题一次，重新洗牌；从该题自己的10条中无放回随机选5条，轨迹顺序也随机 |
| 标签来源 | 本次抽出的五条轨迹的真实strict EM |
| 采样分布 | 与EM无关的自然随机G5，不强制Mixed与非Mixed各一半，不使用类别权重 |
| 每epoch访问量 | 169028组、845140条轨迹访问；轨迹来自已验收G10池，不是每epoch再生成一整套 |
| 训练预算 | 10个完整epoch，每epoch1321次更新，共13210次Adam及EMA更新 |
| 并行与batch | 每次实验2张H100，global batch 128题；每卡64题、micro 8、累积8次 |
| 尾batch | 每卡34题，共68题，按实际样本数归一化 |
| 优化器 | torch.optim.Adam，lr 5e-6；默认betas为0.9和0.999、eps 1e-8、weight decay 0 |
| 学习率 | 常数5e-6，没有scheduler |
| 梯度裁剪 | norm 1 |
| EMA | decay 0.9995；raw接受梯度，EMA逐次跟随 |
| 初始化 | 官方对应Qwen block加随机新head，从头运行这份探针实验 |
| 实际耗时 | 每个seed完成10epoch约3小时18分，包含保存与逐轮评测 |

自然随机G5保留了“这道题G10有对有错，但本次抽出的G5恰好全对或全错”的困难负例。模型训练的是当前五条的多样性，而不是这道题历史上是否曾经出现过Mixed。它解决的是数据构造中的一项具体偏置，实际泛化结果仍以上面的独立评测为准。

# 两个seed和checkpoint选择

seed1的初始化随机种子为9137、采样计划种子为2026102227；seed2分别为19137和2026102302。官方actor、结构、超参数和训练预算相同，两次实验的head初始化及采样随机性不同，各自维护raw、EMA、Adam和RNG。

两个实验均完成10epoch。事先固定的选择规则是：只用seed1开发集，在第1～10epoch的raw和EMA共20个候选中选Mixed@512最高者；相同则优先较早epoch，再优先EMA。实际选中 **E9 raw**。seed1开发集为329/512，seed2直接使用同一E9 raw配置，为321/512；seed2没有另挑最佳epoch。最终使用raw权重，EMA 0.9995是训练中保留的跟踪状态。

开发集已被反复用于比较，因此最终确认另用了两个未参与选型的2048题池；模型、epoch、状态和主指标先冻结，之后才读取最终结果。如今这两个最终池也已经使用，后续改变方案需要新的独立确认数据。

# 使用边界

**顺序敏感性仍然存在。** 在同一开发集上，原顺序、逆序和循环移位顺序选出的Mixed题数，seed1分别为329、325、312，seed2为321、318、314。整组packed输入不是排列不变结构；本次固定并报告原顺序，没有根据最终结果选择有利顺序。

**分类错误仍需区分。** 两个最终池中新模型的Mixed BA约70.59%～72.90%，全对误报为Mixed的比例由基线约23%～26%降至约3.55%～4.63%；全错误报则由约13%升至约23%～28%。Top512排序已明显改善，并不表示所有分类错误都同时下降。

**当前证据属于官方初始7B actor的离线warmup。** 在线RL改变actor后，前缀和特征分布会变化，后续Mixed@512及下游问答EM需要重新观察。Prefix80和G5也是当前结果的组成条件，不能直接把这组数字搬到Prefix5、不同轨迹数或新actor快照。

checkpoint已通过SHA、raw与EMA、Adam、两rank RNG和采样计划的CPU加载验收，所选模型也已完成实际GPU推理。下一次CUDA Adam更新的完整resume一致性尚未实测，后续正式续训应明确这一步。

此前的2018窗口是另一份匹配对照：1954个历史detach分数加64个当前梯度样本。**本次通过最终确认的全量MSE方案没有启用缓存排序窗口**，不能把上述独立收益写成窗口机制的收益。

# 复现资产与来源

训练源码固定为commit `7a8ff6ae8a33324846d85f77ffc28e9ab699c028`；最终运行入口、评测与验收结果归档于commit `d466f60f65d1e2fceff61ae28cd0bd5302551c30`。模型与完整轨迹保留在HPC3共享存储，文档不包含凭据。

**seed1 E9 raw所在完整恢复文件**

```text
/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3/checkpoint-epoch-09.pt
SHA256 5f90fb17bee993353f47a7d687bdf2ea235dcd3d2eba287465567f4c286b13d7
```

**seed2 E9 raw所在完整恢复文件**

```text
/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3/checkpoint-epoch-09.pt
SHA256 4203a480330b5f64def9dcf641ad6e66067c1810a46c113b778b6c79cdc90145
```

完整恢复文件中同时包含raw、EMA、Adam、RNG和采样阶段；评测本次使用其中raw状态。

- [最终独立评测及错误分解](https://github.com/shaoshitong/Search-R1/blob/d466f60f65d1e2fceff61ae28cd0bd5302551c30/reproduction/reports/qwen7b-algorithm-improvement-20261021/fullP80-selected-root-control/FINAL-CONFIRMATION.md)
- [冻结配置与两份checkpoint绑定](https://github.com/shaoshitong/Search-R1/blob/d466f60f65d1e2fceff61ae28cd0bd5302551c30/reproduction/reports/qwen7b-algorithm-improvement-20261021/fullP80-selected-root-control/actual-candidate-freeze-raw.json)
- [最终原始验收回执](https://github.com/shaoshitong/Search-R1/blob/d466f60f65d1e2fceff61ae28cd0bd5302551c30/reproduction/reports/qwen7b-algorithm-improvement-20261021/fullP80-selected-root-control/actual-final-complete-raw.json)
- [逐epoch训练与开发集结果](https://github.com/shaoshitong/Search-R1/blob/d466f60f65d1e2fceff61ae28cd0bd5302551c30/reproduction/reports/qwen7b-algorithm-improvement-20261021/full169028-L15P80-MSE-ownG5/LATEST-COMPLETED-EPOCHS.md)
- [完整目标证据对应表](https://github.com/shaoshitong/Search-R1/blob/d466f60f65d1e2fceff61ae28cd0bd5302551c30/reproduction/reports/qwen7b-algorithm-improvement-20261021/GOAL-EVIDENCE-MAP.md)
- [固定训练源码](https://github.com/shaoshitong/Search-R1/blob/7a8ff6ae8a33324846d85f77ffc28e9ab699c028/reproduction/operations/qwen7b_warmup/online_packed_full169028_MSE_P80.py)
- [固定probe和聚合头源码](https://github.com/shaoshitong/Search-R1/blob/7a8ff6ae8a33324846d85f77ffc28e9ab699c028/reproduction/operations/qwen7b_warmup/packed_variance_probe.py)

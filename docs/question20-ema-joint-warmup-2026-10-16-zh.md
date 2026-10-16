# Question + policy20 联合监督探针预热（2026-10-16）

## 最新授权与执行边界

用户于本会话明确要求合并 A800 scale20 与 HF epoch68 两份数据，按真实采样事件去重；后续明确将原“完整训练一遍”改为 **200 个完整数据 epoch**，采用 question + 同轨迹前20个生成 policy token，并在测试时使用衰减0.99的 EMA 探针。参考会话为 `01a0fcbb-863f-7450-926d-db537f648265`，其当前科学实现位于 `reproduction/prefix20_group_joint100`，冻结 runtime 为 `2fd5c19fa0d9c5d6c4eb514a7e9a308eb6bc6479`。

本实验是独立探针预热，不恢复历史 actor/PPO 200 outer，不改参考会话的运行源码或进程，不继承已训练的探针、Adam 或 RNG。

## 数据前置条件

- 参考 epoch68：348160 个题组事件、1740800 条 G5 轨迹；同一道题在不同采样事件中可以重复出现。
- 已完成 A800 scale20：102400 道唯一题、512000 条轨迹。旧5120题的25600条事件与参考缓存重叠时仅保留一份；预计整合后445440组、2227200条，须以事件身份清单实核为准。
- A800 新增97280题、486400条记录仅保存到第5个 policy token；完整 rollout 在原封存验收后已删除。现有 prefix5 和其 layer27 特征不能充当 prefix20。
- 旧5120题与固定512题测试集有完整真实 raw 指针，可提取前20个 token。缺失部分若重建，必须在独立 namespace 完整生成 G5 并重新计算对应 strict EM；新输入与新标签绑定同一采样事件。禁止用旧事件的 EM 标注新生成的15个 token。
- 缺失数据的处理选择待本会话用户答复。答复前不启动有此依赖的生成或训练。

每条训练记录保存实际源事件、题ID/hash、响应序号、自己的0/1 EM、原 token IDs/位置/turn segment 与文件 SHA。去重键绑定源采样事件，不能按题ID删除同题新G5；组内恰好五条，不能合成G10。固定512题测试按题ID与规范化文本排除训练数据，测试回答不重新采样。短于20 token的真实终止记录按原协议保留，padding不作为有效token，禁止编造第6–20个token。

## 模型与损失

官方 `Qwen/Qwen2.5-3B-Instruct`，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`；冻结完整官方编码器，提取 index27 post-block 的 detach BF16 表示。输入是实际初始 question request 与本轨迹前20个 policy token，编码使用真实上下文、positions 与 turn boundaries，不读未来token。探针复制官方 block35 全部参数，保留原生16个query head、2个KV head、RoPE、RMSNorm、SwiGLU与残差；question槽右补零、policy20尾槽，mask清零后展平，接新 Linear 输出单正确性logit。实际全数据 Qmax 必须扫描确定，不截断。

每题五条响应的 logit 为 `z[i,j]`，其自身 strict EM 为 `c[i,j]`。定义 `m[i] = 1(0 < sum_j c[i,j] < 5)`，包括只有一条正确的情况。损失为：

```
p = sigmoid(z)
h = 1 - product_j(p) - product_j(1-p)
L = mean_trajectory(BCEWithLogits(z, c)) + mean_group(BCE(h, m))
```

两项权重均为1。实现枚举32个Bernoulli模式，在log域分别求mixed与extreme事件的 `logsumexp`，二者之差作为 mixed logit 后使用 BCEWithLogits；与上述概率公式等价，不clamp概率，不detach group项，不增加标准差、六分类、独立mixed head或类别重权。

## 200 epoch 与 EMA

每个 epoch 完整无放回遍历去重后的全部训练G5事件，epoch间重新洗牌。不能把一个5120题source批次叫作一次完整合并数据epoch。各G5在同一个rank、同一个真实Adam步骤内，micro只拆完整G5；尾批保留、不复制、不丢弃，DDP按真实组数归约。原始参考的8卡6/7组配置不能直接套到A800四卡。

若最终确为445440组，可沿参考每100步骤80批51组、20批52组的平均global256轨迹计划：每完整epoch8700次Adam，200epoch共1740000次Adam（四rank副本只计一次）。这是完整合并数据的训练预算，远多于参考每epoch仅一批5120题的100次Adam；不能用参考单批耗时估算整个合并epoch。默认复用参考fresh Adam、weight decay0与余弦LR3e-5到3e-6，但余弦时钟按本实验真实总更新数推进，无旧100epoch硬编码。最终运行配置须绑定实际事件总数和对应计划SHA。

EMA在fresh初始化与DDP同步后精确复制探针初始状态，参数影子使用FP32。每次真实成功Adam之后执行：

```
ema = 0.99 * ema + 0.01 * raw_probe
```

仅探针做EMA，冻结编码器和Adam不做EMA。EMA更新计数必须等于成功Adam计数；跳过步骤不推进EMA。训练反向使用raw探针。每epoch结束，用同一EMA版本评估完整训练与固定测试，主报同事件G5混合balanced accuracy，raw在线loss另列，不冒充最终同权重训练指标。测试使用eval/no_grad，不能将测试标签用于训练、采样、early stopping或调参。

主指标以 `h >= 0.5` 判混合；真实mixed仍为 `0<K<5`，balanced accuracy 为两类正确率的算术平均。同时记录普通混合accuracy、混淆计数、逐轨迹正确率/BCE、group BCE/total loss、实际数据曝光数、成功Adam/EMA次数、epoch用时。EMA不保证不过拟合，效果以固定测试集的真实曲线报告。

## 保存与执行记录

模型、输入、特征、checkpoint、optimizer、完整预测全部留远端；Mac仅保存代码、配置、文档和小审计。checkpoint包括raw与EMA探针、EMA decay/count、Adam、各rank RNG、数据事件清单SHA、epoch/step及实际数据cursor。仅从同一新实验的完整合格状态恢复，缺失EMA不得静默换raw冒充。

运行使用独立目录和一次性launch身份，现场选择空闲授权GPU，不影响其他任务。发布精确源码/配置后核验远端commit和blob，再部署并核验实际文件。至少确认两次真实更新推进和首个完整epoch；实际200epoch、最终train/test指标、完整checkpoint与only-owned资源收尾完成后才能宣称完成。本文是科学合同，尚不构成运行已启动的证明；数据补齐、实际GPU绑定、LR/批次配置与启动回执须在部署配置中明确记录。

## 本次已完成的源码验证

2026-10-16，隔离核心包 `reproduction/question20_ema_warmup` 已完成小型CPU算术与接口验证：联合损失/梯度与直接概率式一致、极端logit有限、EMA初始化与更新、raw/EMA/Adam/RNG保存恢复、两rank不同完整G5 micro及尾批空rank的归约与直接global mean一致。数据模块的合成fixture验证了同题不同事件保留、同事件精确去重、prefix5拒绝、固定测试题隔离及多turn实际positions组装。

以上均为小型CPU验证，不是官方LLM/CUDA前向、真实合并数据资格或200epoch运行验收。核心包当前提供模型、loss、EMA、完整G5更新、指标、checkpoint、合并scan读取与冻结编码器接口；数据生成/搬运、真实manifest、预编码缓存及远端一次性执行控制尚须随数据选择完成，不能将这些接口冒称已启动的完整训练入口。

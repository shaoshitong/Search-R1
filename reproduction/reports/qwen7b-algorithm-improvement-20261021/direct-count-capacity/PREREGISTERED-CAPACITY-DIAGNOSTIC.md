# 冻结E1的count head拟合能力诊断

在实际GPU拟合之前固定本诊断：从生产冻结前256批的8192个训练题按qid-only hash(seed9137)取512 fit题与随后1024独立internal holdout题，禁止既有diag/selection/final加入。所有问题属于原warmup已见训练池，不能称为原warmup未见问题。

从生产原始E1 raw block27与旧trajectory head出发，所有legacy参数冻结、不建立legacy Adam。通过原immutable501 `native_hidden_forward` 的只读encode适配器取得真实masked P20 hidden，保留原BF16 dtype，不重生成rollout。相同ID/mask/position/shape下重复原API，要求cached hidden/logits/embedding完全一致；legacy raw和native buffers的完整hash前后不变。

三个相同seed1729初始count head与EMA、fresh Adam0和相同RNG/epoch顺序，仅head LR分别5e-6、1e-4、1e-3；每支20 epoch，512/32=16次更新/epoch，共320次真实Adam更新。fit保留原4个G5视图，每题总权重1，各视图1/4；不是2048个独立问题。旧trajectory BCE是冻结常数，仅记录不反传。head.encode采用CUDA BF16 autocast并返回float embedding；count classifier和NLL在autocast外FP32，与生产分界相同。

每个问题从原4个G5中，按source_indices字典序预冻结唯一primary G5，不使用EM选择。fit-primary和internal-holdout-primary各自报告真实count NLL、六类accuracy、Mixed BA/双召回/AUC；holdout的Top256/1024始终按唯一primary view score排序，标签只由这5条自身EM确定。原4views额外只作fit NLL诊断。raw和EMA(.99)分别报告0/1/2/5/10/20 epoch，禁止把两个模型的loss与holdout指标混为同一个模型。

这是有限拟合能力诊断，不据内部结果选生产胜者或打开外部/final评测。单次normal/acd_u、1 H100、8 CPU、64G、30分钟、无依赖、actor更新0。原743619及其后续状态不改变。

真实准备：typed config5452B SHA `ab009f9facc47bc2115cbfc7ee8889609075e62e8a18e0809cf27e512d44b17f`；split48155B SHA `e4ea6cc69ebc4f16bccf19a0c4587bf6f96a5d289be30645e240b36a25803e0d`；共用20epoch顺序316845B SHA `bdfe5760992dd137c7d2eb56fbb626cce53e45607ecd2ad7fcd2b64d6d29c37e`。运行代码已发布 `4e2c7fe5`，metadata-only split/fixed-G5 checks实际通过；实际同ID缓存与首更新恢复资格尚待GPU执行，不能提前称通过。

每支首真实完整update做raw/EMA/Adam/RNG重放；每epoch保存完整head状态、原4rank历史RNG档案、数据游标和源绑定，经完整文件SHA与CPU reload后发布latest；先保存再评测。最终验收全部三支320步、全部指标节点、所有中间full checkpoint和冻结legacy hash，并至少复核两次实际进展。模型、hidden缓存、checkpoint与逐题输出只留HPC3；本地仅代码、此配置与小aggregate。

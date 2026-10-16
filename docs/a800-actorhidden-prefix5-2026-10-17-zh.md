# A800 四卡、question＋prefix5、当前 actor hidden 正式训练

日期：2026-10-17（Asia/Shanghai）。参考会话 `01a0fcbb-863f-7450-926d-db537f648265` 的实际运行源码为 `700a0b3d991853573d5d50d4eae6239a356a9f87`。参考旧3420已停止；当前参考3425已真实完成7/200轮，下一轮正在推进。本分支从该实际源码隔离适配，不修改参考任务。

## 人工授权与权重

用户要求四卡、完整question＋前5个policy token，并对齐参考正式训练。原指定本方warmup第3轮，但它只保存为可替换的`latest.pt`，已被第4轮覆盖，实际库存没有独立第3轮副本。用户随后明确同意：**使用切换时实际保留的完整轮EMA权重**。

CPU prepare读取旧run `8af926a49630428097b94027864da68b`、execution `bcfa1de0a90a4fcda70263a75153fcee` 的真实完整epoch元数据，先hardlink冻结对应inode，再验完整SHA、CPU状态、14个模型张量及原生RoPE buffer。只导入EMA模型；不导入warmup Adam、RNG、学习率或数据游标。源epoch、哈希和实际初始化回执决定最终身份，不能使用历史哈希冒充仍存在的权重。

Actor仍从官方Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` fresh初始化，正式总预算200 outer。当前probe warmup只有在新版已发布、远端源码逐文件验SHA、CPU prepare完成后才按精确rank0 birth优雅停止；原完整/partial状态全部保留。新actor训练不是旧probe训练的full-state恢复。

## 每轮训练流程

1. 从共同canonical169028题按新游标取5120个唯一候选，固定512诊断身份按qid和规范化文本排除。
2. 每题G5，当前actor原生vLLM layer27 postblock捕获全部question token及前5个policy token表示。实际采样第6个用于处理第5个token；第6个及其LP不进入探针、PPO或续写状态。无独立encoder前向。
3. 官方完整Qwen block35＋masked flatten＋单logit Linear，物理布局273＋5＝278。预测每条轨迹正确率p，以`1-prod(p)-prod(1-p)`按题选Top512，随机0，同分按原candidate ordinal，选题不读最终EM。
4. 普通轮只续写Top512对应的原G5，保留真实token、old-policy LP与工具上下文。使用原G1 grounding软奖励更新actor20次Adam。正确性探针不充当reward。
5. 每20轮（20/40/…/200）先固定Top512，再把当轮全部未筛选5120 G5续写一次，投影选中组给actor，完整25600轨迹用于探针自然一遍训练。探针100次同步Adam，固定LR1e-6，逐轨迹自身EM BCE＋同事件G5混合BCE各权重1；混合含K1。其余轮探针0更新。
6. 保留参考既定周期诊断与5/50/100/150/200 QA保存点评测，最终51713题。没有额外GPU验收、模型概率扫描或新baseline。

## 四卡等价与A800执行适配

- Actor全局mini128、每outer512×5、PPO epoch1、20次更新不变；BF16前向、FP32 master/reduce/Adam、57完整outer warmup、LR1e-6、entropy/refKL0、DualClip3、clip.2、FP64全局L2clip1保持参考实际实现。
- 参考目标是八个local16轨迹的token均值再等权平均。四物理rank各处理两个virtual16子块，各按本块token数归约再乘1/2，避免变长轨迹权重随卡数改变。A800默认globalmicro16/local4/accum8。
- 探针仍全局平均256轨迹：80步51组、20步52组；四rank每步12/13完整G5，DDP scale为`4*local_groups/global_groups`，每周期每rank1280题。
- 四张训练卡排除辅助5/6；辅助5/6承载两shard FP16 Flat检索，grounding副本在6。使用本实验已验收的官方模型、共同manifest、原冻结G1 reward校准；这些资产SHA写入最终config。所有实际indices/UUID在prepare/launch绑定，忙卡拒绝，不停任何其他任务。
- A800直接owned进程入口替代H100 Slurm。父/子持有OFD UUID lease，进程清理按PID/birth/SID/UID，native Linux pidfd兼容缺Python封装的环境。不能重复once claim。
- 参考专用actor路径只拒绝真实nonfinite，未接旧A800 pre/post KL.2与negative-tail额外监测。本分支按用户要求对齐参考实际路径，不宣称这两条路径原本完全相同。
- 实际OOM重试记录已发生但未保存的partial更新及重载丢失，不能声称回滚或把失败attempt记完整outer。

## 执行与记录

本地控制入口：`reproduction/operations/a800_actorhidden_prefix5_control.py`，沿原control.py物理route，依次`stage → prepare → observe真实CPU exit0 → stop-warmup → observe真实owned全退出 → launch → observe实际推进`。`stage`参数必须是具体handle、已发布Git archive和源码清单；`prepare`不能被重复调用冒充恢复。

运行入口：`python -m reproduction.prefix_selection200.launch run --base /data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-<run>`。

固定512身份与候选池仅做必要CPU元数据准备，不读大型feature tensor、不重新编码或生成teacher。模型、状态、轨迹、检索缓存及最终输出全部留远端；Git只包含代码、配置、文档与小型scalar/hash。

必要纯CPU检查覆盖：四卡virtual8目标及梯度等价、完整20更新计划、native lookahead身份/未来token排除、100步完整G5计划、EMA/raw严格区别、缺权重failclosed、合法zero-EOS；不新增GPU测试。实际部署、CPU prepare、actor Adam和首完整outer的事实应追加执行回执，代码完成或launch成功本身不是训练验收。

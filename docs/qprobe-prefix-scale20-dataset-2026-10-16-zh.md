# 2026-10-16：扩大回答正确性探针数据集

用户停止当前独立预热，要求以更大批量尽快构建20×5120道题的数据。此次范围是数据构建，不启动正式actor训练，也不自动续训旧探针。

## 数据范围

- 102,400道唯一训练题，每题5条冻结官方teacher轨迹，共512,000条。
- 包含已验收原5120题/25,600条；只新增97,280题/486,400条，不重复生成合格旧数据。
- 固定随机512题/2560条测试轨迹保持隔离；同时排除题ID、题文本哈希和归一化题文本重叠。
- 原共同训练产物169,028题及manifest SHA256 `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24` 保持原字节与顺序。只从该产物选择更大子集。
- 官方 Qwen/Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`，冻结权重，保持原采样、检索、上下文规则和strict EM评分。actor与Head的optimizer更新预算均为0。

## 提速方式

保持512题一个生产chunk和每题G5，连续处理190个新增chunk；同一套四卡teacher、私有Ray与双卡检索服务持续复用。执行参数由原seq64/token8192/inflight256/agent256/retrieval16，提高到seq256/token32768/inflight1024/agent1024/retrieval32。这些值在推理引擎初始化前绑定，不改变温度、top-p、最大回答长度、问题数或评分公式。

每批完整生成并评分后，截取真实prompt至第5个policy token（含），保存原token IDs、实际position IDs、policy mask与最终二元正确性。未来token不进入预测前缀；不decode后重新tokenize。短于5个policy token保留真实长度，0个则明确失败。

新增完整rollout先临时落盘并计算SHA；紧凑前缀、标签和身份文件落盘、fsync并通过SHA验证后，显式 `compact-after-verified` 策略清理该新增临时完整rollout。旧数据不删除。模型、原始问题、轨迹和特征全部留在远端；本地仅代码、配置、文档和小型scalar/hash审计。

编码入口提供四个独立BF16冻结AutoModel实例分片提取layer27全有效token特征，默认micro128、按长度和token预算组批，原5120训练题与固定测试特征直接引用。每个worker独立输出，最终合并索引，不复制大型特征数组。编码是后续数据准备阶段；teacher生成完成不等于特征全部完成。

## 时间估算与实际验收

此前实际512题/2560轨迹生成用时219.049秒，因此新增190个chunk在原速度下约11.56小时。提高并发后的4–8小时为待验证预测，不能将batch倍数当作吞吐倍数；首批实际生产数据完成后更新ETA。新增特征编码的旧单卡速度推算约72分钟，四卡理想均分约18分钟，实际还受写入与全SHA读取速度影响。

启动后必须确认两次真实新增数据推进及首个完整chunk，记录实际题数、G5轨迹数、正确性标签、生成时间、吞吐、错误和真实进程身份。仅PID存活或启动回执不足以证明构建推进。最终必须核对102400唯一题、512000唯一轨迹、固定测试隔离、逐chunk身份与SHA、零actor/Head更新及本任务进程释放。

## 执行入口

`reproduction/operations/a800_qprobe_prefix_scale20.py` 提供 `--action stage/run-once/observe --handle <本地handle>`。stage读取最新人工意图、固定runtime fcb源码和已推送操作文件SHA，创建新独立远端目录；run-once只启动一次，未知SSH结果必须观察同一handle。训练卡使用真实空闲四张（排除辅助5/6），实际indices/UUID绑定后不更换，不使用或停止其他进程。当前GPU1和7的其他任务保留。

`qprobe_prefix_teacher_scale20.py` 是远端teacher构建入口，`qprobe_prefix_scale20_encode.py` 是后续特征编码入口。原runtime不热修改。实现、执行入口与必要配置必须先提交推送，并实际核验远端Git SHA/blob。

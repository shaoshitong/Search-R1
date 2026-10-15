# Balanced prefix 探针停训记录与扩大数据计划（2026-10-16）

本次按最新人工指令停止，真实完成 **6 个 epoch、691 次 Head Adam**：前6个完整epoch600次，第7个epoch保留91次部分更新；actor更新0。没有回滚，也没有完成原定20个epoch。原始记录保持 `PREFIX_WARMUP20_FAILED`、wrapper exit1、child exit−2；`KeyboardInterrupt` 来自人工 SIGINT，不改写为训练成功。

最后一个完整epoch为epoch6（600次HeadAdam）的全量指标；691次部分checkpoint没有新增评测，不能把epoch6指标当成其指标。

| epoch6指标 | 训练25,600条 | 固定测试2,560条 |
|---|---:|---:|
| 平衡准确率 | 0.810987773783 | 0.517826954144 |
| 原始二分类准确率 | 0.875664062500 | 0.767187500000 |
| BCE | 0.475501724774 | 1.231681481757 |
| prior校正准确率 | 0.871875000000 | 0.771093750000 |
| prior校正BCE | 0.512136752699 | 1.448583978203 |

训练与测试平衡准确率差为 29.316082 个百分点。用户提出数据量不足、过拟合的假设；本记录保留实测差距，不把扩大数据后的效果当成已验证。此前epoch4测试平衡准确率58.4280%是中间最佳，最后完整epoch6为51.7827%，不以最佳值代替最终记录。

部分状态保留于远端 `/data/shaoshitong/Search-G1-qprobe-prefix-correctness-balanced20-aa4a9ebd63ac4683985bf74a748dcfb2/warmup/partial-head.pt`，931033323 bytes，SHA256 `872aec5cc4012101ca600cc8fa9a82f4120d9b8b3735ecf60889c101f0e168c0`。partial-state实际Adam计数和optimizer步数均691。这里记录既有元数据，不冒称额外finite检查或CUDA重载已通过。实际owned SID存活列表为空，当前作业已释放；当次GPU0/2/3/4/5/6空闲，GPU1的他人PID447952占用630MiB、GPU7他人作业未动，不能把此时空闲当成后续资源预留。

新授权仅构建 **102,400道唯一训练题×5＝512,000条轨迹**：复用原5,120题×5＝25,600条，新增97,280题×5＝486,400条，排除固定测试512题。官方模型、共同训练集合、评分/检索/采样语义保持，新probe或actor训练尚未授权、尚未启动。

原实测每512题G5采样219.05秒；新增数据相当于190个这样的chunk，保持原吞吐约 **11.56小时**。增大batch和并发后的 **4–8小时是尚未验证的预测**，须由新作业实际吞吐修正。旧正式run509仍手停、正式goal与旧prefix监控PAUSED；本记录没有恢复训练或推进小时anchor。

证据：`preparation/qprobe-prefix-scale20-oldfit-release-inventory-20261016.json`，SHA256 `aabd798cec880992b9a9e32bd7d9bf4823a9983fc93146170cbff2c067e11ff8`。小型JSON同名提供精确scalar/hash；无QA、tensor、模型或rollout全文。本报告尚未由交接agent提交或推送。

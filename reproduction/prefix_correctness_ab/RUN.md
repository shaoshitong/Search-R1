# H1008：10-token / 20-token，20-epoch warmup

2026-10-16最新人工授权：真实运行两组离线逐轨迹EM正确率探针。GPU0–3为question+10实际token，4–7为question+20实际token。旧3402和正式actor/PPO保持停止。

用户已明确沿用参考线程最新均衡采样：每组globalbatch256=128正确+128错误，4DDP各64，无梯度累积。每epoch100个有放回均衡batch，20epoch=每rank2000次真实同步Adam；这是25600 draws/epoch，不称独立样本恰好遍历。两组同seed、全局batch索引、官方冻结encoder index27输出、一个full-copy官方末block35、相同flatten长度、唯一BCE、LR3e-4。变化只有可见原始前缀10/20长度。

复用参考线程原A800只读teacher5120×5与新随机独立512×5轨迹，CPU仅提取首轮actual_request_token_ids和raw_generated_token_ids前20（EOS提前终止保留），再将派生短输入从A800经管理机直接送到H100，绝不经Mac、绝不重新生成teacher/测试回答。使用A800同一train/test协议，避免H100旧teacher cacheutil .1与参考 .25的身份差异被静默混合。

特征抽取四GPU各负责stride分片，冻结encoder短forward捕获index27后停止，不算后续8block或完整轨迹。原始ID输入无future检索/答案，positions由已去padding的真实请求规则重建并明确标注。只扫描一次所有train/test的20-token长度，10/20共用Lmax保证head容量一样。模型/特征/完整预测及每epoch训练状态全部留远端。

每组epoch0和每个epoch后评估完整固定512题×G5=2560条。测试不参与采样或拟合，主交付固定epoch20，不挑测试最优点。报告原始balanced-prior logits和基于自然训练正例比例作logit shift的两套accuracy/BCE/Brier/balanced accuracy、precision/recall/F1及概率饱和；校正依赖label-conditional sampling假设，不保证概率已校准。训练日志报告均衡抽样实际draws上的loss/accuracy，不能冒称每轮全自然训练集重新测过。

两组独立torchrun4rank、master端口、输出、Adam/RNG，同一个本机Slurm作业内并行；全8UUID OFD以及自有birth/pidfd/cgroup cleanup沿用。只做正确输入/真实Adam/有限loss-grad/正常保存的必要检查，不重跑baseline或添加数值风险门禁。先观察真实提取字节增长、随后至少完整epoch正常结束，最后核20×100计数、测试qa/trajectory count、actualexit0与资源释放。

参考代码来自本地Search-G1/repo的prefix-correctness features/fit/operator；原参考sampling_mask截取可跨后续turn，新实现明确仅首轮raw前缀，不原样复制该行为。旧A800 prefix5结果须标注表示截取差异，10/20两组内部严格同实现可比较。

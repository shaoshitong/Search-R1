# 同缓存binary / three-event对照：仅准备

新独立入口 `coarse_event_cached_train.py --mode binary|three_event`。目标目录分别为 `algorithm-improvement-20261021/coarse-event-frozenE1-{mode}-fit30016-hold5120-v1`。本轮没有submit动作、GPU分配或新native forward，不改正在运行的805 Count源和原12节点screen。

输入必须是family resolver的真实 `FROZEN_E1_SHARED_CACHE_CP0_COARSE_FORK_INPUTS`。至少包含baseline_config、baseline_publication、cache_complete、baseline_CP0、baseline_CP0_four_hashes、baseline_epoch0_evaluation、family_protocol与frozen_train_prior的真实描述符。cache、CP0或epoch0未出现时返回 `WAITING_FOR_VERIFIED_BASELINE_CACHE_CP0`；不写假SHA或已通过声明。CPU准备仅SHA绑定metadata与small ZIP范围，50GB隐藏文件只stat/继承producer SHA；实际fork allocation必须再次fresh流式SHA并完成CP0四state重载，随后才能第一次更新。

源为805123aebf8af83a7c1c44947f39c294f7aeb624 Count baseline、config `fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767`。只读取其真正保存的小CP0，ZIP上限64MiB且严格只含7个新head参数、Adam、RNG和历史4rank digest/RNG archive；没有旧3.75GB E1载入路径。使用d.restore_new_head/u.restore_rng后，raw/EMA/Adam0/RNG四hash必须与baseline CP0完全相同，不能仅以seed相同代替。

保持同30016题、同938批次顺序重复20epoch、18760步、4个真实G5 view/题、每view0.25、LR1e-4、weight_decay0、EMA.99。encode为BF16 autocast，单batch320真实轨迹；classifier和事件loss在FP32 autocast外。不增加clip、class resample或classweight。冻结trajectory BCE只作常数记录，和Count head baseline一样不回传旧E1。

两模式唯一训练差异为已资格的 `CE(event_logits+log(π_train/τ),target)`。π只使用冻结full938真实训练view计数，数值不导出。推理q使用unadjusted eventsoftmax的Mixed项，保留raw6 latent probability与冻结p_j；q明确为τ参考balanced decision score，非原分布已校准Mixed概率。粗loss不识别K1..4内部，binary也不识别K0/K5内部；不报告6-way准确率/countNLL作为目标正确性证据。

每节点epoch0/1/2/4/8/20先完整保存head4state，再分别评分raw/EMA的fit_all4、固定fit_primaryG5和同5120题预冻结new5。输出qid/q/K/domain/真实trajectory_ids/strict_em、unadjusted_six_count_probabilities、unadjusted_event_probabilities、p_j、coarse_event_NLL与score_semantics。粗事件loss、Mixed BA/两召回/全错全对FPR/AUC/Top为主要指标；不打开外部/final，不在本入口选多个external候选，family独立协议最多选一个。

首次真实更新双重放后保存CP1，独立CPU子进程只重载小CP0/CP1四state并核source archive，未PASS就不得更新2。连续1..6保存且第7继续由实际operator后续验收。每128步及每epoch保存，USR1完成当前事务后保存最高cursor与未完成评估阶段；恢复强校config/cache/CP0/π/顺序和实际progress/intent，禁止从较老checkpoint静默回滚。尚缺历史节点而cursor已经越过时明确拒绝，不将未来权重冒充历史节点。

normal/acd_u单H100、8CPU、128G、2h的entry仅可在真实输入binding与后续明确GO后使用；不设置人工Slurm dependency。本准备没有运行GPU或改变当前Count训练。必要CPU资格使用合成小head CP0文件验证真正载入、四hash相同、next-update重放/持久化恢复一致、Adam及data cursor/source/mode损坏拒绝，不能把toy资格称真实cache/CP0已验收。

# 三种全量方差目标：真实启动回执

169028训练问题×10条真实轨迹，随机M=3..10，单question＋M条Prefix20，一条packed prefill。共享已验收只读fixture，三个独立输出/锁/Adam/EMA；官方初始block21/head，种子与每epoch数据计划相同。每个epoch每题一次，10epoch、13210更新，global128，尾68；每组两H100，LR5e-6，EMA0.9995。

- MSE: job749568，ACD1-51，两rank真实406更新。
- MAE: job749576，ACD1-14，两rank真实124更新。
- zMSE＋原值MSE: job749577，ACD1-16，两rank真实131更新。

上述实测epoch1792685089.8090599，均无当前failure，block/head梯度非零，至少两次实测与连续五更新。标准化组38→131更新，实际global128统计两rank完全相同；mean/总体STD可微，epsilon1e-6，不使用验证统计，anchor权重1。训练目标使用4v，指标按原v单位报告，筛选按预测方差降序。

标准化组第131步：zMSE1.5099430084＋anchor4vMSE0.1513153911＝1.6612584591；这是随机训练batch，不能当固定集泛化结果。每轮固定训练及独立2048题均评raw/EMA，最终4096未触碰。尚未完成10epoch或证明效果改善。

原MAE749569和Z749570在ACD1-13初始CUDA阶段失败，0更新/无checkpoint。集群禁止scontrolrequeue；保存原日志/失败回执后唯一替换作业并排除该节点，source7ac274ac8ff5dea0a12dfd44cfb03fe57491458f及训练入口不变，没有优化器回滚。

完整终态CPU验收器已stage，source799e75f35347055a281d290f8f22777ac97738f5；只在实际E10 complete及Slurm COMPLETED0:0后启动，不打开在写checkpoint。

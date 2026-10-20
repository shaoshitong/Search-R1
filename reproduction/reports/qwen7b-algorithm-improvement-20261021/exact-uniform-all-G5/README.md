# 实验实现：精确均匀平均全部真实G5组合

最新：父任务已明确GO并实际提交唯一作业743417；执行和验收进度见 [DEPLOYMENT.md](DEPLOYMENT.md)。下文记录79743d75实施准备时的设计与CPU资格，历史等待约束已被最新GO覆盖。

这是受控新分支，用于检查四个随机G5视图的Monte Carlo分组噪声。每题已有真实G10/G20分别枚举252/15504个不重复五轨迹子集，不额外生成轨迹，不改ownEM，也不强行制造Mixed。**当前只完成源码、CPU数学与toy恢复测试；未远端部署、未提交GPU、未评验证集。** 必须等待G20最终证据和父任务明确go。

## 精确目标与有界梯度

每原始问题所有子集均匀权重1/C(G,5)，总权重1。目标只含trajectory BCE＋three-event NLL；binaryMixed、within/inter ranking和moment/stat非零权重显式拒绝。均匀所有G5的trajectory BCE可解析简化为源G轨迹平均BCE，因为每条轨迹恰出现C(G−1,4)次，C(G−1,4)/(5C(G,5))=1/G。

three-event NLL逐组合chunk复用已有稳定 `group_event_scores`，真实K=0/1..4/5分别使用allwrong/Mixed/allcorrect事件。每个chunk立即对[Q,G] logits求梯度并丢弃图，累积准确value与dL/dlogits；最终自定义autograd只保存[Q,G]一阶VJP。**不支持高阶导数，不宣称数值累加bit-exact等同单体图。** 不会保留15504组计算图、创建每视图的Python身份dict或任何view×view矩阵；全部inter矩阵分配为0。

chunk64、Q48时最多同时3072个G5行，稳定事件原语内部最多98304个32-pattern行；仅保存480/960个logit梯度元素（FP32为1920/3840字节）。组合索引用generator逐块生成，不存储每个问题15504份元数据。

同一原始真实G10/G20的所有logits仍只走既有的两遍micro前向：第一遍逐micro训练前向后立即detach；第二遍恢复每micro原RNG/native buffer，确认logits逐位相同后做VJP，最后一次Adam及EMA更新。保留单个micro的transformer图、原256MiB native-snapshot边界。**相对当前random4没有新增native模型前向**；增加的只是有限组合事件loss计算。所有组合的目标等于随机均匀G5视图目标的精确期望，不等于某一次random4样本梯度，效果优劣待实测。

## 独立入口与旧API

- 新数学模块 `exact_uniform_g5_loss.py` 支持真实G10和G20，严格核对全局唯一轨迹ID与各自binaryEM。
- 新 `probe_exact_uniform_g5_train.py` 提供独立entry/config factory和两遍bounded update。
- 按父任务授权，旧 `probe_g10_train.run` 仅新增 trusted keyword `update_callback=run_update` 并替换run内3处调用，默认函数和现有调用行为完全不变；配置不能提供可执行函数。当前活动远端immutable代码没有被修改。
- 新entry显式传exact callback，保留原运行器的完整E1 raw/EMA/Adam10464、四rank RNG档案（新单GPU分支用canonical rank0）、首更新双重放、完整SHA checkpoint、恢复一致性及保存先于评估的gate。无需改变原生block、readout或Adam配置。
- 初始entry**只接受实际G10 matched6528、random reference、B48×136/epoch0**，原plan descriptor和136个实际批次顺序原样继承，LR5e−6、micro5、save16保持。G20数学与bounded update支持已测试，但G20 bank-loader/整runner部署尚未提供，不冒称可直接跑G20全流程。
- `prepare`从真实已SHA绑定reference config/plan复制资产和顺序，并从新不可变source-publication绑定完整源码。默认 `explicit_launch_authorized_after_G20_final=false`，`train`会拒绝启动。父任务明确go后才能在新的准备动作显式提供 `--explicit-go-after-G20-final`；本次没有调用。

`recipe-spec.json`是未部署规范，不是假冒真实远端完整config，也不包含虚构资产路径。实际source plan/ref必须由后续prepare从现有reference实读和验收。

## focused CPU验证

10项测试通过，包含独立double brute枚举G10的loss和梯度对照（不调用生产事件原语或解析trajectory简化）、G10/G20全部K的精确Mixed组合质量、置换与chunk不变、G20极端±1000 logits有限梯度、解析BCE等价、非支持权重/重复ID拒绝、toy两遍VJP的参数梯度/Adam/EMA对照、带Dropout/native缓冲和RNG的双重放、完整blob恢复后的下一更新一致，以及现有full checkpoint保存/加载回归。真实GPU下一更新及完整CP资格仍需未来获准后实测，CPU测试不能替代。

命令：

```bash
python -m unittest  reproduction.operations.qwen7b_warmup.test_exact_uniform_g5_loss  reproduction.operations.qwen7b_warmup.test_probe_exact_uniform_g5_train  reproduction.operations.qwen7b_warmup.test_probe_g10_train.ProbeG10TrainTests.test_accepted_fullstate_save_load_next_update_is_equal -q
```

## 有限算时／内存估计

本地Mac synthetic CPU1、Torch2.8、Q48、chunk64，仅loss＋gradient实测：G10 252组合×48约0.096秒；G20 15504组合×48约0.811秒。该进程报告的peak RSS增量约28.5MB/16.0MB（后者在前者之后测，基线peak已抬升），整个进程峰值224.2MB。不能把这个差值当GPU峰值，也不能把Mac CPU时间当HPC3 GPU端到端时间承诺。

native模型的真实前向数完全不增加；组合事件算术增加，GPU同步与设备算时尚未测试。已有GPU模型显存需求不变，只新增有界chunk临时loss张量；真实分配器峰值和每update速度应在获准后的首更新资格中记录。本次未为测速度申请GPU或占用已有任务。

# E1完整状态继续：joint低LR，然后head-only

唯一新namespace：`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-20261022-v1`。Root唯一提交；stage/prepare不提交GPU、不停止任何作业。

```sh
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_e1_lr_headonly.py --action stage
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_e1_lr_headonly.py --action prepare
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_e1_lr_headonly.py --action submit --submit-authorized
```

普通acd_u/normal，1H100、8CPU、128GiB、24h，无Slurm依赖；同名活动作业或已有submission-intent会阻止重复提交，未知结果先核对已有作业。

源为已固定的E1 cursor2577：`cursor-002577.pt`，3781073882B、SHA256 `c0a1c7dc3663d4216966a3440c2103348bcc52db3cd04c2f5c59a784a2792917`。原E1完整receipt/path保持不变；只复制19415B source-config元数据，SHA256 `e5f0027dfcff4b6db442b38e9b2452e86cc2c9adcabd97a194971dc4d8f8d837`。CPU不读取CP payload；新GPU先fresh全CP SHA及完整状态恢复。

从E1继续2642个全局B128更新：2577→3898为joint native/head LR均5e-7；3898→5219冻结raw native及其Adam，仅head LR5e-5。旧Adam矩、采样顺序/难度/位置、RNG、raw/EMA完整继承；EMA全模型继续.99，冻结raw native并不等于冻结EMA native。实际micro64，effective batch128不变。新phase checkpoint诚实记录native/head不同Adam时钟，不伪增冻结Adam步数。

源bank约277GB的已验seal继承，保留fresh metadata/逐row完整性校验，不重复全bank payload扫描。旧27运行文件逐文件沿用准确origin/hash；仅新增独立worker/updater/phased-checkpoint三文件，共30。原训练源码、四epoch产物和已结束746266不动。没有新增损失、阈值、数据或评分变体；用户请求/计划由Root另行发布e433ef7cdae6591d5bf883dd73e844ee85ba4c16。

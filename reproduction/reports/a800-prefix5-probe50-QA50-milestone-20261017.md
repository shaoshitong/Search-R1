# A800 Prefix5/P10 第50轮里程碑 — 2026-10-17 23:56:30（上海）

**QA50 strict EM226/512=44.140625%，F1=52.76351355%。** 与QA5保持相同512题身份、NQ/Hotpot各256和统一BF16交互契约；QA5 EM26.5625%/F136.06585167%，分别提高17.578125/16.69766188个百分点。分源NQ EM103/256=40.234375%、F151.78052850%；Hotpot EM123/256=48.046875%、F153.74649860%。这是actor最终回答QA指标，不是探针Mixed分类指标。

同一run `54ac33dcebe64e62a11fe4177f41f116` / execution `a4a0c2a18cca40bc8c3850d40eed8554` / runtime `75d536575593891876df339467da73b3c4cbdc7c`，仅一次有界metadata观察。完整50/200、actor累计1000次实际Adam、scheduler50；50轮20机会/20返回/0skip，四rank副本计一次。相对真houranchor49新增完整1轮/20实际Adam；当前51已完成5120题/25600 Prefix5→预测Top512→2560完整续写，生成Mixed174/512=33.984375%；actor51当前四rank0机会/0返回，20机会未确认、独立inflight未知，不能计完整51。pipeline和actor51实际存活，无记录中的新失败。

更新轮50原生候选2560G5=12800条全部完整生成，真实Mixed591/2560=23.0859375%；真实EM优先选择512/512 Mixed用于actor，因此selected100%不是探针准确率。reward均值0.7261367359。不混入普通轮预测Top512曲线。

探针50使用全部2560题/12800标签自然1 epoch、LR5e-6、global平均256条（实际255/260按完整G5），新增50次真实Adam，累计250，训练循环7.5626秒。前后对同一实际数据配对：

| 指标 | 训练更新前→后 | 测试更新前→后 |
|---|---:|---:|
|trajectory BCE|0.492521→0.467208|0.766239→0.809725|
|G5 mixed BCE|0.796267→0.661616|1.200252→1.129871|
|total loss|1.288789→1.128824|1.966491→1.939596|
|Mixed普通准确率|59.6484%→69.1406%|40.0391%→41.0156%|
|Mixed平衡准确率|55.8282%→65.4329%|53.2679%→50.3571%|

训练total loss下降12.4120%；测试total loss略降1.3677%，但trajectory BCE上升，Mixed平衡准确率下降2.910714pp，接近随机水平，不能宣称探针泛化改善。测试112Mixed/400nonmixed，全判非混合普通准确率基线78.125%。训练step曲线是不同自然batch在各自更新前的loss，50个标量点保留JSON，不与全量配对loss混为一谈。

每次自然更新的测试Mixed平衡准确率：

| 更新outer | 更新前 | 更新后 | 变化pp |
|---|---:|---:|---:|
|10|59.2365%|59.3344%|+0.0979|
|20|56.0379%|56.2878%|+0.2499|
|30|53.1369%|55.5110%|+2.3741|
|40|53.1724%|54.0164%|+0.8440|
|50|53.2679%|50.3571%|-2.9107|

固定512题身份保持，但每周期当前actor重新采样真实G5，标签/表示会变化；周期内前后配对，跨周期不是固定轨迹测试。在线raw head，EMA仅用于旧warmup模型初始化。

50轮joint保存元数据SHA `80c5b4cdbafed1c92441352cd725cc7b2819319da37f7672498f58524ec79f2c`；probe状态SHA `771f8c5c7b1a359cf786b5353a46261d23d2999567b3c8ba1bef3e0455c16184`、931830044B、250Adam/50逻辑绑定。此检查只读取现成metadata，没有新模型/tensor读取或独立canonical/finite扫描，不将metadata证明当最终200验收。

Grounding50在保存actor0050上真实refit作业exit0，round50版本、两个head SHA和adoption receipt SHA已经生成，四个round50服务实际存活。51的ACTUAL_RESTORE中`restored_grounding_probe_version`仍指父actor50的round0奖励来源，这是恢复 provenance；本地现行源码在写ACTUAL_RESTORE前已要求当轮round50并四rank状态一致，不是继续使用旧奖励的证据。51完整actor reward metadata尚未产生，后续正常检查核对，不额外插入测试。

GPU当次、窗口峰、本execution累计峰和采样空档未投影，均unknown。没有新评测/测试/review/子智能体/额外rollout/概率或tensor扫描；所有QA/探针评分来自原生既有流程。里程碑不推进houranchor49；源码不热改，训练同流程继续200，100/150刷新及最终51713、checkpoint完整finite/SHA、only-owned cleanup仍待实际验收。

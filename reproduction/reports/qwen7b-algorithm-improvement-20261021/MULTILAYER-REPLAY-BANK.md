# matched9088题多层真实causal replay bank工具

仅本地实现；未远端操作、未准备真实fixture、未提交GPU作业、未生成特征。原G10 runner、18c46aff replay模块、Q1 official view、原Native bank及活动源码均未改动。

新增代码：`replay_bank_prepare.py`、`replay_bank_capture.py`、`deploy_replay_bank.py`、`test_replay_bank.py`。

## 固定数据范围

从已经实际准备的 matched128/B51 plan 读取6528个不同fit qid，另保留原固定512诊断及selection2048。准备器验证三个池的qid、question_hash、trajectory ID互不重叠，保留每题原始同一次G5；不按标签、探针分数或prefix难度挑子集，不缩样本。

四片按每个role的既定顺序round-robin：每片fit1632＋diag128＋selection512＝2272题、11360条真实轨迹；合计9088题、45440条。以上是必须满足的规格及合成测试结果，**不是本次远端实际生成计数**。真正CPU prepare成功才会返回实测identity SHA、四份selection SHA、脚本SHA及prefix几何统计。

准备只扫描已有groups身份、sealed batch/input/prefix-only及packed index元数据，校验真实prefix SHA、qid/prompt/native locator和原采样协议一致；不读full response或own-EM文件、不读tensor、不加载模型。原bank JSON含标签，但题目/轨迹选择代码不访问该字段。每片保存独立prefix-only子集和SHA绑定selection。

## 统一回放后端及等价限制

CPU准备绑定实际已完成Q1的code/selection/model SHA回执，复用其official snapshot view。发布工具要求 `replay_probe_layers.py` 与已运行commit `18c46affb9c6c69330aa19eb1e00124e1f39d019` 完全相同。新的独立GPU入口复用该模块的normalize、replay_features、ReplayLayerWriter，不放宽或热改旧128条qualification入口；三层20/23/26在同一次实际causal segment前向捕获，对应人类层21/24/27。

Q2/downstream已由父任务实测：120条fixture完成；E1 q MAE约0.009/max0.0268、1/24阈值flip；E3 MAE约0.00646/max0.0271、0flip、Spearman约0.9983。这是父任务交接事实，本实施未重新远端查询；它说明HF/native并非bitexact。因此本bank只供**所有variants的fit及评估一起使用相同HF replay backend**的新实验，绝不宣称原Native无损替换、等价resume或生产可用。

每片normal/acd_u、1GPU/8CPU/64G/1h，没有dependency。64题分块封存，已完成块可SHA校验后续接；未知partial块保留并拒绝覆盖，需要检查后换隔离attempt。每片有独立锁、提交intent/job ID、实际progress/failure/complete。任何异常/未证实提交均禁止盲重投。

按父任务120条约13秒的capture样本，单片11360条的纯capture线性估计约1231秒；这**不是完工ETA**，未包含完整模型SHA、source/feature I/O及reader验收。若1h预算实际不足，可在新的准备目录显式 `--hours 2`，不删题或缩样本。

## 准备、状态及提交接口

发布后父任务统一运行；本实施未调用这些命令。已有小回执 `actual-launches/matched-loss-prepare.json` 绑定event config SHA为：
`749388a1525a59f23a129ab1c057e3b9b553d6b6af99666647444bb873fb8b67`。
默认路径为`.../algorithm-improvement-20261021/matched-loss128-v2/event/config.json`。执行时仍会实际校验该文件SHA。

```bash
python3 -m reproduction.operations.qwen7b_warmup.deploy_replay_bank \
 --control-root /actual/control-root --action prepare \
 --matched-config-sha256 749388a1525a59f23a129ab1c057e3b9b553d6b6af99666647444bb873fb8b67
```

默认新隔离根为`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-matched9088-bank-20261021`。`stage`只上传/校验源码，不启动CPU；`prepare`后台启动CPU metadata owner，不提交任何GPU任务；`status --prepared-commit <实际发布commit>`读实测状态。不要把CPU launch成功当成fixture准备成功。

CPU terminal成功状态必须为 `ACTUAL_REAL_9088Q_FOUR_SHARD_CPU_FIXTURES_PREPARED_NOT_CAPTURED`，其preparation/selection/script真实SHA通过后，父任务分别明确提交：

```text
deploy_replay_bank --action submit --prepared-commit <部署commit> \
  --control-root /actual/control-root --submit-authorized --shard 0
```

shard 0/1/2/3各自提交一次，无前作业依赖。helper检查同名queue、已有job ID、intent及capture目录，超时未知结果只能查状态而不能重投。既有作业不会被工具停止、改优先级或热改。

## 数据验收与reader

每个GPU job重新验证全部official model SHA及启动源码SHA，保持actor冻结，不生成新响应或标签。实际三层特征写在新目录，原native文件只作为历史来源元数据。

四片真实capture complete后，远端CPU可执行：

```text
runtime.sh -m reproduction.operations.qwen7b_warmup.replay_bank_prepare \
  --base <隔离根>/prepared --merge
```

merge重新通过`ReplayFeatureReader`验证全部shared replay文件SHA/geometry/trajectory coverage，发布`bank-complete.json`及`features-layer-20.json`、`features-layer-23.json`、`features-layer-26.json`。每个列表都是直接可供`ReplayFeatureReader(descriptors,capture_layer)`使用的SHA descriptors。`group-identities.json`只含身份与role，训练标签仍从原SHA绑定own-EM bank按轨迹IDjoin。

离线测试20项通过（新工具5项＋既有replay/reader15项），含9088完整规格的合成身份分片、重叠拒绝、G5封存prefix读取、脚本normal/无依赖/锁与唯一token、孤立代码包实际CPU import、旧alignment/reader测试。代码闭包11文件、约43KB gzip、11个4KiB chunks；所有**经过shell quoting后的**控制命令均小于16KB，最大约8789B。没有执行远端、HF下载、GPU或Slurm。

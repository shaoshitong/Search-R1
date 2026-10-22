# 7B探针历史实验完整记录

汇总时间：2026-10-22T16:29:40.883824+00:00。共 **363条 checkpoint×权重×评测池/判定规则记录**，其中336条有实际指标，27条明确未完成或未留存指标。这些条目包含同一模型的参考复评，不能当作独立实验数量。

BA指Mixed均衡准确率：0.5×(Mixed召回率＋nonMixed召回率)，不使用轨迹准确率替代。raw为实际训练权重，EMA为滑动平均权重；缺失记“—”。

旧独立2048池有641道Mixed；当前HF固定G5独立2048池有663道Mixed。内部holdout的问题可能被原始warmup见过，不是未知问题验证。不同G5、特征后端、阈值和数据池不可混排行。表内保留来源中的原始池名称。当前2048池已用于多次探索，不是最终无偏测试；最终4096仍未评分。

全量直接Mixed固定训练池A为2048题/1082 Mixed；新layer/prefix及方差pilot固定训练池B为2048题/888 Mixed。训练BA跨A/B不能直接比较。方差模型BA使用事先固定的v≥0.08诊断，其输出不是Mixed概率。

完整机器记录含精确小数、各项loss、召回率/误报率（原始报告存在时）、checkpoint引用、原始文件及JSON pointer。此表只读小型已有汇总；没有重新做GPU评分或读取最终4096。

[机器可读完整记录](ALL-HISTORICAL-RESULTS.json)

## 7B native L26→block27 Prefix5 trajectoryBCE+MixedBCE warmup

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R001 | epoch-0001.pt | EMA0.99 | 59.63% | 55.93% | 169/512 | 33.01% | 2048 | MixedBCE=0.891335 | — |
| R003 | epoch-0002.pt | EMA0.99 | 66.84% | 55.59% | 181/512 | 35.35% | 2048 | MixedBCE=0.736732 | — |
| R005 | epoch-0003.pt | EMA0.99 | 73.71% | 54.58% | 177/512 | 34.57% | 2048 | MixedBCE=0.603227 | — |
| R007 | epoch-0004.pt | EMA0.99 | 80.04% | 53.06% | 178/512 | 34.77% | 2048 | MixedBCE=0.484652 | — |

- 评测池：old unseen NQ-dev2048/G5, generated20261020; 641 Mixed; bankSHA5876bd9607b1e956d8752e13c6e54def5d77be683a940af5f03b4f0db176a64c; ordered qidsSHA02a3dad1b18eff01915dad654934510de148fa938658b40d32920104862baf0b
- Independent Top512 pool is the old641-Mixed NQ-dev pool, NOT the later663-Mixed HF fixedG5 pool. fixed_diagnostic_BA is a separate512-question diagnostic. No cross-pool ranking is valid.
- [原始汇总：metrics-and-acceptance.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-four-epoch-unseen2048-20261020/delivery/metrics-and-acceptance.json)

## Mixed-only continuation from warmupE2, Prefix20, rank weight0.1

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R011 | epoch-0001.pt | EMA0.99 | — | 50.31% | 182/512 | 35.55% | 2048 | MixedBCE=0.152067 | — |

- 评测池：old unseen NQ-dev2048/G5, generated20261020; 641 Mixed; bankSHA5876bd9607b1e956d8752e13c6e54def5d77be683a940af5f03b4f0db176a64c; ordered qidsSHA02a3dad1b18eff01915dad654934510de148fa938658b40d32920104862baf0b
- Independent Top512 pool is the old641-Mixed NQ-dev pool, NOT the later663-Mixed HF fixedG5 pool. fixed_diagnostic_BA is a separate512-question diagnostic. No cross-pool ranking is valid.
- [原始汇总：prefix20-epoch1-complete-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-prefix20-mixed-20261020/epoch1-complete/prefix20-epoch1-complete-audit.json)

## Prefix20 1:1 Mixed/nonMixed continuation, rank weight1.0

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R013 | epoch-0002.pt | EMA0.99 | 73.82% | 56.36% | 182/512 | 35.55% | 2048 | MixedBCE=0.593195 | — |
| R015 | epoch-0003.pt | EMA0.99 | 76.82% | 56.14% | 190/512 | 37.11% | 2048 | MixedBCE=0.545247 | — |
| R017 | epoch-0004.pt | EMA0.99 | 79.65% | — | 179/512 | 34.96% | 2048 | MixedBCE=0.489931 | — |
| R019 | epoch-0005.pt | EMA0.99 | 81.68% | — | 188/512 | 36.72% | 2048 | MixedBCE=0.453052 | — |

- 评测池：old unseen NQ-dev2048/G5, generated20261020; 641 Mixed; bankSHA5876bd9607b1e956d8752e13c6e54def5d77be683a940af5f03b4f0db176a64c; ordered qidsSHA02a3dad1b18eff01915dad654934510de148fa938658b40d32920104862baf0b
- Independent Top512 pool is the old641-Mixed NQ-dev pool, NOT the later663-Mixed HF fixedG5 pool. fixed_diagnostic_BA is a separate512-question diagnostic. No cross-pool ranking is valid.
- [原始汇总：mixed-ft-status-1792510209.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/mixed-ft-status-1792510209.json)
- [原始汇总：mixed-ft-status-1792514366.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/mixed-ft-status-1792514366.json)
- [原始汇总：mixed-ft-status-1792519011.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/mixed-ft-status-1792519011.json)
- [原始汇总：mixed-ft-status-1792523075.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/mixed-ft-status-1792523075.json)

## official-direct-Mixed-fresh-block27

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R025 | CP128 | EMA | — | 54.85% | 205/512 | 40.04% | 512 | — | BCE=0.685792 |
| R026 | CP128 | EMA | — | 49.32% | 162/512 | 31.64% | 2048 | — | BCE=0.681879 |
| R027 | CP128 | raw | — | 53.29% | 205/512 | 40.04% | 512 | — | BCE=0.692914 |
| R028 | CP128 | raw | — | 50.91% | 176/512 | 34.38% | 2048 | — | BCE=0.686178 |
| R029 | CP512 | EMA | — | 53.38% | 205/512 | 40.04% | 512 | — | BCE=0.681295 |
| R030 | CP512 | EMA | — | 51.04% | 190/512 | 37.11% | 2048 | — | BCE=0.662300 |
| R031 | CP512 | raw | — | 54.76% | 205/512 | 40.04% | 512 | — | BCE=0.685186 |
| R032 | CP512 | raw | — | 51.82% | 206/512 | 40.23% | 2048 | — | BCE=0.668843 |
| R033 | E0 | EMA | — | 48.39% | 205/512 | 40.04% | 512 | — | BCE=0.731487 |
| R034 | E0 | EMA | — | 49.41% | 164/512 | 32.03% | 2048 | — | BCE=0.772123 |
| R035 | E0 | raw | — | 48.39% | 205/512 | 40.04% | 512 | — | BCE=0.731487 |
| R036 | E0 | raw | — | 49.41% | 164/512 | 32.03% | 2048 | — | BCE=0.772123 |

- 评测池：diagnostic512
- 评测池：fixed_selection2048
- Direct Mixed BCE only; native block+group head; official initial block; 50:50 real G5 role sampler with repeated questions. diagnostic512 selecting512 is the entire pool, not a2048→512 ranking.
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/CP128/EMA/diagnostic512`
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/CP128/EMA/fixed_selection2048`
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/CP128/raw/diagnostic512`
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/CP128/raw/fixed_selection2048`
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/CP512/EMA/diagnostic512`
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/CP512/EMA/fixed_selection2048`
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/CP512/raw/diagnostic512`
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/CP512/raw/fixed_selection2048`
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/E0/EMA/diagnostic512`
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/E0/EMA/fixed_selection2048`
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/E0/raw/diagnostic512`
- [原始汇总：TERMINAL-PACKET.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval/TERMINAL-PACKET.json)，JSON位置 `/metrics/E0/raw/fixed_selection2048`

## official-direct-Mixed-original-hard-replay

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R037 | E1 | EMA | 58.99% | 52.88% | 205/512 | 40.04% | 2048 | BCE=0.671798 | BCE=0.653016 |
| R038 | E1 | EMA | 58.99% | 57.53% | 205/512 | 40.04% | 512 | BCE=0.671798 | BCE=0.669290 |
| R039 | E1 | raw | 58.87% | 52.28% | 201/512 | 39.26% | 2048 | BCE=0.671493 | BCE=0.643807 |
| R040 | E1 | raw | 58.87% | 56.64% | 205/512 | 40.04% | 512 | BCE=0.671493 | BCE=0.664226 |
| R041 | E2 | EMA | — | 48.07% | 158/512 | 30.86% | 2048 | — | BCE=0.967913 |
| R042 | E2 | EMA | — | 44.01% | 205/512 | 40.04% | 512 | — | BCE=0.866391 |
| R043 | E2 | raw | — | 48.25% | 156/512 | 30.47% | 2048 | — | BCE=1.035228 |
| R044 | E2 | raw | — | 44.18% | 205/512 | 40.04% | 512 | — | BCE=0.904104 |
| R045 | E3 | EMA | — | 51.26% | 205/512 | 40.04% | 2048 | — | BCE=0.623642 |
| R046 | E3 | EMA | — | 55.50% | 205/512 | 40.04% | 512 | — | BCE=0.680392 |
| R047 | E3 | raw | — | 50.99% | 199/512 | 38.87% | 2048 | — | BCE=0.624133 |
| R048 | E3 | raw | — | 55.26% | 205/512 | 40.04% | 512 | — | BCE=0.681262 |
| R049 | E4 | EMA | — | 53.38% | 191/512 | 37.30% | 2048 | — | BCE=0.768949 |
| R050 | E4 | EMA | — | 47.68% | 205/512 | 40.04% | 512 | — | BCE=0.719438 |
| R051 | E4 | raw | — | 52.01% | 198/512 | 38.67% | 2048 | — | BCE=0.802757 |
| R052 | E4 | raw | — | 49.79% | 205/512 | 40.04% | 512 | — | BCE=0.735016 |

- 评测池：fixed_selection2048
- 评测池：diagnostic512
- Original question-visit epochs; repeated role sampling, four G5 views/question occurrence, previous-hard-quartile replay after boundaries. Not fair evidence for LR causality; no fixed train eval for E2–4. E1 fixed-train result was subsequently obtained on the exact matched E1 checkpoint.
- Original question-visit epochs; repeated role sampling, four G5 views/question occurrence, previous-hard-quartile replay after boundaries. Not fair evidence for LR causality; no fixed train eval for E2–4. E1 fixed-train result was subsequently obtained on the exact matched E1 checkpoint. Whole512 pool selection is not ranking.
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-final-metrics/actual-independent-metrics.json)，JSON位置 `/metrics/1/EMA`
- [原始汇总：actual-evaluations-readonly-2.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-continuation/actual-evaluations-readonly-2.json)，JSON位置 `/epochs/epoch-01/models/EMA/diagnostic512`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-final-metrics/actual-independent-metrics.json)，JSON位置 `/metrics/1/raw`
- [原始汇总：actual-evaluations-readonly-2.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-continuation/actual-evaluations-readonly-2.json)，JSON位置 `/epochs/epoch-01/models/raw/diagnostic512`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-final-metrics/actual-independent-metrics.json)，JSON位置 `/metrics/2/EMA`
- [原始汇总：actual-evaluations-readonly-2.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-continuation/actual-evaluations-readonly-2.json)，JSON位置 `/epochs/epoch-02/models/EMA/diagnostic512`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-final-metrics/actual-independent-metrics.json)，JSON位置 `/metrics/2/raw`
- [原始汇总：actual-evaluations-readonly-2.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-continuation/actual-evaluations-readonly-2.json)，JSON位置 `/epochs/epoch-02/models/raw/diagnostic512`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-final-metrics/actual-independent-metrics.json)，JSON位置 `/metrics/3/EMA`
- [原始汇总：actual-evaluations-readonly-2.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-continuation/actual-evaluations-readonly-2.json)，JSON位置 `/epochs/epoch-03/models/EMA/diagnostic512`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-final-metrics/actual-independent-metrics.json)，JSON位置 `/metrics/3/raw`
- [原始汇总：actual-evaluations-readonly-2.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-continuation/actual-evaluations-readonly-2.json)，JSON位置 `/epochs/epoch-03/models/raw/diagnostic512`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-final-metrics/actual-independent-metrics.json)，JSON位置 `/metrics/4/EMA`
- [原始汇总：actual-evaluations-readonly-2.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-continuation/actual-evaluations-readonly-2.json)，JSON位置 `/epochs/epoch-04/models/EMA/diagnostic512`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-final-metrics/actual-independent-metrics.json)，JSON位置 `/metrics/4/raw`
- [原始汇总：actual-evaluations-readonly-2.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-b128-continuation/actual-evaluations-readonly-2.json)，JSON位置 `/epochs/epoch-04/models/raw/diagnostic512`

## E1-fork747723-hard-replay-lowLR-then-head10x

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R053 | E1 | EMA | 58.99% | 52.88% | 205/512 | 40.04% | 2048 | BCE=0.671798 | BCE=0.653016 |
| R054 | E1 | raw | 58.87% | 52.28% | 201/512 | 39.26% | 2048 | BCE=0.671493 | BCE=0.643807 |
| R055 | E2 | EMA | — | 48.78% | 176/512 | 34.38% | 2048 | — | BCE=0.946060 |
| R056 | E2 | raw | — | 48.11% | 165/512 | 32.23% | 2048 | — | BCE=0.941090 |
| R057 | E3 | EMA | — | 50.87% | 194/512 | 37.89% | 2048 | — | BCE=0.626374 |
| R058 | E3 | raw | — | 50.95% | 191/512 | 37.30% | 2048 | — | BCE=0.627012 |
| R059 | E2 | raw | — | 44.83% | 205/512 | 40.04% | 512 | — | BCE=0.855174 |
| R060 | E2 | EMA | — | 45.23% | 205/512 | 40.04% | 512 | — | BCE=0.860349 |
| R061 | E3 | raw | — | 55.50% | 205/512 | 40.04% | 512 | — | BCE=0.690570 |
| R062 | E3 | EMA | — | 55.83% | 205/512 | 40.04% | 512 | — | BCE=0.685751 |

- 评测池：fixed_selection2048
- 评测池：diagnostic512
- Exact shared official E1 baseline, not a new independently trained model.
- Continues exact E1 raw+EMA+Adam; old hard-replay/repeated-question sampler remains. E2 joint5e-7; E3 native frozen/head5e-5. Not fair isolated LR comparison. Fixed-train checkpoint BA absent; online train windows are separate.
- Same repeated-hard branch; entire diagnostic512 selected, not2048→512 ranking.
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-results/actual-independent-metrics.json)，JSON位置 `/reference_E1_metrics/EMA`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-results/actual-independent-metrics.json)，JSON位置 `/reference_E1_metrics/raw`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-results/actual-independent-metrics.json)，JSON位置 `/metrics/2/EMA`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-results/actual-independent-metrics.json)，JSON位置 `/metrics/2/raw`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-results/actual-independent-metrics.json)，JSON位置 `/metrics/3/EMA`
- [原始汇总：actual-independent-metrics.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-results/actual-independent-metrics.json)，JSON位置 `/metrics/3/raw`
- [原始汇总：actual-progress-7.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-20261022/actual-progress-7.json)，JSON位置 `/epoch_2_evaluation/metrics/raw/diagnostic512`
- [原始汇总：actual-progress-7.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-20261022/actual-progress-7.json)，JSON位置 `/epoch_2_evaluation/metrics/EMA/diagnostic512`
- [原始汇总：actual-progress-7.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-20261022/actual-progress-7.json)，JSON位置 `/epoch_3_evaluation/metrics/raw/diagnostic512`
- [原始汇总：actual-progress-7.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-20261022/actual-progress-7.json)，JSON位置 `/epoch_3_evaluation/metrics/EMA/diagnostic512`

## corrected-unique-class-balanced749015

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R063 | E1 | EMA | 58.99% | 57.53% | 205/512 | 40.04% | 512 | BCE=0.671798 | BCE=0.669290 |
| R064 | E1 | EMA | 58.99% | 52.88% | 205/512 | 40.04% | 2048 | BCE=0.671798 | BCE=0.653016 |
| R065 | E1 | raw | 58.87% | 56.64% | 205/512 | 40.04% | 512 | BCE=0.671493 | BCE=0.664226 |
| R066 | E1 | raw | 58.87% | 52.28% | 201/512 | 39.26% | 2048 | BCE=0.671493 | BCE=0.643807 |
| R067 | E2 | EMA | 59.69% | 57.94% | 205/512 | 40.04% | 512 | BCE=0.662616 | BCE=0.660824 |
| R068 | E2 | EMA | 59.69% | 53.57% | 201/512 | 39.26% | 2048 | BCE=0.662616 | BCE=0.636835 |
| R069 | E2 | raw | 59.18% | 58.66% | 205/512 | 40.04% | 512 | BCE=0.661705 | BCE=0.662766 |
| R070 | E2 | raw | 59.18% | 53.75% | 200/512 | 39.06% | 2048 | BCE=0.661705 | BCE=0.640283 |
| R071 | E3 | EMA | 59.98% | 57.77% | 205/512 | 40.04% | 512 | BCE=0.662745 | BCE=0.660665 |
| R072 | E3 | EMA | 59.98% | 53.97% | 201/512 | 39.26% | 2048 | BCE=0.662745 | BCE=0.636255 |
| R073 | E3 | raw | 60.00% | 58.02% | 205/512 | 40.04% | 512 | BCE=0.662476 | BCE=0.661065 |
| R074 | E3 | raw | 60.00% | 54.20% | 203/512 | 39.65% | 2048 | BCE=0.662476 | BCE=0.636744 |

- 评测池：diagnostic512
- 评测池：fixed_selection2048
- E1 raw/EMA separately restored; raw updates EMA(.99). E2 joint and E3 head-only both5e-7. Each new epoch has169028 unique questions, randomized order and authentic own-G10→G5 draws, tail68. Class-balanced84514/84514 true G5 with unique questions; no previous-hard replay. Whole512 selection is not2048→512 ranking.
- E1 raw/EMA separately restored; raw updates EMA(.99). E2 joint and E3 head-only both5e-7. Each new epoch has169028 unique questions, randomized order and authentic own-G10→G5 draws, tail68. Class-balanced84514/84514 true G5 with unique questions; no previous-hard replay.
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent1/metrics/EMA/diagnostic512`
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent1/metrics/EMA/fixed_selection2048`
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent1/metrics/raw/diagnostic512`
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent1/metrics/raw/fixed_selection2048`
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent2/metrics/EMA/diagnostic512`
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent2/metrics/EMA/fixed_selection2048`
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent2/metrics/raw/diagnostic512`
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent2/metrics/raw/fixed_selection2048`
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent3/metrics/EMA/diagnostic512`
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent3/metrics/EMA/fixed_selection2048`
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent3/metrics/raw/diagnostic512`
- [原始汇总：main-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/main-latest.json)，JSON位置 `/independent3/metrics/raw/fixed_selection2048`

## corrected-labelblind-uniformG5-classweighted749120

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R075 | E1 | EMA | 58.99% | 57.53% | 205/512 | 40.04% | 512 | BCE=0.671798 | BCE=0.669290 |
| R076 | E1 | EMA | 58.99% | 52.88% | 205/512 | 40.04% | 2048 | BCE=0.671798 | BCE=0.653016 |
| R077 | E1 | raw | 58.87% | 56.64% | 205/512 | 40.04% | 512 | BCE=0.671493 | BCE=0.664226 |
| R078 | E1 | raw | 58.87% | 52.28% | 201/512 | 39.26% | 2048 | BCE=0.671493 | BCE=0.643807 |
| R079 | E2 | EMA | 58.70% | 56.88% | 205/512 | 40.04% | 512 | BCE=0.663397 | BCE=0.661889 |
| R080 | E2 | EMA | 58.70% | 52.90% | 197/512 | 38.48% | 2048 | BCE=0.663397 | BCE=0.636712 |
| R081 | E2 | raw | 59.01% | 56.80% | 205/512 | 40.04% | 512 | BCE=0.664192 | BCE=0.658410 |
| R082 | E2 | raw | 59.01% | 52.19% | 198/512 | 38.67% | 2048 | BCE=0.664192 | BCE=0.632126 |
| R326 | E3 | raw | 59.76% | 52.40% | 195/512 | 38.09% | 2048 | BCE=0.664387 | BCE=0.630993 |
| R327 | E3 | EMA | 59.12% | 52.76% | 195/512 | 38.09% | 2048 | BCE=0.663482 | BCE=0.634047 |

- 评测池：diagnostic512
- 评测池：fixed_selection2048
- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- E1 raw/EMA separately restored; raw updates EMA(.99). E2 joint and E3 head-only both5e-7. Each new epoch has169028 unique questions, randomized order and authentic own-G10→G5 draws, tail68. Label-blind uniform real G5, class weights; no forced50:50 subset selection and no hard replay. Snapshot may predate E3 evaluation. Whole512 selection is not2048→512 ranking.
- E1 raw/EMA separately restored; raw updates EMA(.99). E2 joint and E3 head-only both5e-7. Each new epoch has169028 unique questions, randomized order and authentic own-G10→G5 draws, tail68. Label-blind uniform real G5, class weights; no forced50:50 subset selection and no hard replay. Snapshot may predate E3 evaluation.
- 已实际完成E3/cursor5219；更新旧的“E3未完成”状态。
- [原始汇总：uniform-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/uniform-latest.json)，JSON位置 `/independent1/metrics/EMA/diagnostic512`
- [原始汇总：uniform-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/uniform-latest.json)，JSON位置 `/independent1/metrics/EMA/fixed_selection2048`
- [原始汇总：uniform-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/uniform-latest.json)，JSON位置 `/independent1/metrics/raw/diagnostic512`
- [原始汇总：uniform-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/uniform-latest.json)，JSON位置 `/independent1/metrics/raw/fixed_selection2048`
- [原始汇总：uniform-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/uniform-latest.json)，JSON位置 `/independent2/metrics/EMA/diagnostic512`
- [原始汇总：uniform-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/uniform-latest.json)，JSON位置 `/independent2/metrics/EMA/fixed_selection2048`
- [原始汇总：uniform-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/uniform-latest.json)，JSON位置 `/independent2/metrics/raw/diagnostic512`
- [原始汇总：uniform-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/live-observations/uniform-latest.json)，JSON位置 `/independent2/metrics/raw/fixed_selection2048`
- [原始汇总：recent-all-pilot-epochs-remote.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/historical-all-experiments/recent-all-pilot-epochs-remote.json)

## joint124194-native-block27-plus-readout

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R083 | newE0=historical-balanced-P20-E4 | raw | — | 54.02% | 205/512 | 40.04% | 512 | — | — |
| R084 | newE0=historical-balanced-P20-E4 | raw | — | 55.21% | 190/512 | 37.11% | 2048 | — | — |
| R085 | newE0=historical-balanced-P20-E4 | EMA | — | 52.31% | 205/512 | 40.04% | 512 | — | — |
| R086 | newE0=historical-balanced-P20-E4 | EMA | — | 55.77% | 196/512 | 38.28% | 2048 | — | — |

- 评测池：diagnostic512
- 评测池：fixed_selection2048
- Trajectory BCE+product-formula Mixed BCE1:1; current start already historicalE4 Adam14418. Mixed/nonMixed50:50 role sampler with hard mining. No posttraining epoch1–4 evaluation in this completed-audit snapshot.
- [原始汇总：actual-epoch-quality-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/joint124194-root-latest/actual-epoch-quality-audit.json)，JSON位置 `/epoch_results/0/states/raw/diagnostic512`
- [原始汇总：actual-epoch-quality-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/joint124194-root-latest/actual-epoch-quality-audit.json)，JSON位置 `/epoch_results/0/states/raw/fixed_selection2048`
- [原始汇总：actual-epoch-quality-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/joint124194-root-latest/actual-epoch-quality-audit.json)，JSON位置 `/epoch_results/0/states/EMA/diagnostic512`
- [原始汇总：actual-epoch-quality-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/joint124194-root-latest/actual-epoch-quality-audit.json)，JSON位置 `/epoch_results/0/states/EMA/fixed_selection2048`

## matched-loss128/trajectory

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R099 | cursor128 | EMA | — | 51.87% | 187/512 | 36.52% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：trajectory-acceptance.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/matched-loss128-results/trajectory-acceptance.json)

## matched-loss128/joint

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R100 | cursor128 | EMA | — | 53.72% | 175/512 | 34.18% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：joint-acceptance.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/matched-loss128-results/joint-acceptance.json)

## matched-loss128/event

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R101 | cursor128 | EMA | — | 53.82% | 195/512 | 38.09% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：event-acceptance.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/matched-loss128-results/event-acceptance.json)

## followup-loss128/within1

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R102 | cursor128 | EMA | — | 54.16% | — | — | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/followup-loss128-results/independent-own-em-metric-audit.json)

## followup-loss128/inter1

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R103 | cursor128 | EMA | — | 54.52% | — | — | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/followup-loss128-results/independent-own-em-metric-audit.json)

## followup-loss128/bothrank1

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R104 | cursor128 | EMA | — | 54.52% | — | — | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/followup-loss128-results/independent-own-em-metric-audit.json)

## followup-loss128/mean1

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R105 | cursor128 | EMA | — | 53.40% | — | — | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/followup-loss128-results/independent-own-em-metric-audit.json)

## followup-loss128/var5

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R106 | cursor128 | EMA | — | 53.84% | — | — | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/followup-loss128-results/independent-own-em-metric-audit.json)

## followup-loss128/mean1-var5

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R107 | cursor128 | EMA | — | 53.35% | — | — | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/followup-loss128-results/independent-own-em-metric-audit.json)

## followup-loss128/bothrank1-mean1-var5

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R108 | cursor128 | EMA | — | 55.84% | — | — | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/followup-loss128-results/independent-own-em-metric-audit.json)

## followup-loss128/inter1-hard16

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R109 | cursor128 | EMA | — | 54.35% | — | — | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/followup-loss128-results/independent-own-em-metric-audit.json)

## l27-readout128/last-prefix

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R110 | cursor128 | EMA | — | 47.93% | 153/512 | 29.88% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-l27-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/l27-readout128-results/independent-l27-own-em-metric-audit.json)

## l27-readout128/mean-prefix

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R111 | cursor128 | EMA | — | 50.62% | 152/512 | 29.69% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-l27-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/l27-readout128-results/independent-l27-own-em-metric-audit.json)

## l27-readout128/mean-question-prefix

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R112 | cursor128 | EMA | — | 50.00% | 154/512 | 30.08% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-l27-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/l27-readout128-results/independent-l27-own-em-metric-audit.json)

## l27-readout128/attention-prefix

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R113 | cursor128 | EMA | — | 50.51% | 153/512 | 29.88% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-l27-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/l27-readout128-results/independent-l27-own-em-metric-audit.json)

## l27-readout128/attention-question-prefix

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R114 | cursor128 | EMA | — | 50.00% | 155/512 | 30.27% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-l27-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/l27-readout128-results/independent-l27-own-em-metric-audit.json)

## l27-readout128/flatten

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R115 | cursor128 | EMA | — | 50.49% | 159/512 | 31.05% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- EMA evaluated by archived runner; raw fixed-pool evaluation and fixed-training BA were not archived. 512 diagnostic is not training accuracy. Original all-169028 warmup had already seen canonical training texts. Fixed 2048 is recipe selection, not untouched final.
- [原始汇总：independent-l27-own-em-metric-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/l27-readout128-results/independent-l27-own-em-metric-audit.json)

## fresh-official-layer21

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R116 | epoch1 | EMA | — | 50.64% | 174/512 | 33.98% | 2048 | — | — |
| R117 | epoch2 | EMA | — | 50.88% | 182/512 | 35.55% | 2048 | — | — |
| R118 | epoch3 | EMA | — | 51.94% | 189/512 | 36.91% | 2048 | — | — |
| R119 | epoch4 | EMA | — | 53.06% | 189/512 | 36.91% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- Fresh official block/new common head, not E1 resume. Same6528/4epoch plan among layers21/24/27. EMA evaluated; raw/fixed-training BA not archived. All rows use OLD641-Mixed selection2048; do not rank against new663-Mixed online8192 pool.
- [原始汇总：terminal-read.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/multilayer-final-results/terminal-read.json)

## fresh-official-layer24

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R120 | epoch1 | EMA | — | 49.98% | 182/512 | 35.55% | 2048 | — | — |
| R121 | epoch2 | EMA | — | 50.57% | 187/512 | 36.52% | 2048 | — | — |
| R122 | epoch3 | EMA | — | 51.30% | 184/512 | 35.94% | 2048 | — | — |
| R123 | epoch4 | EMA | — | 52.80% | 188/512 | 36.72% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- Fresh official block/new common head, not E1 resume. Same6528/4epoch plan among layers21/24/27. EMA evaluated; raw/fixed-training BA not archived. All rows use OLD641-Mixed selection2048; do not rank against new663-Mixed online8192 pool.
- [原始汇总：terminal-read.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/multilayer-final-results/terminal-read.json)

## fresh-official-layer27

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R124 | epoch1 | EMA | — | 51.64% | 169/512 | 33.01% | 2048 | — | — |
| R125 | epoch2 | EMA | — | 53.61% | 166/512 | 32.42% | 2048 | — | — |
| R126 | epoch3 | EMA | — | 53.65% | 164/512 | 32.03% | 2048 | — | — |
| R127 | epoch4 | EMA | — | 54.28% | 177/512 | 34.57% | 2048 | — | — |

- 评测池：historical selection2048 ownG5 / 641 Mixed; differs from fixed-HF G5 / 663 Mixed
- Fresh official block/new common head, not E1 resume. Same6528/4epoch plan among layers21/24/27. EMA evaluated; raw/fixed-training BA not archived. All rows use OLD641-Mixed selection2048; do not rank against new663-Mixed online8192 pool.
- [原始汇总：terminal-read.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/multilayer-final-results/terminal-read.json)

## peer8192-rank/inter1

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R128 | cursor256 | EMA | — | 55.48% | 189/512 | 36.91% | 2048 | — | — |

- 评测池：fixed-HF selection2048 genuine fixedG5 / 663 Mixed
- EMA-only fixed-pool scoring; raw/fixed-train BA missing. Same E1 initial state+256 updates as no-rank peer control. Different feature backend and G5 identity from old641 pool.
- [原始汇总：actual-inter1-final-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-ranking-matched256/actual-inter1-final-audit.json)

## peer8192/no-rank-control

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R129 | cursor256 | EMA | — | 54.66% | 182/512 | 35.55% | 2048 | — | — |

- 评测池：fixed-HF selection2048 genuine fixedG5 / 663 Mixed
- The actual separately trained original no-rank peer CP256, re-used as matched control. Not an additional fit in this summary. Fixed-train/raw BA missing.
- [原始汇总：actual-inter1-final-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-ranking-matched256/actual-inter1-final-audit.json)

## peer8192-rank/bothrank1

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R130 | cursor256 | EMA | — | 55.09% | 186/512 | 36.33% | 2048 | — | — |

- 评测池：fixed-HF selection2048 genuine fixedG5 / 663 Mixed
- EMA-only fixed-pool scoring; raw/fixed-train BA missing. Same E1 initial state+256 updates as no-rank peer control. Different feature backend and G5 identity from old641 pool.
- [原始汇总：actual-bothrank1-final-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-ranking-matched256/actual-bothrank1-final-audit.json)

## frozen-L26-linear/prefix_mean/L2=0.001

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R131 | LBFGS80 | plain linear head (no EMA) | — | 53.61% | — | — | 1589 | BCE=0.230629 | — |

- 评测池：internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048
- Threshold selected on this internal holdout; optimistic for threshold BA. Train BA not recorded. Only the unique winner, prefix_mean/L2=.1, was later scored externally; other five have NO external metric. No original block/Adam updates.
- [原始汇总：independent-fixed-and-CV-threshold-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/peer8192-linear-group-head/independent-fixed-and-CV-threshold-audit.json)

## frozen-L26-linear/prefix_mean/L2=0.01

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R132 | LBFGS80 | plain linear head (no EMA) | — | 54.65% | — | — | 1589 | BCE=0.285870 | — |

- 评测池：internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048
- Threshold selected on this internal holdout; optimistic for threshold BA. Train BA not recorded. Only the unique winner, prefix_mean/L2=.1, was later scored externally; other five have NO external metric. No original block/Adam updates.
- [原始汇总：independent-fixed-and-CV-threshold-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/peer8192-linear-group-head/independent-fixed-and-CV-threshold-audit.json)

## frozen-L26-linear/prefix_mean/L2=0.1

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R133 | LBFGS80 | plain linear head (no EMA) | — | 56.12% | — | — | 1589 | BCE=0.421061 | — |

- 评测池：internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048
- Threshold selected on this internal holdout; optimistic for threshold BA. Train BA not recorded. Only the unique winner, prefix_mean/L2=.1, was later scored externally; other five have NO external metric. No original block/Adam updates.
- [原始汇总：independent-fixed-and-CV-threshold-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/peer8192-linear-group-head/independent-fixed-and-CV-threshold-audit.json)

## frozen-L26-linear/prefix_last/L2=0.001

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R134 | LBFGS80 | plain linear head (no EMA) | — | 52.32% | — | — | 1589 | BCE=0.275267 | — |

- 评测池：internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048
- Threshold selected on this internal holdout; optimistic for threshold BA. Train BA not recorded. Only the unique winner, prefix_mean/L2=.1, was later scored externally; other five have NO external metric. No original block/Adam updates.
- [原始汇总：independent-fixed-and-CV-threshold-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/peer8192-linear-group-head/independent-fixed-and-CV-threshold-audit.json)

## frozen-L26-linear/prefix_last/L2=0.01

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R135 | LBFGS80 | plain linear head (no EMA) | — | 52.91% | — | — | 1589 | BCE=0.319483 | — |

- 评测池：internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048
- Threshold selected on this internal holdout; optimistic for threshold BA. Train BA not recorded. Only the unique winner, prefix_mean/L2=.1, was later scored externally; other five have NO external metric. No original block/Adam updates.
- [原始汇总：independent-fixed-and-CV-threshold-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/peer8192-linear-group-head/independent-fixed-and-CV-threshold-audit.json)

## frozen-L26-linear/prefix_last/L2=0.1

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R136 | LBFGS76 | plain linear head (no EMA) | — | 54.17% | — | — | 1589 | BCE=0.432031 | — |

- 评测池：internal linear-head holdout1589 q /6356 equal-weight views; NOT external2048
- Threshold selected on this internal holdout; optimistic for threshold BA. Train BA not recorded. Only the unique winner, prefix_mean/L2=.1, was later scored externally; other five have NO external metric. No original block/Adam updates.
- [原始汇总：independent-fixed-and-CV-threshold-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/peer8192-linear-group-head/independent-fixed-and-CV-threshold-audit.json)

## frozen-L26-linear/winner-prefix_mean-L2=.1/fixed0.5

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R137 | same winner LBFGS80 | plain linear head (no EMA) | — | 49.95% | 158/512 | 30.86% | 2048 | — | — |

- 评测池：fixed-HF external viewed2048 /663 Mixed; exact same logits both thresholds
- One unique fitted winner; two predeclared thresholds on same single external scoring pass. AUC/Top512 unchanged across thresholds. Not two models, not an EMA comparison. Train BA missing.
- [原始汇总：independent-fixed-and-CV-threshold-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/peer8192-linear-group-head/independent-fixed-and-CV-threshold-audit.json)

## frozen-L26-linear/winner-prefix_mean-L2=.1/internal-CV-threshold

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R138 | same winner LBFGS80 | plain linear head (no EMA) | — | 51.82% | 158/512 | 30.86% | 2048 | — | — |

- 评测池：fixed-HF external viewed2048 /663 Mixed; exact same logits both thresholds
- One unique fitted winner; two predeclared thresholds on same single external scoring pass. AUC/Top512 unchanged across thresholds. Not two models, not an EMA comparison. Train BA missing.
- [原始汇总：independent-fixed-and-CV-threshold-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/peer8192-linear-group-head/independent-fixed-and-CV-threshold-audit.json)

## prefix-confidence/matched_platt2

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R139 | completed auxiliary fit | auxiliary scalar model (no EMA) | — | 63.44% | 229/512 | 44.73% | 2048 | — | — |

- 评测池：INTERNAL NEWSEED5 whole-qid holdout2048 /655 Mixed; already seen by original probe warmup
- Auxiliary-calibrator internal holdout, not unseen-to-original-probe external2048; original frozen probe/Adam unchanged. Failed internal gate, external predictions not opened. Train BA not archived.
- [原始汇总：prefix-confidence-final-delivery.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/prefix-confidence/actual/prefix-confidence-final-delivery.json)

## prefix-confidence/prefix_full8

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R140 | completed auxiliary fit | auxiliary scalar model (no EMA) | — | 63.36% | 232/512 | 45.31% | 2048 | — | — |

- 评测池：INTERNAL NEWSEED5 whole-qid holdout2048 /655 Mixed; already seen by original probe warmup
- Auxiliary-calibrator internal holdout, not unseen-to-original-probe external2048; original frozen probe/Adam unchanged. Failed internal gate, external predictions not opened. Train BA not archived.
- [原始汇总：prefix-confidence-final-delivery.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/prefix-confidence/actual/prefix-confidence-final-delivery.json)

## prefix-confidence/raw_E4

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R141 | frozen E4 prediction reference | uncalibrated frozen E4 prediction; raw label is not raw-weight state | — | 64.05% | 238/512 | 46.48% | 2048 | — | — |

- 评测池：INTERNAL NEWSEED5 whole-qid holdout2048 /655 Mixed; already seen by original probe warmup
- Auxiliary-calibrator internal holdout, not unseen-to-original-probe external2048; original frozen probe/Adam unchanged. Failed internal gate, external predictions not opened. Train BA not archived.
- [原始汇总：prefix-confidence-final-delivery.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/prefix-confidence/actual/prefix-confidence-final-delivery.json)

## G10-exact-uniform-all252

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R142 | cursor136 | EMA | — | 55.39% | 197/512 | 38.48% | 2048 | — | — |

- 评测池：fixed native-backend selection2048 ownG5 /663 Mixed; not same numerical HF feature backend as peer8192
- EMA-only archive, raw/train BA missing. Same E1/fullstate/order/B48x136 as random4 control. Exact averaging did not pass E4 jointgate.
- [原始汇总：actual-final-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/exact-uniform-all-G5/actual-final-independent-audit.json)

## G10-exact-reference/native_E4

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R143 | source reference or cursor136 | EMA | — | 55.53% | 200/512 | 39.06% | 2048 | — | — |

- 评测池：fixed native-backend selection2048 ownG5 /663 Mixed
- Archived already-fitted reference, not another new fit. Numerical backend differs from HF peer fixedG5 comparisons.
- [原始汇总：actual-final-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/exact-uniform-all-G5/actual-final-independent-audit.json)

## G10-exact-reference/same_G10_random4_B48x136

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R144 | source reference or cursor136 | EMA | — | 55.15% | 194/512 | 37.89% | 2048 | — | — |

- 评测池：fixed native-backend selection2048 ownG5 /663 Mixed
- Archived already-fitted reference, not another new fit. Numerical backend differs from HF peer fixedG5 comparisons.
- [原始汇总：actual-final-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/exact-uniform-all-G5/actual-final-independent-audit.json)

## 直接hidden六类count（block27＋head）

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R145 | CP256 | 已发布评分模型；未拆raw/EMA | — | 49.89% | 171/512 | 33.40% | 2048 | — | — |
| R146 | CP512 | 已发布评分模型；未拆raw/EMA | — | 49.96% | 177/512 | 34.57% | 2048 | — | — |
| R147 | CP768 | 已发布评分模型；未拆raw/EMA | — | 50.00% | 182/512 | 35.55% | 2048 | — | — |
| R148 | CP938 | 已发布评分模型；未拆raw/EMA | — | 50.00% | 183/512 | 35.74% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 30016题/938更新；固定训练BA未留在该评测回执，不能用在线batch替代
- [原始汇总：actual-cursor-000256-fixed-evaluation.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-hidden-count/actual-cursor-000256-fixed-evaluation.json)
- [原始汇总：actual-cursor-000512-fixed-evaluation.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-hidden-count/actual-cursor-000512-fixed-evaluation.json)
- [原始汇总：actual-cursor-000768-fixed-evaluation.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-hidden-count/actual-cursor-000768-fixed-evaluation.json)
- [原始汇总：actual-final938-summary.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-hidden-count/actual-final938-summary.json)

## 直接count：仅head LR升至1e-4

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R149 | CP256 | 已发布评分模型；未拆raw/EMA | — | 50.00% | 188/512 | 36.72% | 2048 | — | — |
| R150 | CP512 | 已发布评分模型；未拆raw/EMA | — | 50.00% | 195/512 | 38.09% | 2048 | — | — |
| R151 | CP768 | 已发布评分模型；未拆raw/EMA | — | 50.00% | 190/512 | 37.11% | 2048 | — | — |
| R152 | CP938 | 已发布评分模型；未拆raw/EMA | — | 50.00% | 199/512 | 38.87% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 与原count单次938更新计划匹配；不是head-only小样本诊断
- [原始汇总：actual-cursor-000256-accepted.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-headlr1e4/actual-cursor-000256-accepted.json)
- [原始汇总：actual-cursor-000512-accepted.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-headlr1e4/actual-cursor-000512-accepted.json)
- [原始汇总：actual-cursor-000768-accepted.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-headlr1e4/actual-cursor-000768-accepted.json)
- [原始汇总：actual-cursor-000938-accepted.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-headlr1e4/actual-cursor-000938-accepted.json)

## 冻结E1 count-head拟合容量 LR=0.0001

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R153 | E0 | EMA | 50.00% | 50.00% | 97/256 | 37.89% | 1024 | — | — |
| R154 | E0 | raw | 50.00% | 50.00% | 97/256 | 37.89% | 1024 | — | — |
| R155 | E1 | EMA | 50.00% | 50.07% | 96/256 | 37.50% | 1024 | — | — |
| R156 | E1 | raw | 50.00% | 50.00% | 77/256 | 30.08% | 1024 | — | — |
| R157 | E10 | EMA | 56.67% | 50.48% | 87/256 | 33.98% | 1024 | — | — |
| R158 | E10 | raw | 80.94% | 52.23% | 102/256 | 39.84% | 1024 | — | — |
| R159 | E2 | EMA | 51.18% | 50.41% | 95/256 | 37.11% | 1024 | — | — |
| R160 | E2 | raw | 50.00% | 50.00% | 76/256 | 29.69% | 1024 | — | — |
| R161 | E20 | EMA | 96.50% | 52.69% | 97/256 | 37.89% | 1024 | — | — |
| R162 | E20 | raw | 98.10% | 52.83% | 98/256 | 38.28% | 1024 | — | — |
| R163 | E5 | EMA | 51.71% | 49.84% | 83/256 | 32.42% | 1024 | — | — |
| R164 | E5 | raw | 50.58% | 50.00% | 83/256 | 32.42% | 1024 | — | — |

- 评测池：head-only内部：fit512/holdout1024
- 原native已见题目；Top256不能充当2048→512独立结果
- [原始汇总：FINAL.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-capacity/FINAL.json)

## 冻结E1 count-head拟合容量 LR=0.001

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R165 | E0 | EMA | 50.00% | 50.00% | 97/256 | 37.89% | 1024 | — | — |
| R166 | E0 | raw | 50.00% | 50.00% | 97/256 | 37.89% | 1024 | — | — |
| R167 | E1 | EMA | 50.45% | 49.77% | 86/256 | 33.59% | 1024 | — | — |
| R168 | E1 | raw | 50.00% | 50.00% | 92/256 | 35.94% | 1024 | — | — |
| R169 | E10 | EMA | 58.26% | 51.16% | 89/256 | 34.77% | 1024 | — | — |
| R170 | E10 | raw | 51.74% | 50.09% | 90/256 | 35.16% | 1024 | — | — |
| R171 | E2 | EMA | 50.00% | 50.00% | 84/256 | 32.81% | 1024 | — | — |
| R172 | E2 | raw | 50.00% | 50.00% | 93/256 | 36.33% | 1024 | — | — |
| R173 | E20 | EMA | 58.54% | 51.65% | 95/256 | 37.11% | 1024 | — | — |
| R174 | E20 | raw | 62.58% | 53.05% | 99/256 | 38.67% | 1024 | — | — |
| R175 | E5 | EMA | 50.59% | 49.98% | 89/256 | 34.77% | 1024 | — | — |
| R176 | E5 | raw | 58.63% | 50.68% | 88/256 | 34.38% | 1024 | — | — |

- 评测池：head-only内部：fit512/holdout1024
- 原native已见题目；Top256不能充当2048→512独立结果
- [原始汇总：FINAL.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-capacity/FINAL.json)

## 冻结E1 count-head拟合容量 LR=5e-06

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R177 | E0 | EMA | 50.00% | 50.00% | 97/256 | 37.89% | 1024 | — | — |
| R178 | E0 | raw | 50.00% | 50.00% | 97/256 | 37.89% | 1024 | — | — |
| R179 | E1 | EMA | 50.00% | 50.00% | 97/256 | 37.89% | 1024 | — | — |
| R180 | E1 | raw | 50.00% | 50.07% | 97/256 | 37.89% | 1024 | — | — |
| R181 | E10 | EMA | 54.22% | 52.14% | 102/256 | 39.84% | 1024 | — | — |
| R182 | E10 | raw | 52.44% | 51.12% | 95/256 | 37.11% | 1024 | — | — |
| R183 | E2 | EMA | 50.00% | 50.00% | 96/256 | 37.50% | 1024 | — | — |
| R184 | E2 | raw | 50.15% | 49.92% | 95/256 | 37.11% | 1024 | — | — |
| R185 | E20 | EMA | 55.48% | 51.12% | 100/256 | 39.06% | 1024 | — | — |
| R186 | E20 | raw | 55.50% | 51.73% | 97/256 | 37.89% | 1024 | — | — |
| R187 | E5 | EMA | 50.00% | 49.92% | 95/256 | 37.11% | 1024 | — | — |
| R188 | E5 | raw | 51.49% | 52.97% | 100/256 | 39.06% | 1024 | — | — |

- 评测池：head-only内部：fit512/holdout1024
- 原native已见题目；Top256不能充当2048→512独立结果
- [原始汇总：FINAL.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-capacity/FINAL.json)

## 冻结E1大样本count-head /NQ

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R189 | E0 | raw | 50.00% | 50.00% | 208/640 | 32.50% | 2560 | — | — |
| R192 | E0 | EMA | 50.00% | 50.00% | 208/640 | 32.50% | 2560 | — | — |
| R195 | E1 | raw | 50.03% | 50.03% | 242/640 | 37.81% | 2560 | — | — |
| R198 | E1 | EMA | 50.02% | 50.03% | 244/640 | 38.12% | 2560 | — | — |
| R201 | E2 | raw | 50.08% | 50.03% | 246/640 | 38.44% | 2560 | — | — |
| R204 | E2 | EMA | 50.06% | 50.03% | 246/640 | 38.44% | 2560 | — | — |
| R207 | E4 | raw | 50.12% | 50.03% | 246/640 | 38.44% | 2560 | — | — |
| R210 | E4 | EMA | 50.11% | 50.06% | 249/640 | 38.91% | 2560 | — | — |
| R213 | E8 | raw | 50.27% | 50.16% | 248/640 | 38.75% | 2560 | — | — |
| R216 | E8 | EMA | 50.46% | 50.33% | 254/640 | 39.69% | 2560 | — | — |
| R219 | E20 | raw | 52.35% | 52.25% | 255/640 | 39.84% | 2560 | — | — |
| R222 | E20 | EMA | 51.57% | 51.56% | 256/640 | 40.00% | 2560 | — | — |

- 评测池：原warmup题目内部留出/NQ
- fit30016，holdout5120(NQ2560+Hotpot2560)，head-only20ep；不是native未见问题
- [原始汇总：actual-complete-fit-curve.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-large-capacity/actual-complete-fit-curve.json)

## 冻结E1大样本count-head /HotpotQA

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R190 | E0 | raw | 50.00% | 50.04% | 289/640 | 45.16% | 2560 | — | — |
| R193 | E0 | EMA | 50.00% | 50.04% | 289/640 | 45.16% | 2560 | — | — |
| R196 | E1 | raw | 50.03% | 50.04% | 305/640 | 47.66% | 2560 | — | — |
| R199 | E1 | EMA | 50.02% | 50.04% | 311/640 | 48.59% | 2560 | — | — |
| R202 | E2 | raw | 50.08% | 50.10% | 294/640 | 45.94% | 2560 | — | — |
| R205 | E2 | EMA | 50.06% | 50.06% | 302/640 | 47.19% | 2560 | — | — |
| R208 | E4 | raw | 50.12% | 50.22% | 283/640 | 44.22% | 2560 | — | — |
| R211 | E4 | EMA | 50.11% | 50.13% | 293/640 | 45.78% | 2560 | — | — |
| R214 | E8 | raw | 50.27% | 50.03% | 305/640 | 47.66% | 2560 | — | — |
| R217 | E8 | EMA | 50.46% | 50.01% | 296/640 | 46.25% | 2560 | — | — |
| R220 | E20 | raw | 52.35% | 51.01% | 318/640 | 49.69% | 2560 | — | — |
| R223 | E20 | EMA | 51.57% | 50.52% | 315/640 | 49.22% | 2560 | — | — |

- 评测池：原warmup题目内部留出/HotpotQA
- fit30016，holdout5120(NQ2560+Hotpot2560)，head-only20ep；不是native未见问题
- [原始汇总：actual-complete-fit-curve.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-large-capacity/actual-complete-fit-curve.json)

## 冻结E1大样本count-head /pooled

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R191 | E0 | raw | 50.00% | 50.02% | 469/1280 | 36.64% | 5120 | — | — |
| R194 | E0 | EMA | 50.00% | 50.02% | 469/1280 | 36.64% | 5120 | — | — |
| R197 | E1 | raw | 50.03% | 50.03% | 581/1280 | 45.39% | 5120 | — | — |
| R200 | E1 | EMA | 50.02% | 50.03% | 579/1280 | 45.23% | 5120 | — | — |
| R203 | E2 | raw | 50.08% | 50.08% | 561/1280 | 43.83% | 5120 | — | — |
| R206 | E2 | EMA | 50.06% | 50.05% | 567/1280 | 44.30% | 5120 | — | — |
| R209 | E4 | raw | 50.12% | 50.14% | 545/1280 | 42.58% | 5120 | — | — |
| R212 | E4 | EMA | 50.11% | 50.10% | 562/1280 | 43.91% | 5120 | — | — |
| R215 | E8 | raw | 50.27% | 50.09% | 564/1280 | 44.06% | 5120 | — | — |
| R218 | E8 | EMA | 50.46% | 50.14% | 572/1280 | 44.69% | 5120 | — | — |
| R221 | E20 | raw | 52.35% | 51.62% | 591/1280 | 46.17% | 5120 | — | — |
| R224 | E20 | EMA | 51.57% | 50.92% | 578/1280 | 45.16% | 5120 | — | — |

- 评测池：原warmup题目内部留出/pooled
- fit30016，holdout5120(NQ2560+Hotpot2560)，head-only20ep；不是native未见问题
- [原始汇总：actual-complete-fit-curve.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-large-capacity/actual-complete-fit-curve.json)

## 大样本count同cache原E1产品基线/NQ

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R225 | E1冻结 | published baseline | — | 50.48% | 207/640 | 32.34% | 2560 | — | — |

- 评测池：原warmup题目内部留出/NQ
- 不是新增训练；和该内部count实验逐题匹配
- [原始汇总：actual-complete-fit-curve.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-large-capacity/actual-complete-fit-curve.json)

## 大样本count同cache原E1产品基线/HotpotQA

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R226 | E1冻结 | published baseline | — | 53.03% | 329/640 | 51.41% | 2560 | — | — |

- 评测池：原warmup题目内部留出/HotpotQA
- 不是新增训练；和该内部count实验逐题匹配
- [原始汇总：actual-complete-fit-curve.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-large-capacity/actual-complete-fit-curve.json)

## 大样本count同cache原E1产品基线/pooled

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R227 | E1冻结 | published baseline | — | 51.21% | 543/1280 | 42.42% | 5120 | — | — |

- 评测池：原warmup题目内部留出/pooled
- 不是新增训练；和该内部count实验逐题匹配
- [原始汇总：actual-complete-fit-curve.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-large-capacity/actual-complete-fit-curve.json)

## 旧product-q同HF复评/E1

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R228 | E1 | 源报告公布模型 | — | 50.68% | 185/512 | 36.13% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 旧轨迹BCE+产品Mixed路线/同HF公平复评；与原vLLM指标分开
- [原始汇总：RESULT.md](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/platt-g10-calibration/RESULT.md)

## 旧product-q同HF复评/E2

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R229 | E2 | 源报告公布模型 | — | 56.25% | 181/512 | 35.35% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 旧轨迹BCE+产品Mixed路线/同HF公平复评；与原vLLM指标分开
- [原始汇总：RESULT.md](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/platt-g10-calibration/RESULT.md)

## 旧product-q同HF复评/E4

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R230 | E4 | 源报告公布模型 | — | 55.77% | 196/512 | 38.28% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 旧轨迹BCE+产品Mixed路线/同HF公平复评；与原vLLM指标分开
- [原始汇总：RESULT.md](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/platt-g10-calibration/RESULT.md)

## 旧product-q同HF复评/peer256

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R231 | peer256 | 源报告公布模型 | — | 54.66% | 182/512 | 35.55% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 旧轨迹BCE+产品Mixed路线/同HF公平复评；与原vLLM指标分开
- [原始汇总：RESULT.md](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/platt-g10-calibration/RESULT.md)

## Platt两参数标定peer256（12内部候选冻结1）

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R232 | 冻结winner | frozen calibrated model | — | 54.46% | 189/512 | 36.91% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 训练6144、参数holdout2048，外部2048只评冻结winner；未用外部调阈值
- [原始汇总：independent-final-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/platt-g10-calibration/independent-final-audit.json)

## 训练先验decision诊断/E2

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R233 | primary | same saved weights | — | 56.25% | 181/512 | 35.35% | 2048 | — | — |
| R234 | train_prior_decision | same saved weights | — | 55.00% | 181/512 | 35.35% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 只改训练侧先验decision规则，非新训练；Top排序保持不变
- [原始汇总：FINAL.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/full938-prior-decision-diagnostic/FINAL.json)

## 训练先验decision诊断/E4

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R235 | primary | same saved weights | — | 55.77% | 196/512 | 38.28% | 2048 | — | — |
| R236 | train_prior_decision | same saved weights | — | 55.26% | 196/512 | 38.28% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 只改训练侧先验decision规则，非新训练；Top排序保持不变
- [原始汇总：FINAL.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/full938-prior-decision-diagnostic/FINAL.json)

## 训练先验decision诊断/control256

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R237 | primary | same saved weights | — | 49.89% | 171/512 | 33.40% | 2048 | — | — |
| R238 | train_prior_decision | same saved weights | — | 50.41% | 171/512 | 33.40% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 只改训练侧先验decision规则，非新训练；Top排序保持不变
- [原始汇总：FINAL.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/full938-prior-decision-diagnostic/FINAL.json)

## 训练先验decision诊断/control512

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R239 | primary | same saved weights | — | 49.96% | 177/512 | 34.57% | 2048 | — | — |
| R240 | train_prior_decision | same saved weights | — | 53.69% | 177/512 | 34.57% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 只改训练侧先验decision规则，非新训练；Top排序保持不变
- [原始汇总：FINAL.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/full938-prior-decision-diagnostic/FINAL.json)

## 训练先验decision诊断/control768

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R241 | primary | same saved weights | — | 50.00% | 182/512 | 35.55% | 2048 | — | — |
| R242 | train_prior_decision | same saved weights | — | 53.89% | 182/512 | 35.55% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 只改训练侧先验decision规则，非新训练；Top排序保持不变
- [原始汇总：FINAL.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/full938-prior-decision-diagnostic/FINAL.json)

## Append-only peer256继续扩数据

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R243 | CP512 | published evaluator | — | 55.69% | 185/512 | 36.13% | 2048 | — | — |
| R244 | CP768 | published evaluator | — | 55.23% | 195/512 | 38.09% | 2048 | — | — |
| R245 | CP938 | published evaluator | — | 55.98% | 194/512 | 37.89% | 2048 | — | — |

- 评测池：HF固定G5独立2048（旧产品/count支线）
- 保留原256更新，增加682至938；无Adam重置、未重复原问题前缀
- [原始汇总：actual-cursor-000512-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/append-only-682/actual-cursor-000512-independent-audit.json)
- [原始汇总：actual-cursor-000768-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/append-only-682/actual-cursor-000768-independent-audit.json)
- [原始汇总：actual-cursor-000938-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/append-only-682/actual-cursor-000938-independent-audit.json)

## G10 exact全部252个G5子集

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R246 | CP136 | published evaluator | — | 55.39% | 197/512 | 38.48% | 2048 | — | — |

- 评测池：旧native固定G5独立2048
- 6528题/136更新，与random4同预算；后端口径与HF复评分开
- [原始汇总：actual-final-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/exact-uniform-all-G5/actual-final-independent-audit.json)

## Exact-G5匹配参考/native_E4

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R247 | frozen reference | published evaluator | — | 55.53% | 200/512 | 39.06% | 2048 | — | — |

- 评测池：旧native固定G5独立2048
- 同exact252协议逐题配对，非额外新增训练
- [原始汇总：actual-final-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/exact-uniform-all-G5/actual-final-independent-audit.json)

## Exact-G5匹配参考/same_G10_random4_B48x136

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R248 | frozen reference | published evaluator | — | 55.15% | 194/512 | 37.89% | 2048 | — | — |

- 评测池：旧native固定G5独立2048
- 同exact252协议逐题配对，非额外新增训练
- [原始汇总：actual-final-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/exact-uniform-all-G5/actual-final-independent-audit.json)

## E2-E4-fixed-ensemble/trajectory_probability_mix-w25

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R249 | frozen E2 and E4 | EMA prediction ensemble | — | 55.91% | 193/512 | 37.70% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- Exactly6 preregistered CPU combinations; no fitting/threshold tuning/new probe. E2/E4 numerical HF backend,663 Mixed pool. Not an independent trained model; all jointgates failed.
- [原始汇总：e2-e4-fixed-cpu-actual.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/e2-e4-fixed-cpu-actual.json)

## E2-E4-fixed-ensemble/trajectory_probability_mix-w50

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R250 | frozen E2 and E4 | EMA prediction ensemble | — | 56.14% | 187/512 | 36.52% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- Exactly6 preregistered CPU combinations; no fitting/threshold tuning/new probe. E2/E4 numerical HF backend,663 Mixed pool. Not an independent trained model; all jointgates failed.
- [原始汇总：e2-e4-fixed-cpu-actual.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/e2-e4-fixed-cpu-actual.json)

## E2-E4-fixed-ensemble/trajectory_probability_mix-w75

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R251 | frozen E2 and E4 | EMA prediction ensemble | — | 56.36% | 189/512 | 36.91% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- Exactly6 preregistered CPU combinations; no fitting/threshold tuning/new probe. E2/E4 numerical HF backend,663 Mixed pool. Not an independent trained model; all jointgates failed.
- [原始汇总：e2-e4-fixed-cpu-actual.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/e2-e4-fixed-cpu-actual.json)

## E2-E4-fixed-ensemble/direct_Mixed_score_mix-w25

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R252 | frozen E2 and E4 | EMA prediction ensemble | — | 56.08% | 195/512 | 38.09% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- Exactly6 preregistered CPU combinations; no fitting/threshold tuning/new probe. E2/E4 numerical HF backend,663 Mixed pool. Not an independent trained model; all jointgates failed.
- [原始汇总：e2-e4-fixed-cpu-actual.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/e2-e4-fixed-cpu-actual.json)

## E2-E4-fixed-ensemble/direct_Mixed_score_mix-w50

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R253 | frozen E2 and E4 | EMA prediction ensemble | — | 55.98% | 187/512 | 36.52% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- Exactly6 preregistered CPU combinations; no fitting/threshold tuning/new probe. E2/E4 numerical HF backend,663 Mixed pool. Not an independent trained model; all jointgates failed.
- [原始汇总：e2-e4-fixed-cpu-actual.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/e2-e4-fixed-cpu-actual.json)

## E2-E4-fixed-ensemble/direct_Mixed_score_mix-w75

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R254 | frozen E2 and E4 | EMA prediction ensemble | — | 56.30% | 183/512 | 35.74% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- Exactly6 preregistered CPU combinations; no fitting/threshold tuning/new probe. E2/E4 numerical HF backend,663 Mixed pool. Not an independent trained model; all jointgates failed.
- [原始汇总：e2-e4-fixed-cpu-actual.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/e2-e4-fixed-cpu-actual.json)

## peer8192-rank/within1

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R255 | cursor256 | EMA | — | 54.70% | 184/512 | 35.94% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- Actual third matched peer ranking fork, +within_rank1 on authentic distinguishable pairs; same256 updates and E1 state as no-rank/inter1/bothrank1. Raw/fixed-train BA absent.
- [原始汇总：peer-within1-final-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/reports/peer-within1-final-independent-audit.json)

## frozen-question-prefix-contribution/E4/legacy

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R256 | frozen source checkpoint | frozen EMA ablation | — | 55.77% | 196/512 | 38.28% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- No training, only fixed input/readout logit decomposition diagnostic. legacy is original BF16 production; shadow_full is FP32 shadow; question/prefix are fixed components, not new learned models. Prefix hidden states already contain question context; this does not isolate semantic information. No threshold fitting.
- [原始汇总：readout-contribution-final-delivery.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/reports/readout-contribution-final-delivery.json)

## frozen-question-prefix-contribution/E4/prefix

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R257 | frozen source checkpoint | frozen EMA ablation | — | 49.94% | 143/512 | 27.93% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- No training, only fixed input/readout logit decomposition diagnostic. legacy is original BF16 production; shadow_full is FP32 shadow; question/prefix are fixed components, not new learned models. Prefix hidden states already contain question context; this does not isolate semantic information. No threshold fitting.
- [原始汇总：readout-contribution-final-delivery.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/reports/readout-contribution-final-delivery.json)

## frozen-question-prefix-contribution/E4/question

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R258 | frozen source checkpoint | frozen EMA ablation | — | 53.73% | 177/512 | 34.57% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- No training, only fixed input/readout logit decomposition diagnostic. legacy is original BF16 production; shadow_full is FP32 shadow; question/prefix are fixed components, not new learned models. Prefix hidden states already contain question context; this does not isolate semantic information. No threshold fitting.
- [原始汇总：readout-contribution-final-delivery.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/reports/readout-contribution-final-delivery.json)

## frozen-question-prefix-contribution/E4/shadow_full

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R259 | frozen source checkpoint | frozen EMA ablation | — | 55.66% | 196/512 | 38.28% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- No training, only fixed input/readout logit decomposition diagnostic. legacy is original BF16 production; shadow_full is FP32 shadow; question/prefix are fixed components, not new learned models. Prefix hidden states already contain question context; this does not isolate semantic information. No threshold fitting.
- [原始汇总：readout-contribution-final-delivery.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/reports/readout-contribution-final-delivery.json)

## frozen-question-prefix-contribution/peer256/legacy

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R260 | frozen source checkpoint | frozen EMA ablation | — | 54.66% | 182/512 | 35.55% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- No training, only fixed input/readout logit decomposition diagnostic. legacy is original BF16 production; shadow_full is FP32 shadow; question/prefix are fixed components, not new learned models. Prefix hidden states already contain question context; this does not isolate semantic information. No threshold fitting.
- [原始汇总：readout-contribution-final-delivery.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/reports/readout-contribution-final-delivery.json)

## frozen-question-prefix-contribution/peer256/prefix

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R261 | frozen source checkpoint | frozen EMA ablation | — | 49.74% | 149/512 | 29.10% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- No training, only fixed input/readout logit decomposition diagnostic. legacy is original BF16 production; shadow_full is FP32 shadow; question/prefix are fixed components, not new learned models. Prefix hidden states already contain question context; this does not isolate semantic information. No threshold fitting.
- [原始汇总：readout-contribution-final-delivery.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/reports/readout-contribution-final-delivery.json)

## frozen-question-prefix-contribution/peer256/question

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R262 | frozen source checkpoint | frozen EMA ablation | — | 52.63% | 176/512 | 34.38% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- No training, only fixed input/readout logit decomposition diagnostic. legacy is original BF16 production; shadow_full is FP32 shadow; question/prefix are fixed components, not new learned models. Prefix hidden states already contain question context; this does not isolate semantic information. No threshold fitting.
- [原始汇总：readout-contribution-final-delivery.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/reports/readout-contribution-final-delivery.json)

## frozen-question-prefix-contribution/peer256/shadow_full

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R263 | frozen source checkpoint | frozen EMA ablation | — | 54.62% | 182/512 | 35.55% | 2048 | — | — |

- 评测池：already viewed fixed-HF2048 ownG5 /663 Mixed
- No training, only fixed input/readout logit decomposition diagnostic. legacy is original BF16 production; shadow_full is FP32 shadow; question/prefix are fixed components, not new learned models. Prefix hidden states already contain question context; this does not isolate semantic information. No threshold fitting.
- [原始汇总：readout-contribution-final-delivery.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/qwen7b-warmup-20261019/reports/readout-contribution-final-delivery.json)

## G20-matched6528-random

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R264 | CP136 | EMA | — | 53.60% | 205/512 | 40.04% | 512 | — | — |
| R265 | CP136 | EMA | — | 55.44% | 201/512 | 39.26% | 2048 | — | — |

- 评测池：diagnostic512
- 评测池：selection2048FreshG5
- EMA only was externally scored; raw checkpoint exists but raw external metrics were not measured. Same sealed E1 Adam10464→10600,6528 questions/B48×136,authenticG20/P20/native feature backend. Four unique authentic G5 draws/question, each weight1/4; random and globally balanced branches matched136 updates. diagnostic512 selects the entire512 pool, not2048→512 ranking.
- EMA only was externally scored; raw checkpoint exists but raw external metrics were not measured. Same sealed E1 Adam10464→10600,6528 questions/B48×136,authenticG20/P20/native feature backend. Four unique authentic G5 draws/question, each weight1/4; random and globally balanced branches matched136 updates.
- [原始汇总：g20-final-native-E4-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/actual-launches/g20-final-native-E4-independent-audit.json)，JSON位置 `/results/random/evaluations/diagnostic512`
- [原始汇总：g20-final-native-E4-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/actual-launches/g20-final-native-E4-independent-audit.json)，JSON位置 `/results/random/evaluations/selection2048FreshG5`

## G20-matched6528-global-weighted-balanced

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R266 | CP136 | EMA | — | 53.43% | 205/512 | 40.04% | 512 | — | — |
| R267 | CP136 | EMA | — | 54.72% | 193/512 | 37.70% | 2048 | — | — |

- 评测池：diagnostic512
- 评测池：selection2048FreshG5
- EMA only was externally scored; raw checkpoint exists but raw external metrics were not measured. Same sealed E1 Adam10464→10600,6528 questions/B48×136,authenticG20/P20/native feature backend. Four unique authentic G5 draws/question, each weight1/4; random and globally balanced branches matched136 updates. diagnostic512 selects the entire512 pool, not2048→512 ranking.
- EMA only was externally scored; raw checkpoint exists but raw external metrics were not measured. Same sealed E1 Adam10464→10600,6528 questions/B48×136,authenticG20/P20/native feature backend. Four unique authentic G5 draws/question, each weight1/4; random and globally balanced branches matched136 updates.
- [原始汇总：g20-final-native-E4-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/actual-launches/g20-final-native-E4-independent-audit.json)，JSON位置 `/results/global-weighted-balanced/evaluations/diagnostic512`
- [原始汇总：g20-final-native-E4-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/actual-launches/g20-final-native-E4-independent-audit.json)，JSON位置 `/results/global-weighted-balanced/evaluations/selection2048FreshG5`

## G20-exact-all15504-DP

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R268 | CP136 | EMA | — | 52.62% | 205/512 | 40.04% | 512 | — | — |
| R269 | CP136 | EMA | — | 55.28% | 197/512 | 38.48% | 2048 | — | — |

- 评测池：diagnostic512
- 评测池：selection2048FreshG5
- EMA only was externally scored; raw checkpoint exists but raw external metrics were not measured. Same sealed E1 Adam10464→10600,6528 questions/B48×136,authenticG20/P20/native feature backend. Exact all15504 G5 subsets streamed DP, per-question weight1; not a synthetic speed result; higher subset compute than random4. diagnostic512 selects the entire512 pool, not2048→512 ranking.
- EMA only was externally scored; raw checkpoint exists but raw external metrics were not measured. Same sealed E1 Adam10464→10600,6528 questions/B48×136,authenticG20/P20/native feature backend. Exact all15504 G5 subsets streamed DP, per-question weight1; not a synthetic speed result; higher subset compute than random4.
- [原始汇总：final-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/exact-G20-DP-qualification/final-independent-audit.json)，JSON位置 `/result/evaluations/diagnostic512`
- [原始汇总：final-independent-audit.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/exact-G20-DP-qualification/final-independent-audit.json)，JSON位置 `/result/evaluations/selection2048FreshG5`

## 8192题条件均衡G5 / L15P20

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R270 | E0 | EMA | 50.00% | 50.00% | 174/512 | 33.98% | 2048 | BCE=0.726487 | BCE=0.762976 |
| R271 | E0 | raw | 50.00% | 50.00% | 174/512 | 33.98% | 2048 | BCE=0.726487 | BCE=0.762976 |
| R272 | E1 | EMA | 50.17% | 49.85% | 176/512 | 34.38% | 2048 | BCE=0.706883 | BCE=0.729239 |
| R273 | E1 | raw | 51.14% | 50.23% | 204/512 | 39.84% | 2048 | BCE=0.673568 | BCE=0.644386 |
| R274 | E2 | EMA | 57.57% | 54.56% | 194/512 | 37.89% | 2048 | BCE=0.687232 | BCE=0.700568 |
| R275 | E2 | raw | 58.30% | 54.75% | 204/512 | 39.84% | 2048 | BCE=0.692777 | BCE=0.746309 |
| R276 | E3 | EMA | 59.70% | 57.10% | 205/512 | 40.04% | 2048 | BCE=0.670080 | BCE=0.684058 |
| R277 | E3 | raw | 63.36% | 58.58% | 200/512 | 39.06% | 2048 | BCE=0.644551 | BCE=0.687674 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- E0为初始评估；E1–3各64次更新。训练池不同于全量直接Mixed的训练A2048。
- [原始汇总：recent-all-pilot-epochs-remote.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/historical-all-experiments/recent-all-pilot-epochs-remote.json)

## 8192题条件均衡G5 / L18P20

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R278 | E0 | EMA | 50.00% | 50.00% | 170/512 | 33.20% | 2048 | BCE=0.724775 | BCE=0.760060 |
| R279 | E0 | raw | 50.00% | 50.00% | 170/512 | 33.20% | 2048 | BCE=0.724775 | BCE=0.760060 |
| R280 | E1 | EMA | 50.01% | 50.35% | 183/512 | 35.74% | 2048 | BCE=0.704615 | BCE=0.726087 |
| R281 | E1 | raw | 51.13% | 50.45% | 203/512 | 39.65% | 2048 | BCE=0.672440 | BCE=0.635205 |
| R282 | E2 | EMA | 57.82% | 56.59% | 198/512 | 38.67% | 2048 | BCE=0.681156 | BCE=0.693011 |
| R283 | E2 | raw | 57.80% | 55.74% | 205/512 | 40.04% | 2048 | BCE=0.691085 | BCE=0.757424 |
| R284 | E3 | EMA | 61.40% | 57.22% | 205/512 | 40.04% | 2048 | BCE=0.662219 | BCE=0.676125 |
| R285 | E3 | raw | 61.88% | 58.65% | 206/512 | 40.23% | 2048 | BCE=0.658723 | BCE=0.727862 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- E0为初始评估；E1–3各64次更新。训练池不同于全量直接Mixed的训练A2048。
- [原始汇总：recent-all-pilot-epochs-remote.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/historical-all-experiments/recent-all-pilot-epochs-remote.json)

## 8192题条件均衡G5 / L21P20

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R286 | E0 | EMA | 50.00% | 50.00% | 161/512 | 31.45% | 2048 | BCE=0.755505 | BCE=0.809350 |
| R287 | E0 | raw | 50.00% | 50.00% | 161/512 | 31.45% | 2048 | BCE=0.755505 | BCE=0.809350 |
| R288 | E1 | EMA | 50.25% | 50.40% | 177/512 | 34.57% | 2048 | BCE=0.716316 | BCE=0.747301 |
| R289 | E1 | raw | 52.63% | 51.32% | 198/512 | 38.67% | 2048 | BCE=0.667904 | BCE=0.645880 |
| R290 | E2 | EMA | 57.05% | 55.12% | 201/512 | 39.26% | 2048 | BCE=0.685247 | BCE=0.705526 |
| R291 | E2 | raw | 59.21% | 55.52% | 210/512 | 41.02% | 2048 | BCE=0.676995 | BCE=0.735285 |
| R292 | E3 | EMA | 61.90% | 56.34% | 207/512 | 40.43% | 2048 | BCE=0.661907 | BCE=0.685009 |
| R293 | E3 | raw | 63.92% | 57.23% | 198/512 | 38.67% | 2048 | BCE=0.644591 | BCE=0.711915 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- E0为初始评估；E1–3各64次更新。训练池不同于全量直接Mixed的训练A2048。
- [原始汇总：recent-all-pilot-epochs-remote.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/historical-all-experiments/recent-all-pilot-epochs-remote.json)

## 8192题条件均衡G5 / L24P20

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R294 | E0 | EMA | 50.00% | 50.00% | 162/512 | 31.64% | 2048 | BCE=0.802561 | BCE=0.879699 |
| R295 | E0 | raw | 50.00% | 50.00% | 162/512 | 31.64% | 2048 | BCE=0.802561 | BCE=0.879699 |
| R296 | E1 | EMA | 50.00% | 50.00% | 177/512 | 34.57% | 2048 | BCE=0.737582 | BCE=0.782847 |
| R297 | E1 | raw | 52.66% | 51.21% | 182/512 | 35.55% | 2048 | BCE=0.671140 | BCE=0.653137 |
| R298 | E2 | EMA | 54.46% | 53.02% | 182/512 | 35.55% | 2048 | BCE=0.695136 | BCE=0.721805 |
| R299 | E2 | raw | 58.95% | 54.25% | 189/512 | 36.91% | 2048 | BCE=0.682249 | BCE=0.736328 |
| R300 | E3 | EMA | 60.88% | 54.19% | 188/512 | 36.72% | 2048 | BCE=0.667702 | BCE=0.693372 |
| R301 | E3 | raw | 63.65% | 54.21% | 187/512 | 36.52% | 2048 | BCE=0.648925 | BCE=0.729479 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- E0为初始评估；E1–3各64次更新。训练池不同于全量直接Mixed的训练A2048。
- [原始汇总：recent-all-pilot-epochs-remote.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/historical-all-experiments/recent-all-pilot-epochs-remote.json)

## 8192题条件均衡G5 / L27P20

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R302 | E0 | EMA | 50.42% | 49.48% | 164/512 | 32.03% | 2048 | BCE=0.737873 | BCE=0.772002 |
| R303 | E0 | raw | 50.42% | 49.48% | 164/512 | 32.03% | 2048 | BCE=0.737873 | BCE=0.772002 |
| R304 | E1 | EMA | 51.08% | 52.06% | 175/512 | 34.18% | 2048 | BCE=0.702257 | BCE=0.712781 |
| R305 | E1 | raw | 50.75% | 50.72% | 181/512 | 35.35% | 2048 | BCE=0.675297 | BCE=0.644315 |
| R306 | E2 | EMA | 56.16% | 53.04% | 181/512 | 35.35% | 2048 | BCE=0.680421 | BCE=0.683403 |
| R307 | E2 | raw | 59.34% | 54.62% | 187/512 | 36.52% | 2048 | BCE=0.679874 | BCE=0.726237 |
| R308 | E3 | EMA | 59.22% | 53.90% | 198/512 | 38.67% | 2048 | BCE=0.662560 | BCE=0.674934 |
| R309 | E3 | raw | 59.01% | 53.88% | 200/512 | 39.06% | 2048 | BCE=0.681651 | BCE=0.776072 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- E0为初始评估；E1–3各64次更新。训练池不同于全量直接Mixed的训练A2048。
- [原始汇总：recent-all-pilot-epochs-remote.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/historical-all-experiments/recent-all-pilot-epochs-remote.json)

## 8192题条件均衡G5 / L21P80

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R310 | E0 | EMA | 50.00% | 50.00% | 168/512 | 32.81% | 2048 | BCE=0.778502 | BCE=0.844597 |
| R311 | E0 | raw | 50.00% | 50.00% | 168/512 | 32.81% | 2048 | BCE=0.778502 | BCE=0.844597 |
| R312 | E1 | EMA | 50.13% | 50.18% | 183/512 | 35.74% | 2048 | BCE=0.724932 | BCE=0.764415 |
| R313 | E1 | raw | 54.46% | 53.66% | 202/512 | 39.45% | 2048 | BCE=0.659688 | BCE=0.640664 |
| R314 | E2 | EMA | 59.21% | 56.02% | 203/512 | 39.65% | 2048 | BCE=0.683680 | BCE=0.712993 |
| R315 | E2 | raw | 62.58% | 58.10% | 219/512 | 42.77% | 2048 | BCE=0.667388 | BCE=0.736454 |
| R316 | E3 | EMA | 64.03% | 58.75% | 214/512 | 41.80% | 2048 | BCE=0.656175 | BCE=0.690154 |
| R317 | E3 | raw | 66.19% | 59.94% | 215/512 | 41.99% | 2048 | BCE=0.623105 | BCE=0.686762 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- E0为初始评估；E1–3各64次更新。训练池不同于全量直接Mixed的训练A2048。
- [原始汇总：recent-all-pilot-epochs-remote.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/historical-all-experiments/recent-all-pilot-epochs-remote.json)

## 8192题条件均衡G5 / L27P80

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R318 | E0 | EMA | 49.70% | 49.98% | 164/512 | 32.03% | 2048 | BCE=0.738297 | BCE=0.764650 |
| R319 | E0 | raw | 49.70% | 49.98% | 164/512 | 32.03% | 2048 | BCE=0.738297 | BCE=0.764650 |
| R320 | E1 | EMA | 50.64% | 50.72% | 179/512 | 34.96% | 2048 | BCE=0.705040 | BCE=0.716012 |
| R321 | E1 | raw | 51.14% | 51.16% | 191/512 | 37.30% | 2048 | BCE=0.671670 | BCE=0.642975 |
| R322 | E2 | EMA | 57.08% | 53.92% | 188/512 | 36.72% | 2048 | BCE=0.679330 | BCE=0.687368 |
| R323 | E2 | raw | 61.76% | 55.60% | 196/512 | 38.28% | 2048 | BCE=0.671557 | BCE=0.723003 |
| R324 | E3 | EMA | 60.65% | 55.95% | 200/512 | 39.06% | 2048 | BCE=0.657220 | BCE=0.675358 |
| R325 | E3 | raw | 66.30% | 57.88% | 197/512 | 38.48% | 2048 | BCE=0.630708 | BCE=0.698552 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- E0为初始评估；E1–3各64次更新。训练池不同于全量直接Mixed的训练A2048。
- [原始汇总：recent-all-pilot-epochs-remote.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/historical-all-experiments/recent-all-pilot-epochs-remote.json)

## 8192题天然随机G5 / layer21-prefix20

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R328 | E0 | EMA | 50.00% | 50.00% | 161/512 | 31.45% | 2048 | BCE=0.755505 | BCE=0.809350 |
| R329 | E0 | raw | 50.00% | 50.00% | 161/512 | 31.45% | 2048 | BCE=0.755505 | BCE=0.809350 |
| R330 | E1 | EMA | 50.22% | 50.22% | 169/512 | 33.01% | 2048 | BCE=0.718717 | BCE=0.750513 |
| R331 | E1 | raw | 51.67% | 50.34% | 189/512 | 36.91% | 2048 | BCE=0.673081 | BCE=0.647189 |
| R332 | E2 | EMA | 56.01% | 53.66% | 188/512 | 36.72% | 2048 | BCE=0.690657 | BCE=0.710407 |
| R333 | E2 | raw | 59.96% | 56.14% | 193/512 | 37.70% | 2048 | BCE=0.672824 | BCE=0.703884 |
| R334 | E3 | EMA | 60.12% | 55.67% | 198/512 | 38.67% | 2048 | BCE=0.668511 | BCE=0.685142 |
| R335 | E3 | raw | 63.37% | 56.50% | 204/512 | 39.84% | 2048 | BCE=0.657142 | BCE=0.711410 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- 与条件均衡G5 pilot共享题目、架构、预算；不同G5抽样。
- [原始汇总：actual-matched-metric-summary-complete.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/natural-g5-online-8192/actual-matched-metric-summary-complete.json)

## 8192题天然随机G5 / layer27-prefix20

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R336 | E0 | EMA | 50.42% | 49.48% | 164/512 | 32.03% | 2048 | BCE=0.737873 | BCE=0.772002 |
| R337 | E0 | raw | 50.42% | 49.48% | 164/512 | 32.03% | 2048 | BCE=0.737873 | BCE=0.772002 |
| R338 | E1 | EMA | 51.23% | 51.52% | 176/512 | 34.38% | 2048 | BCE=0.700209 | BCE=0.708126 |
| R339 | E1 | raw | 50.55% | 50.56% | 182/512 | 35.55% | 2048 | BCE=0.676273 | BCE=0.647950 |
| R340 | E2 | EMA | 56.58% | 53.26% | 175/512 | 34.18% | 2048 | BCE=0.681737 | BCE=0.683364 |
| R341 | E2 | raw | 58.57% | 54.11% | 193/512 | 37.70% | 2048 | BCE=0.679249 | BCE=0.706723 |
| R342 | E3 | EMA | 59.61% | 54.24% | 191/512 | 37.30% | 2048 | BCE=0.668094 | BCE=0.675181 |
| R343 | E3 | raw | 62.18% | 55.08% | 202/512 | 39.45% | 2048 | BCE=0.667410 | BCE=0.719879 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- 与条件均衡G5 pilot共享题目、架构、预算；不同G5抽样。
- [原始汇总：actual-matched-metric-summary-complete.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/natural-g5-online-8192/actual-matched-metric-summary-complete.json)

## packed变量M3–10方差MSE / 8192题

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R344 | E0 | EMA | 50.00% | 50.00% | 175/512 | 34.18% | 2048 | vMSE=0.010707 | vMSE=0.011985 |
| R345 | E0 | raw | 50.00% | 50.00% | 175/512 | 34.18% | 2048 | vMSE=0.010707 | vMSE=0.011985 |
| R346 | E1 | EMA | 50.59% | 50.68% | 188/512 | 36.72% | 2048 | vMSE=0.009146 | vMSE=0.009444 |
| R347 | E1 | raw | 50.00% | 49.96% | 198/512 | 38.67% | 2048 | vMSE=0.009409 | vMSE=0.008538 |
| R348 | E2 | EMA | 55.48% | 53.04% | 201/512 | 39.26% | 2048 | vMSE=0.008880 | vMSE=0.008637 |
| R349 | E2 | raw | 52.42% | 51.05% | 201/512 | 39.26% | 2048 | vMSE=0.009180 | vMSE=0.008488 |
| R350 | E3 | EMA | 53.90% | 51.24% | 206/512 | 40.23% | 2048 | vMSE=0.008913 | vMSE=0.008457 |
| R351 | E3 | raw | 58.15% | 53.19% | 202/512 | 39.45% | 2048 | vMSE=0.008627 | vMSE=0.008414 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- 8192题/3epoch已完成，EMA0.99。回归population方差；BA仅为预设方差阈值0.08的诊断，非Mixed概率阈值0.5；排名按方差。
- [原始汇总：actual-live-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/packed-variance/actual-live-latest.json)

## packed变量M3–10方差 / 全169028 / mse

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R352 | E0 | EMA | 50.00% | 50.00% | 175/512 | 34.18% | 2048 | vMSE=0.010707 | vMSE=0.011985 |
| R353 | E0 | raw | 50.00% | 50.00% | 175/512 | 34.18% | 2048 | vMSE=0.010707 | vMSE=0.011985 |
| R354 | E1 | EMA | 56.75% | 55.91% | 211/512 | 41.21% | 2048 | vMSE=0.008836 | vMSE=0.008625 |
| R355 | E1 | raw | 56.26% | 52.79% | 219/512 | 42.77% | 2048 | vMSE=0.008852 | vMSE=0.008313 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- 全169028题/10epoch在跑，EMA0.9995。回归population方差；BA仅为预设方差阈值0.08的诊断，非Mixed概率阈值0.5；排名按方差。
- [原始汇总：actual-live-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/packed-variance-full169028/actual-live-latest.json)

## packed变量M3–10方差 / 全169028 / mae

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R356 | E0 | EMA | 50.00% | 50.00% | 175/512 | 34.18% | 2048 | vMSE=0.010707 | vMSE=0.011985 |
| R357 | E0 | raw | 50.00% | 50.00% | 175/512 | 34.18% | 2048 | vMSE=0.010707 | vMSE=0.011985 |
| R358 | E1 | EMA | 50.00% | 50.00% | 187/512 | 36.52% | 2048 | vMSE=0.015128 | vMSE=0.012193 |
| R359 | E1 | raw | 50.00% | 50.00% | 138/512 | 26.95% | 2048 | vMSE=0.015463 | vMSE=0.012453 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- 全169028题/10epoch在跑，EMA0.9995。回归population方差；BA仅为预设方差阈值0.08的诊断，非Mixed概率阈值0.5；排名按方差。
- [原始汇总：actual-live-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/packed-variance-full169028/actual-live-latest.json)

## packed变量M3–10方差 / 全169028 / z_mse_anchor

| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| R360 | E0 | EMA | 50.00% | 50.00% | 175/512 | 34.18% | 2048 | vMSE=0.010707 | vMSE=0.011985 |
| R361 | E0 | raw | 50.00% | 50.00% | 175/512 | 34.18% | 2048 | vMSE=0.010707 | vMSE=0.011985 |
| R362 | E1 | EMA | 57.38% | 57.27% | 215/512 | 41.99% | 2048 | vMSE=0.008795 | vMSE=0.008832 |
| R363 | E1 | raw | 56.99% | 53.78% | 218/512 | 42.58% | 2048 | vMSE=0.008756 | vMSE=0.008289 |

- 评测池：当前独立HF固定G5 / 2048题 / 663 Mixed
- 全169028题/10epoch在跑，EMA0.9995。回归population方差；BA仅为预设方差阈值0.08的诊断，非Mixed概率阈值0.5；排名按方差。
- [原始汇总：actual-live-latest.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/packed-variance-full169028/actual-live-latest.json)

## 未完成、未评分或没有保留该指标的记录

这些记录保留以免把“提交/准备/局部进度”误写成已完成实验。

| 编号 | 实验 | checkpoint | 权重 | 原始状态 |
|---|---|---|---|---|
| R002 | 7B native L26→block27 Prefix5 trajectoryBCE+MixedBCE warmup | epoch-0001.pt | raw | RAW_METRICS_MISSING_IN_EXISTING_LOCAL_REPORTS |
| R004 | 7B native L26→block27 Prefix5 trajectoryBCE+MixedBCE warmup | epoch-0002.pt | raw | RAW_METRICS_MISSING_IN_EXISTING_LOCAL_REPORTS |
| R006 | 7B native L26→block27 Prefix5 trajectoryBCE+MixedBCE warmup | epoch-0003.pt | raw | RAW_METRICS_MISSING_IN_EXISTING_LOCAL_REPORTS |
| R008 | 7B native L26→block27 Prefix5 trajectoryBCE+MixedBCE warmup | epoch-0004.pt | raw | RAW_METRICS_MISSING_IN_EXISTING_LOCAL_REPORTS |
| R009 | 7B native Prefix5 warmup E5 preserved tail ONLY | inflight.pt | EMA0.99 | INCOMPLETE_E5_ONLY4_UPDATES_NO_COMPLETE_EPOCH_METRICS |
| R010 | 7B native Prefix5 warmup E5 preserved tail ONLY | inflight.pt | raw | RAW_METRICS_MISSING_IN_EXISTING_LOCAL_REPORTS |
| R012 | Mixed-only continuation from warmupE2, Prefix20, rank weight0.1 | epoch-0001.pt | raw | RAW_METRICS_MISSING_IN_EXISTING_LOCAL_REPORTS |
| R014 | Prefix20 1:1 Mixed/nonMixed continuation, rank weight1.0 | epoch-0002.pt | raw | RAW_METRICS_MISSING_IN_EXISTING_LOCAL_REPORTS |
| R016 | Prefix20 1:1 Mixed/nonMixed continuation, rank weight1.0 | epoch-0003.pt | raw | RAW_METRICS_MISSING_IN_EXISTING_LOCAL_REPORTS |
| R018 | Prefix20 1:1 Mixed/nonMixed continuation, rank weight1.0 | epoch-0004.pt | raw | RAW_METRICS_MISSING_IN_EXISTING_LOCAL_REPORTS |
| R020 | Prefix20 1:1 Mixed/nonMixed continuation, rank weight1.0 | epoch-0005.pt | raw | RAW_METRICS_MISSING_IN_EXISTING_LOCAL_REPORTS |
| R021 | Original Prefix5 genuineMixed-only + auxiliary ranking from warmupE2 | 未产生 | unknown | NO_LOCAL_ACCEPTED_PREFIX5_FINETUNE_CHECKPOINT_METRICS_FOUND |
| R022 | Original Prefix5 genuineMixed-only + auxiliary ranking from warmupE2 | 未产生 | unknown | NO_LOCAL_ACCEPTED_PREFIX5_FINETUNE_CHECKPOINT_METRICS_FOUND |
| R023 | Original Prefix5 genuineMixed-only + auxiliary ranking from warmupE2 | 未产生 | unknown | NO_LOCAL_ACCEPTED_PREFIX5_FINETUNE_CHECKPOINT_METRICS_FOUND |
| R024 | Uniform ownG5 class-weighted fork from direct E1 (later metadata only) | 未产生 | unknown | METRICS_MISSING_LOCAL_STATUS_ONLY |
| R087 | joint124194-native-block27-plus-readout | newE1 | raw | missing_posttraining_evaluation |
| R088 | joint124194-native-block27-plus-readout | newE1 | EMA | missing_posttraining_evaluation |
| R089 | joint124194-native-block27-plus-readout | newE2 | raw | missing_posttraining_evaluation |
| R090 | joint124194-native-block27-plus-readout | newE2 | EMA | missing_posttraining_evaluation |
| R091 | joint124194-native-block27-plus-readout | newE3 | raw | missing_posttraining_evaluation |
| R092 | joint124194-native-block27-plus-readout | newE3 | EMA | missing_posttraining_evaluation |
| R093 | joint124194-native-block27-plus-readout | newE4 | raw | missing_posttraining_evaluation |
| R094 | joint124194-native-block27-plus-readout | newE4 | EMA | missing_posttraining_evaluation |
| R095 | joint-native-real-seven-update-qualification | CP7 | raw | real_training_qualification_no_external_metrics |
| R096 | joint-native-real-seven-update-qualification | CP7 | EMA | real_training_qualification_no_external_metrics |
| R097 | joint-native-full169028-prepare-only | not_started | raw | prepared_only_no_result |
| R098 | joint-native-full169028-prepare-only | not_started | EMA | prepared_only_no_result |

# 已闭合消融评测总表

`release/recursive_bran_a/ablation150_infra/collect_closed_comparison.py` 在本地合并已接受的小型报告 capsule，复用原 `ablation150_reports_v1/report.py` 输出 JSON、CSV 和中文汇总表。它不连接节点、不加载模型、不触发推理，也不改训练。

```bash
python3 -B -m release.recursive_bran_a.ablation150_infra.collect_closed_comparison \
  --capsule /absolute/path/closed-export.json \
  --output /absolute/path/new-report-directory
```

只接收冻结方案的11个标签、5/50/100/150节点及原始 aggregate 字节。合并前核对 capsule 每个成员大小和 SHA、正式512计数、同轮次A的汇总EM及复用关系。输出目录必须全新，既有快照和原始报告保持原样。

`accepted_label_points` 的完整目标为44，其中复用标签不是新增独立训练。缺失节点明确列出；即使44个汇总节点全部齐全，`goal_complete` 仍为 false，训练终态与存档须另行验收。

原报告的成本覆盖范围原样保留：boundary-only 或不完整前缀不能称全量耗时/生成量；阶段计时可能重叠，GPU小时是记录的 wall×卡数。A完整前缀成本和C1故障前已复用采样的补充证据保存在各自已发布报告中，不覆盖冻结的原报告。只比较汇总EM，不追加逐题评测、配对统计或bootstrap。

`plot_aggregate_comparison.py` 从已提交的总表读取 `aggregate.json` 和 `curve.csv`，生成整体、NQ、HotpotQA 的 PNG/PDF 曲线及输入输出 SHA 来源记录：

```bash
python3 -B release/recursive_bran_a/ablation150_infra/plot_aggregate_comparison.py \
  --report /absolute/path/committed-comparison-report \
  --output /absolute/path/new-figure-directory
```

只画实际已接受的评测点；复用标签明确列出，不重复画成独立训练。连接线不代表中间轮次已经评测，缺失节点保持缺失。绘图仅在本地运行，不连接节点、不读权重或原始轨迹、不新增训练或推理；图表不代替整个 goal 的训练终态与存档验收。

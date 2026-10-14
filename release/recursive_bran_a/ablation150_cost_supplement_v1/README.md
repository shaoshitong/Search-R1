# C1 原首轮采样成本补充器

此工具仅组合已闭合的小型 JSON，输出独立 `summary.json` 和短中文说明；不读取权重、原始轨迹或模型，不修改 b95 报告和训练。适用于 C1-root-only 及其 generation-1 复用，其他组返回 `NO_FIX_REQUIRED`。

```bash
python3 -B -m release.recursive_bran_a.ablation150_cost_supplement_v1.supplement \
  --aggregate CLOSED_B95_MILESTONE/aggregate.json \
  --original-cost original-outer1-recorded-sampling-cost.json \
  --prefix-scalars closed-prefix-native-scalars.json \
  --destination cost-supplement-C1-outer-005
```

第三个输入使用 `ablation150_closed_prefix_native_scalar_v1`：`run_id`、`config_sha256`、`scientific_source_commit`、`configuration_receipt_sha256`、`recovered_outer1:true`、`completed_prefix_outer:5/50/100/150`、完整 `outer1_metrics`、`prefix_metrics:{"1":metrics,...}` 和 `native_line_SHA_by_outer:{"1":sha,...}`。捕获端须核实真实原生闭合和身份，保留所有标量字段，不能先删除生成字段再宣称首轮缺失。

工具核对同一冻结配置、连续已闭合 prefix、native 行 SHA、首轮明确复用／计时排除标记以及 raw/generated 字段确实不存在。其余轮优先读取原生 `raw_emitted_tokens`，兼容回退到 `generated_tokens`；不要求两种原字段数值相等。既有 prefix token 和 wall 必须与 b95 原值一致，再补原首轮物理 834,605 emitted token 一次。不存在字段与字段为 0 不等价；字段存在时拒绝补加。

输入不足、冲突、另一缺口或重复补加返回 `INCOMPLETE`，总量为 null。已存在且完全相同的补充输出可原样复用，不能把本工具输出再次作为 b95 输入。公开输出仅聚合和输入 SHA，不包含私有路径、inode、原题或 token IDs。

计时不补成“完整 GPU 成本”：原 native wall/gen 数值保持不变，401.640 秒首 RPC 至 compose 窗口单独列出；该窗口缺少首 RPC 前准备。共享行动、RPC、HTTP、compose 有包含或重叠关系。精确 gen、GPU 活跃时间与资源分配计费时间仍未知。完整 emitted prefix 不表示完整端到端计时，也不新增训练或评测。

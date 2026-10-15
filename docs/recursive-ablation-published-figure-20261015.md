# 从已发布汇总生成科学曲线

`release/recursive_bran_a/ablation150_infra/plot_closed_report.py`仅在Mac读取已经提交的三个小汇总：`summary.json`、`curve.csv`、`native-scalar-history.csv`。不连接Node、不读取原始预测/轨迹/权重，不增加训练、评测或调试采样。使用已有matplotlib；不要为运行它安装新环境。

源码先提交、正常推送并回读；之后在新的独立目录生成PNG、PDF及输入/输出SHA来源摘要。运行时核对绘图源码及汇总的Git字节。示例：

```bash
python3 -B release/recursive_bran_a/ablation150_infra/plot_closed_report.py \
  --report reports/recursive-ablation-C1-outer050-20261015 \
  --output reports/recursive-ablation-C1-outer050-figures-20261015
```

图只显示真实5/50/100/150节点；连线不是中间EM测量。PPO loss来自原生记录，不是独立重算或固定验证指标。gen包括原生成/整理计时；native step含关键节点的保存和评测；rolling在step之外。缺失计时留空，原C1首轮故障前采样不填入native时钟；完整成本边界以同轮报告为准。整套goal完成需所有计划条件，绘图完成不代表实验完成。后续条件可复用相同CSV列格式。

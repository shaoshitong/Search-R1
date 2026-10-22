# Current development decision

现有信号仍在**已浏览的 NQ 开发集**：MSE-M5 E7/raw 两seed为278/512与277/512，均值54.20%；原E1 EMA为205/512（40.04%）。这尚非最终4096确认，也不是七集结论。

CURRENT64与WINDOW2018在完全相同E10/raw、每seed相同采样计划及1280次global64批次尝试下，窗口的两seed平均Top512仅提高2.2461个百分点，配对题目95%区间[0.0000,+4.6875]个百分点触及零；单seed区间均跨零，不能称稳健确认提升。BA诊断反而平均下降7.40个百分点。

## 固定训练A：原E1历史诊断

A为2048题、1082 Mixed；仅原E1有这份指标。后续模型未在A评估。不能把A的训练结果与B直接作前后改进对照。

| checkpoint | 固定诊断loss | Mixed BA | Mixed召回 | 全错误报 | 全对误报 | Top512 Mixed |
|---|---:|---:|---:|---:|---:|---:|
| E1 baseline / reference / E1 EMA | Mixed BCE 0.671798 | 58.99% | 62.29% | 40.00% | 56.91% | 344/512 (67.19%) |

## 固定训练B：后续模型一致诊断

B及所有8192题pilot均为NQ。B为2048题、888 Mixed、950全错、210全对，固定ownG5；原E1没有这份已保存评估，空缺保持为空。TopK固定池loss没有计算/保存，不能拿在线batch loss补上。MSE/MAE数值使用原方差v单位，实际训练输出回归4v；Z行的固定诊断MSE不等于其标准化训练总loss。

| checkpoint | 固定诊断loss | Mixed BA | Mixed召回 | 全错误报 | 全对误报 | Top512 Mixed |
|---|---:|---:|---:|---:|---:|---:|
| E1 baseline / reference / E1 EMA | — | — | — | — | — | — |
| CURRENT64 / seed1 / E5 raw | 缺失（未计算） | 56.35% | 14.86% | 2.42% | 0.95% | 384/512 (75.00%) |
| CURRENT64 / seed2 / E5 raw | 缺失（未计算） | 64.78% | 37.50% | 8.95% | 3.33% | 397/512 (77.54%) |
| MSE-M5 / seed1 / E7 raw | variance MSE(v) 0.007022 | 69.89% | 61.60% | 22.95% | 16.67% | 379/512 (74.02%) |
| MSE-M5 / seed2 / E7 raw | variance MSE(v) 0.006932 | 71.49% | 70.05% | 29.58% | 15.71% | 371/512 (72.46%) |
| CURRENT64 / seed1 / E10 raw | 缺失（未计算） | 67.65% | 36.94% | 1.89% | 0.48% | 449/512 (87.70%) |
| CURRENT64 / seed2 / E10 raw | 缺失（未计算） | 75.94% | 69.03% | 17.89% | 13.81% | 449/512 (87.70%) |
| WINDOW2018 / seed1 / E10 raw | 缺失（未计算） | 55.25% | 97.75% | 90.42% | 72.86% | 415/512 (81.05%) |
| WINDOW2018 / seed2 / E10 raw | 缺失（未计算） | 50.73% | 100.00% | 98.42% | 99.05% | 419/512 (81.84%) |
| fullP20 mse / seed1 / E10 raw | variance MSE(v) 0.004808 | 79.03% | 63.40% | 4.74% | 8.10% | 480/512 (93.75%) |
| fullP20 mse / seed1 / E10 EMA | variance MSE(v) 0.005290 | 77.59% | 62.16% | 5.79% | 12.38% | 477/512 (93.16%) |
| fullP20 mae / seed1 / E10 raw | variance MAE(v) 0.070400 | 57.67% | 16.55% | 0.63% | 3.81% | 339/512 (66.21%) |
| fullP20 mae / seed1 / E10 EMA | variance MAE(v) 0.071838 | 56.86% | 16.22% | 1.58% | 6.67% | 325/512 (63.48%) |
| fullP20 z_mse_anchor / seed1 / E10 raw | variance MSE(v) 0.004169 | 82.17% | 67.00% | 2.11% | 5.24% | 500/512 (97.66%) |
| fullP20 z_mse_anchor / seed1 / E10 EMA | variance MSE(v) 0.004396 | 81.74% | 66.33% | 2.00% | 6.67% | 500/512 (97.66%) |

## 已浏览开发集：同一2048题及固定ownG5

开发池为663 Mixed、1055全错、330全对，全部NQ；所选轨迹标签已独立通过actual ownEM/seal核验。Top512按预测分数排序，不按strictEM或分类阈值选题。TopK的0.5阈值只作BA诊断；方差模型使用预设v≥0.08。

| checkpoint | 固定诊断loss | Mixed BA | Mixed召回 | 全错误报 | 全对误报 | Top512 Mixed |
|---|---:|---:|---:|---:|---:|---:|
| E1 baseline / reference / E1 EMA | Mixed BCE 0.653016 | 52.88% | 21.87% | 13.93% | 23.03% | 205/512 (40.04%) |
| CURRENT64 / seed1 / E5 raw | 缺失（未计算） | 53.71% | 11.46% | 5.12% | 0.61% | 266/512 (51.95%) |
| CURRENT64 / seed2 / E5 raw | 缺失（未计算） | 59.67% | 31.98% | 14.98% | 5.15% | 267/512 (52.15%) |
| MSE-M5 / seed1 / E7 raw | variance MSE(v) 0.007594 | 63.51% | 54.45% | 30.43% | 17.88% | 278/512 (54.30%) |
| MSE-M5 / seed2 / E7 raw | variance MSE(v) 0.007660 | 65.38% | 64.25% | 37.44% | 20.91% | 277/512 (54.10%) |
| CURRENT64 / seed1 / E10 raw | 缺失（未计算） | 57.67% | 24.28% | 9.57% | 6.97% | 257/512 (50.20%) |
| CURRENT64 / seed2 / E10 raw | 缺失（未计算） | 62.48% | 55.51% | 32.70% | 23.64% | 262/512 (51.17%) |
| WINDOW2018 / seed1 / E10 raw | 缺失（未计算） | 54.84% | 96.83% | 91.94% | 71.82% | 269/512 (52.54%) |
| WINDOW2018 / seed2 / E10 raw | 缺失（未计算） | 50.51% | 100.00% | 99.24% | 98.18% | 273/512 (53.32%) |
| fullP20 mse / seed1 / E10 raw | variance MSE(v) 0.009399 | 57.23% | 39.37% | 22.46% | 32.73% | 222/512 (43.36%) |
| fullP20 mse / seed1 / E10 EMA | variance MSE(v) 0.009031 | 57.52% | 39.52% | 22.75% | 30.00% | 221/512 (43.16%) |
| fullP20 mae / seed1 / E10 raw | variance MAE(v) 0.062502 | 52.82% | 10.56% | 4.55% | 6.06% | 222/512 (43.36%) |
| fullP20 mae / seed1 / E10 EMA | variance MAE(v) 0.062024 | 53.39% | 11.76% | 4.08% | 7.88% | 219/512 (42.77%) |
| fullP20 z_mse_anchor / seed1 / E10 raw | variance MSE(v) 0.009570 | 59.30% | 39.82% | 19.91% | 25.45% | 245/512 (47.85%) |
| fullP20 z_mse_anchor / seed1 / E10 EMA | variance MSE(v) 0.009245 | 58.55% | 40.57% | 21.80% | 28.79% | 235/512 (45.90%) |

## 当前行动与边界

- 领先探索信号为MSE-M5 E7/raw两seed均约54%，但MSE与CURRENT64的已选checkpoint比较还混合epoch/global batch/更新预算差异，不是单独损失因果比较。
- WINDOW2018应按同E10比较，不能用CURRENT64的E5替代对照，也不能把触零区间说成已确认收益。
- fullP20三臂E10的raw/EMA全部保留。full Z第二seed仍按单独状态记录，不拿进行中的epoch替代第一seedE10。
- fullP80物理源输入仍以实际生产验收状态为准；本报告生成时的qualifier状态见下，不把source-stage当训练ready。训练只由root取得actual ready后唯一提交。
- 最终4096仍未读取，没有新推理、模型加载、CP读取、阈值拟合或Slurm操作。完整JSON保留全部原指标、AUC缺项、来源与配对区间；未保存的指标明确为null。

Qualifier snapshot: `WAITING_FRESH_FULL_P20_ALIGNMENT_NO_TRAINING`, timestamp `1792700575.78527`, fixture complete=`True`, accepted=`True`。

[完整指标、来源及比较区间](CURRENT-DEVELOPMENT-DECISION.json)

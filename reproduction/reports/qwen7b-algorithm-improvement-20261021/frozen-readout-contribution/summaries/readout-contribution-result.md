冻结读出诊断已完成，Slurm743513 COMPLETED0:0，用时2:28。E4与原peerCP256分别完整处理同一新HF固定2048题×5条；没有参数、Adam或actor更新，没有读取final4096。

两个模型全部10240个生产p及2048个生产q均逐值复现封存预测，阈值分类0变化、Top512身份重合512。Probe为block27后masked293×3584展平线性头；question项含原bias，prefix项不含bias。prefix隐藏状态仍含question上下文，因此该分解并非去掉question输入的实验。

| 冻结模型 | question平均绝对logit占比 | question全局方差贡献 | 同题question方差 | prefix组内AUC | corr(生产q, question-only q) | corr(生产q, prefix-only q) |
|---|---:|---:|---:|---:|---:|---:|

| E4 | 64.71% | 63.42% | 0 | 0.53325 | 0.56697 | -0.33890 |

| peer256 | 70.58% | 55.11% | 0 | 0.53576 | 0.74265 | -0.52049 |

| 模型／固定读出诊断 | BA | Mixed recall | nonMixed recall | AW FPR | AC FPR | AUC | Mixed@512 |
|---|---:|---:|---:|---:|---:|---:|---:|

| E4/legacy | 55.771% | 67.572% | 43.971% | 51.185% | 71.515% | 0.57648 | 196/512 |

| E4/question | 53.734% | 44.796% | 62.671% | 30.237% | 60.000% | 0.55120 | 177/512 |

| E4/prefix | 49.942% | 83.710% | 16.173% | 89.384% | 66.061% | 0.46986 | 143/512 |

| peer256/legacy | 54.655% | 67.722% | 41.588% | 50.237% | 84.545% | 0.55990 | 182/512 |

| peer256/question | 52.628% | 36.953% | 68.303% | 25.024% | 53.030% | 0.54822 | 176/512 |

| peer256/prefix | 49.744% | 93.062% | 6.426% | 95.829% | 86.364% | 0.47340 | 149/512 |

判断：question静态项贡献较大，尤其peer的生产Mixed评分与question-only评分相关达0.743。同题五条的question项完全相同，FP32影子中的组内变化来自prefix（另有已单列的舍入误差）；但prefix区分同题正确/错误轨迹的AUC仅约0.53，prefix-only q与真实Mixed的相关系数仅−0.038/−0.043。prefix仍有小幅增量：完整E4比question-only多19道Mixed，peer多6道，不能说prefix完全无信息。question-only/prefix-only均为不拟合、不校准的冻结读出诊断，不是晋级候选。

E4：FP32 full−(question+prefix)重构绝对误差max=2.9802322e-06、mean=5.1134859e-07；FP32影子与原BF16 legacy差异mean=0.0056111593、max=0.028089523，没有声称bitexact。

peer256：FP32 full−(question+prefix)重构绝对误差max=3.6209822e-06、mean=1.4986117e-06；FP32影子与原BF16 legacy差异mean=0.0027240394、max=0.017491341，没有声称bitexact。

各项logit的题内／题间方差、轨迹EM／Mixed相关、正确−错误差、全部TPR/TNR/FPR/AUC以及独立复算结果在同目录readout-contribution-final-delivery.json。所有逐题logit与张量只存远端；本地只有小汇总。两模型训练来源与Adam步数不同，本诊断不把模型间差异解释为某次补训的因果效应。

独立审计：/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-readout-question-prefix-diagnostic-v1/independent-audit.json SHA256 f2bbc12da224f800dc6f670ab47079c82fde51e3b25f1c1603975f289f9ceff8

贡献比例来自实际激活与冻结权重的logit分解，不用权重范数代替贡献；全局方差份额将协方差各分一半。同题question项为常数，因此FP32 full的组内排序与prefix完全一致；不同分量的sigmoid概率差会受共同偏置和饱和影响，不直接用概率差大小跨分量排名。

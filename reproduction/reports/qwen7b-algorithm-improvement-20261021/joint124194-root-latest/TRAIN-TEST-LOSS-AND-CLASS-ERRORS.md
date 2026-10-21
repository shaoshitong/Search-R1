当前124194原生block27+readout预训练指标（实际curve快照至update1359）

本轮E0=历史balanced-prefix20 E4（Adam14418），不是随机未训练模型。当前new epoch1尚未结束（每epoch3882更新），new epoch2/3/4未开始。E0旧模型、当前online raw训练batch和未来完整epoch测试不能混淆。

优化目标总loss=trajectory BCE+binary Mixed BCE（1:1）；event/rank只是日志诊断，优化权重0。训练为50/50 Mixed/nonMixed权重，独立测试保持真实自然类别比例。训练统计是变化中的online preupdate batch，不是固定模型在整个训练集上的重新评分。

|阶段|轨迹BCE|Mixed BCE|总loss|Mixed均衡准确率|
|---|---:|---:|---:|---:|
|epoch1目前1357条有效curve更新（3–1359）|0.797294|0.950886|1.748179|54.6881%|
|epoch1前500条更新（3–502）|0.818437|0.957136|1.775573|55.2844%|
|epoch1最近500条更新（860–1359）|0.784771|0.944819|1.729589|54.3562%|
|E0 fixed2048 raw|0.689791|0.985798|1.675588|55.2138%|
|E0 fixed2048 EMA|0.671472|0.985551|1.657022|55.7714%|
|new epoch1–4独立测试|待完成|待完成|待完成|待完成|

训练窗口：Mixed召回64.2781%→66.4438%，nonMixed召回46.2906%→42.2688%，所以虽然loss小降，当前online BalancedAcc未提高；需独立snapshot/完整epoch评估确定泛化误判变化，不能把不同batch变化等同固定测试集变化。

E0 EMA fixed2048三真实类×binary .5决策：
- 全错1055题：540误判Mixed/515正确nonMixed，假Mixed率51.1848%；该类MixedBCE1.061605。
- 真Mixed663题：448正确Mixed/215漏报nonMixed，漏报率32.4284%；该类MixedBCE0.617518。
- 全对330题：236误判Mixed/94正确nonMixed，假Mixed率71.5152%；该类MixedBCE1.481819。
- 991次总错误中，非Mixed→Mixed=776（78.3047%），Mixed→nonMixed=215（21.6953%）。全对单类误判率最高；全错因为样本多贡献540次错误，绝对数最多。
- Top512真实组成：209全错+196Mixed+107全对，真实Mixed38.28125%。

上述三类误判是本轮起点，尚无新增epoch独立结果，不能写成“训练后全对误报已增/已降”。当前系统主判别为Mixed/nonMixed，并不自动把漏报Mixed细分成预测全对/全错。固定2048用于选型，最终4096未读取。当前checkpoint独立只读评分正在准备，不热改main。

完整938批训练先验的两固定decision诊断已正常完成，5.50秒，约130MiB RSS，2CPU，无GPU、模型/feature读取、无新评分。训练prior具体值与K计数只在远端private receipt，本地不导出。938实际factory/source、各batch_epochs、30,016题真实ownID和120,064个G5 view已绑定。

|模型|原0.5 BA|训练prior decision BA|prior Mixed召回|prior非Mixed召回|原q AUC|原Top512 Mixed|
|---|---:|---:|---:|---:|---:|---:|
|E2|56.25%|55.00%|87.33%|22.67%|0.5691|181|
|E4|55.77%|55.26%|82.65%|27.87%|0.5765|196|
|control256|49.89%|50.41%|98.94%|1.88%|0.5262|171|
|control512|49.96%|53.69%|68.33%|39.06%|0.5439|177|
|control768|50.00%|53.89%|58.97%|48.81%|0.5501|182|

control768的零Mixed召回确有decision点因素：换成训练prior规则后召回58.97%，但非Mixed召回48.81%，BA53.89%仍低于E4原规则55.77%和E2原规则56.25%；AUC/Top完全不变。control256几乎全判Mixed、非Mixed召回仅1.88%，显示改decision不能解决弱区分。E2/E4的新规则BA反而下降。

这只解释分类阈值的一部分现象，不是新模型提升、不改变primary0.5或goal gate、不扫阈值/用外部标签拟合，不读取final4096。五组原0.5 BA/AUC/Top均与已有同源验收聚合逐项吻合。

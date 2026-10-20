L27 readout六项均完成128更新并正常退出。**没有任何一项同时提高selection2048的Mixed balanced accuracy与Mixed@512**。全部Mixed@512低于E1/E2的182题及三事件基底195题，暂不接受为改进。

| 方案 | 512诊断BA | 2048 BA | 2048 AUC | Mixed@512 | Top512 K0/1/2/3/4/5 |
|---|---:|---:|---:|---:|---|
|三事件基底，继承旧head|52.8641%|53.8157%|0.569742|195/512 = 38.0859%|187/45/35/42/73/130|
|原E1|51.3864%|50.3138%|0.533673|182/512 = 35.5469%|249/44/33/49/56/81|
|balanced E2|53.5235%|56.3608%|0.578071|182/512 = 35.5469%|217/42/33/36/71/113|
|last / prefix|50.5021%|47.9255%|0.481770|153/512 = 29.8828%|267/39/31/40/43/92|
|mean / prefix|52.7004%|50.6158%|0.497409|152/512 = 29.6875%|253/33/30/31/58/107|
|mean / question+prefix|50.0000%|50.0000%|0.490660|154/512 = 30.0781%|252/39/27/34/54/106|
|attention / prefix|52.7004%|50.5092%|0.497427|153/512 = 29.8828%|253/34/30/32/57/106|
|attention / question+prefix|50.0000%|50.0000%|0.490755|155/512 = 30.2734%|251/39/27/35/54/106|
|fresh flatten head|50.3289%|50.4924%|0.509409|159/512 = 31.0547%|246/40/29/37/53/107|

六项均从相同E1 block raw/EMA、block Adam10464起步，使用相同6528题、128更新、B51和三事件目标。新readout/head Adam从0开始，最后block没有重置。这是独立readout分支，不能称整模型无损resume。flat control也更换head，三事件基底继承旧head。E2训练更长，不能作等预算因果比较。

mean和attention的question+prefix方案在两个池都把全部问题判为Mixed：Mixed召回100%、nonMixed召回0%、BA50%。其余readout也没有提高排名指标。当前预算下，本轮readout替换未通过双指标门槛；不能外推所有结构或训练预算均无效。

独立验收覆盖：fresh Slurm COMPLETED/0:0、六项连续128步、block10592/head128、初态10464/0、实际首更新双重放通过；十二份readout预测文件SHA、qid/trajectory ID/自身strict EM、G5公式q、固定阈值0.5的BA/两类召回/AUC、按q与qid确定排序的Top512和K0..5。E1/E2与三事件基底也fresh重算，九组question/trajectory/EM身份摘要全部相同。

仅读取已被查看的512诊断及2048 recipe-selection validation，没有读取untouched final、拟合阈值或校准器、申请GPU、重提任务或远端写入。checkpoint本次只核对完成回执和文件字节数，未重载模型tensor或重新计算checkpoint内容SHA；运行器自身的保存重载验收与本次预测审计明确区分。

完整小审计：`independent-l27-own-em-metric-audit.json`；独立CPU源码：`audit_l27_metrics.py`。

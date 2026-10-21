截至本次实际查询：主训练745501仍由原PID300719/birth116543024推进，新增更新1217/15528；首epoch 1217/3882，完整CP1152。无failure/interrupted。训练固定124194题，native block27+readout共同更新、LR5e-6、真实G10随机G5视图Mixed/nonMixed50/50，actor未更新。首个新增epoch尚未完成，未有本轮独立效果提升。

全量169028 canonical题真实raw G10已实际验收，共1690280条；缺失0/多余0，own trajectory IDs唯一，old5身份/EM与canonical精确一致，独立6656题qid/text hash无交集。域NQ78871+Hotpot90157；6组原始重复文本各2题按真实qid保留。两个原tail生产器745518/745519已COMPLETED0:0并释放4GPU。

同协议L26/P20特征已完整验收133538题（124194+9344）。其余35490题（donor14080、next14592、final6818）继续独立采集/验收；donor与14592已有两次真实增长，final6818实际raw通过后已交接prepare，不称全量feature-ready。当前fit不扩容、不热改。

效果的已知对照仍是本轮训练前E0 EMA：独立2048原池Mixed32.373%，选512中196题真实Mixed=38.28125%，MixedBA55.7714%；其中209全错/107全对，错误偏好全对仍需改善。训练batch loss不代替独立MixedBA/真实Top512收益。每epoch后运行固定512/2048 raw与EMA对比；最终4096结果继续保留未读。

剩余：完成当前4epoch；补完35490题特征并实际启动全169028独立联合预训练数据scaling；比较独立Mixed均衡准确率/真实Mixed@512，最终独立4096确认；完整恢复状态验收与资源/源码收尾。全169028 worker与公平matched-update评估入口已准备，bank join7项必要CPU测试通过，尚未提交全量训练GPU。

# Question＋prefix5 探针 batch 优化与完整状态续接（2026-10-16）

用户要求增大batch以缩短原生Qwen最后一个block＋预测头的拟合时间。本次只调整计算分块，保留全部合并数据、同事件G5、联合BCE权重1:1、每个完整epoch的8700次Adam、EMA0.99和200个完整epoch。

| 执行参数 | 原执行 | 优化执行 |
| --- | ---: | ---: |
| 每卡训练micro G5数 | 6 | 13 |
| 每卡训练前向轨迹数 | 30 | 至多65 |
| 每卡评测micro G5数 | 6 | 64 |
| 每卡评测前向轨迹数 | 30 | 320 |
| 全局每次Adam G5 pattern | 51/51/51/51/52 | 相同 |
| 每完整epoch实际Adam／EMA | 8700 | 8700 |

原每卡每次Adam分配12或13个完整G5，却拆为2或3次前后向；micro13可一次完成。原全量EMA评测每卡约18560次小前向，micro64可降到约1740次。调用次数减少不等于总耗时同比下降：总样本量和计算量没有取消，实际吞吐需在新执行中测量。

原首轮计时2499.441秒（41.66分钟）包含训练及全量EMA训练／测试评测，计时在checkpoint保存前结束。首轮已完整提交，测试混合balanced accuracy为54.3517%，并非良好泛化；[首轮完整标量与基线](../reproduction/reports/question5-ema-joint200-epoch001-2026-10-16.md)保留。

## 切换顺序与状态保护

先完成隔离源码、部署／恢复入口、观察入口及Git发布，再stage新namespace。准备期间原训练继续，不热改源码或配置。

仅向实际绑定的本任务rank0发出原生停止请求；四rank完成当前Adam边界后共同保存raw模型、EMA、Adam、四rank RNG、LR和精确采样游标。确认旧writer及所有旧训练进程实际退出，再核对完整文件SHA及四rank边界，保留旧源码、日志、claim和状态。

新执行使用同一实验run ID和独立execution／输出目录。兼容恢复只允许已发布源码身份和训练micro6→13变化；其余科学binding逐字段一致。完整状态恢复后从已保存的真实游标继续，不丢弃第2轮更新，不重复完整epoch，不以fresh0冒充续训。已有数据和冻结特征只读复用，不下载、编码、rollout或重新评分。

新执行须产生四rank原生恢复回执及实际Adam／EMA增量后，才能报告续接成功。只读取正常训练产生的记录，不新增GPU验收、模型概率扫描、额外评测或review。

## 当前状态

源码 `ed2a997e5162cd3a6e1dff0bd1d91fcc659caf2a` 已发布，四个关键文件从GitHub独立下载与本地SHA一致；A800实际931个源码文件完整对齐。第一次停止入口因远端Python缺少标准库pidfd接口，在发送信号前拒绝；该namespace／日志／intent保留。兼容修复采用Linux原生pidfd系统调用，仍保持出生身份校验和descriptor信号，不改SSH或运行环境。

23:01:51向原rank0精确发送一次原生停止请求；四rank在第2轮游标7337共同保存完整状态，累计16037次Adam／EMA、完整epoch1、已处理821094个G5。所有旧writer和训练进程均自然exit0。1242420704字节checkpoint的完整SHA为 `32d6b8453774b0ccccb96404750fdcda7804a14e8229000a5c47f200002b4f2b`，没有丢弃第2轮更新。

23:02:36一次启动同run `8af926a49630428097b94027864da68b`／新execution `141a9ada1d464f0d9925d8397a194944`。远端独立base为 `/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-8af926a49630428097b94027864da68b-resume-141a9ada1d464f0d9925d8397a194944`，wrapper PID1529509／birth624254768。实际handle为 `preparation/question5-largebatch-pidfd-resume-deployment-20261016.json`。

stage／stop／seal／launch回执分别为 `question5-largebatch-resume-stage-1792162882258401000.json`、`question5-largebatch-resume-stop-1792162911202882000.json`、`question5-largebatch-resume-seal-1792162940224021000.json`、`question5-largebatch-resume-launch-1792162956214725000.json`。这些证明完整状态保存、SHA和一次派发；**CUDA恢复、实际新增Adam／EMA及优化后的吞吐仍待原生回执**，不以启动成功冒称提速成功。

当前唯一只读入口为 `python -m reproduction.operations.a800_question5_largebatch_resume_observer --handle preparation/question5-largebatch-pidfd-resume-deployment-20261016.json`。该入口严格分开继承epoch与新执行更新；正常每30分钟一次有界读取，不增加测试或评测。

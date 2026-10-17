# A800 P10 完整状态恢复与执行切换记录（2026-10-17）

截至上海时间 2026-10-17T08:24:18.906444+08:00，已完整提交 **28/200 outer**，同一 pipeline 正在训练第29轮。第29轮已有12,800条候选前缀、评分和2,560条选中轨迹续写完成；其 actor ledger 当次各 rank 返回0次，不能计作完整29。

四 rank 副本只计一次：已保存 actor Adam **560**、scheduler **28**；其中父执行继承540，新执行真正新增20。探针仍为第20轮完整状态、累计100 Adam，新执行尚未更新探针。当前第29轮 in-flight 和 skip 数未记录，保留未知。

## 故障与完整恢复链

父执行 `cb12bf6ab10b46b0a72f1f0c2e753d78` / 源码 `858381986d3332898c4e8aa8e5c31e966363ceea` 在第28轮 continuation 的 vLLM sampler 出现 CUDA illegal memory access。父执行第28轮未进行 actor 更新；joint27、540 actor Adam、探针20的100 Adam均保留。原失败产物保留，3,041个记录的 owned birth 全部退出并完成本任务资源清理。

新执行 `1bf0183c24c84abfa8a65c587089145d` 使用运行源码 `0de9565c9b1c735635744f7e80a2526a2852e568`。恢复完整 actor/probe、Adam、scheduler、各 rank RNG 和候选池状态；复用第28轮已合格的12,800条 prefix及评分，未重采样、未重做actor27、未重置优化器、未重跑warmup。原失败 continuation 碎片没有导入。执行侧开启 `enforce_eager=True`，保持采样与科学算法不变。

此前一次 CPU 准备因两种 JSON 编码的文件 SHA 不同而拒绝；实际嵌入 pool 对象相同。已保留该失败 namespace，并修正为同时验证原 driver 编码/path/文件SHA、嵌入对象以及 reservation 的对象digest，没有放宽身份或文件SHA约束。该次失败未启动CUDA、未导入状态。

CPU全SHA/finite资格本身不等于CUDA恢复。此后两次真实记录分别证明：第28轮同prefix续写全完成并返回15次actor Adam；随后joint28完整提交、actor新增20次成功Adam，并继续第29轮。第28和第29轮 continuation 的四rank真实记录均为 eager。当前 pipeline **2043511 / birth627546281** 连续存活，未记录新错误。

## 分开计算的进度窗口

| 执行窗口 | 新增完整outer | actor Adam增量 | reward均值（各轮均值范围） | selected实际mixed |
| --- | --- | --- | --- | --- |
| 父执行：已送达anchor24之后，第25–27轮 | 3 | 60 | 0.690011178（0.664781622–0.709900560） | 785/1536 = 51.106771% |
| 新执行：继承27之后，第28轮 | 1 | 20 | 0.718482055（单轮） | 253/512 = 49.4140625% |

两窗口各自计算，未混用执行均值或峰值。以上均是普通轮 Top512 筛选后轨迹的真实 mixed 比例，既非 probe 分类准确率，也非每10轮利用真实EM选混合题时的 oracle 比例。

## 已有评分与未记录项

没有新 QA。现有最近一次 QA 仍为第5轮原512题：EM **23.046875%**，F1 **0.2888516865**。第20轮探针诊断属于已有记录，不能作为本次恢复的新测验或QA。第50轮QA尚未生成。

本 metadata 入口未提供当次GPU样本、窗口峰或本执行累计峰，均保留未知。小时anchor仍为父执行已实际送达的24轮，本文件未修改anchor；由主线发布并实际送达后再建立新执行切换anchor。

已满足两次真实正向流程与首完整恢复轮28的初始复核。另需跟进恢复后的第28–32轮全部完整及第33轮启动；这一五轮跟进目前仍待下次正常30分钟检查核实，未声称完成。

本次是恢复里程碑；训练继续到既定200轮及最终验收，**并未完成200轮实验**。

完整scalar与SHA见同名JSON。joint28 SHA：`c1c00046abcc2ec44185d835a802a26f1acf5a4ba8714dbd08787c40ffbe1c8c`。CPU资格SHA：`b72b9c09354ee45aba5b353f9cb55405a534e56dcbe956c10615933a801556da`；导入receipt SHA：`a51da822ffa72b174fb0b4144da37f964a1c20e1a69bfd564c6360bd331ffba4`。

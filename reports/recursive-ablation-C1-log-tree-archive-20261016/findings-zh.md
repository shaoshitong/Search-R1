# C1 必要日志与树记录补充归档

C1 的必要日志与四个训练节点的树原记录已实际复制到原同集群共享存储：43,960个文件、4,507,864,666字节。实际归档子进程 wait0；源 metadata/SHA 前后稳定及目标全量SHA通过，完成receipt SHA为 `685d68d474a60606f67d2f6d7f5fa5393f8a9ac4f667554d6edfaa16f2d19206`。本次从实际子进程启动到wait约220.55秒，包含清单、传输与校验，不是纯网络吞吐。

原95文件的关键HF/model/Adam/scheduler/RNG/driver归档保持不变；其权重没有在本次重新读取或复制。补充范围是原生日志、闭合证明、5/50/100/150四个TRAIN节点各512题的event/root/child/snapshot JSON、叶轨迹gzip文本及原manifest、既有树汇总与provenance。两个清单含少量重复闭合证明，不将文件数相加宣称唯一资产数。

`actor-inputs.pt`、模型与optimizer重复副本、环境及cache不在补充范围。dev512保留原生汇总日志，没有重做评测或补造逐题预测。本次只补齐证据持久性，不产生新的效果结论，也不改变训练数学。Mac仅保存代码、配置和小摘要，本次补充涉及的源日志和文本原件保留；共享服务器保留日期仍为UNKNOWN，不宣称永久保存。

独立CPU实现及入口发布于 `c8b5aeec873e9c226e575cd37e602f7bce750232`：`release/recursive_bran_a/ablation150_log_archive_v1/`，9个实际部署运行文件均绑定发布SHA。C2继续原科学源码 `233b0dc95de3f7b797d5b424d3b7dcaafca5328f` 与原控制器，本次没有GPU工作、训练锁、停训或额外评测。C2及其余消融仍未完成，整个目标保持ACTIVE。

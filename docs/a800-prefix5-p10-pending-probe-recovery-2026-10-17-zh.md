# A800 P10：保留actor10、补跑探针10的恢复（2026-10-17）

本次修复针对run `387d3f6e228a488f90c418e2ba47e28e` 的实际故障。父execution为 `90aea944c40d41eab4cf84efac02b22a`，运行源码为 `31f616264ceeb4f0e4d33b39bd85df1214afc15d`。

## 实际故障与保留状态

2026-10-17 04:39的单次既定观察确认父pipeline已经退出，随后必要错误核对确认全部本任务writer退出、cleanup通过。第10轮actor已成功返回20次Adam，累计四rank各200次Adam、scheduler10，并保存完整状态。探针在更新前的指标计算阶段因旧辅助函数只允许512/5120题而拒绝合法2560题；这是探针第一次Adam之前的错误。探针输出目录尚未创建，最新探针仍为0次在线更新。

因此父联合完整轮数为9，actor完整轮数为10。不能把第10轮提前算为联合完成，也不能丢弃或重复已保存的actor10更新。父源码、日志、失败job、完整actor/Adam/scheduler/RNG、probe0、候选游标及原始采样事件全部保留。

故障探针日志SHA256为 `75ccfac23d6df1e331a272fdd7ccd4a057f148d314caea04aaca032513c40a84`。实际错误类别为旧题数守卫的 `ValueError`；没有已记录的OOM或非有限值故障。

## 必要修改

指标辅助函数显式支持完整512、2560和5120题G5，保留轨迹唯一性、完整G5、own-EM二元标签、有限logit及问题身份检查。联合损失、混合定义、分类阈值和权重均不变。针对该故障的CPU回归覆盖新2560路径、旧512/5120兼容及缺失/重复/错误标签/冲突身份拒绝；不新增GPU验收或评测。

新增显式pending-probe恢复入口。先在远端一次CPU验收现有actor10与probe0的完整文件SHA、有限性、Adam/scheduler/RNG和pool10身份，再导入独立execution。大文件保留远端原始路径或使用同文件系统hardlink，Mac只保存代码及小型审计摘要。

## 恢复顺序

1. 发布修复源码、运行入口和本执行文档，核验远程Git SHA及文件blob。
2. 独立namespace stage准确源码；CPU资格与导入绑定父run/execution/source/config、终态和全部保存状态。
3. 一次实际准入四张空闲训练卡，排除辅助5/6；辅助卡也须空闲。UUID固定，不停止他人。
4. 仅复用第10轮的完整原始prefix、scored features、全候选own-EM、固定诊断缓存、真实EM选题和已保存actor10。只执行probe10的50次真实Adam，随后提交joint10。
5. 同一pipeline从第11轮继续到总200；不重复第10轮actor、rollout或诊断事件。恢复后跟进连续五个完整正常iteration及下一轮真实推进。

首个恢复候选execution `7d297c0c9828411b9c10565cbe9fdd7b` 在CPU资格阶段被拒绝，未启动GPU或导入状态。原因是资格检查将Python整数直方图键与持久JSON字符串键直接比较；这不是父选题或训练状态错误。该目录、源码、日志和失败回执保留，禁止重复其一次性prepare/launch。

修正后显式按持久JSON schema核对六个bin，仍严格比较全部512条有序选择、分数、版本和来源SHA；先完成小型语义检查，再读取大型状态。新的独立execution为 `cb12bf6ab10b46b0a72f1f0c2e753d78`，远端base为 `/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-387d3f6e228a488f90c418e2ba47e28e-resume-cb12bf6ab10b46b0a72f1f0c2e753d78`，仍只从原父execution `90aea944c40d41eab4cf84efac02b22a` 导入。部署/CPU资格/一次启动/实际推进必须分别以回执证明；本说明本身不是启动或恢复成功证明。

## 科学配置保持

候选2560题×G5，正式每轮512题×G5，actor每轮20次Adam；普通轮按探针选题，更新轮真实混合题优先，不足512时按探针分数补齐。probe仅在10、20、…、190轮各自然训练一遍2560题G5，即50次Adam、LR5e-6；200不更新。虚拟0仅加入真实全对组的归一化统计，全错与混合组保留原公式，不增加虚拟轨迹。

恢复完整状态不代表重新初始化actor或探针。总目标仍为200个完整联合outer、既定最终51713题评测、完整checkpoint有限性/SHA和only-owned收尾；旧warmup200及旧实验保持独立历史，不因本次故障恢复重新启动。

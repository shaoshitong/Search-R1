# 严格匹配的head LR1e-4候选

本预注册在候选实际GPU提交前固定。512题head-only诊断表明原5e-6拟合较慢、1e-4可以拟合但内部holdout泛化弱，因此本次只检验原完整30016题/938次joint更新中的head LR这个因素，不把tiny-fit成绩或本候选内部指标称为goal胜者。

| 控制因素 | 原743619/immutable501 | 新候选 |
|---|---|---|
| 起点 | 原E1 raw/EMA/legacy Adam10464、rank0 RNG及四rank历史档案 | 完全相同，CP SHA02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724 |
| 数据/次序 | 已冻结30016 NQ、真实G10、B32×938、原四唯一G5视图/每题总权重1 | 相同bank/plan/seed/epoch/order，不加入新15616或12800题 |
| legacy Adam | LR5e-6、完整继承，不reset | 相同 |
| 新count head | 同seed1729、同结构fresh Adam0、LR5e-6 | **仅LR改1e-4**，其余相同 |
| loss/数值 | trajectory BCE1+count NLL1，encode BF16→float、classifier/NLL FP32，EMA.99 | 相同 |
| 保存/验证 | full9状态、先CP再固定256/512/768/938节点、同既有diag/固定新G5 selection | 相同，不读final |
| 资源/隔离 | 原作业保持 | 新独立normal/acd_u 1H100、8CPU、64G、3h、无依赖 |

候选配置17000B SHA `71c3ee90f0232b889d4ced2d0cb50d69e852eb1ebbdbb55fdff03eaefdbbcc71`；sbatch1244B SHA `8f966ce8dacec3473774837faced274a02f71be5cf301ed75f0ebef0c5899d5c`。真实CPU prepare复核config剥离source/provenance字段后，除 `new_head.learning_rate` 外科学字段逐值相等。原control config SHA3e910b12...，共同plan SHA8d6f5b04...，共同bank SHA12f5e895...。

源码为独立 `direct_count_headlr_train.py` / `direct_count_headlr_update.py`，从immutable501复制产生显式 `ENTRY-DIFF.patch`；不修改原源码或743619。实际focused CPU两检查通过：配置只改所述字段；首个toy joint update的legacy完整状态等同control、head更新不同且全状态重放一致。仅CPU准备未加载真实权重。

根已读typed config及ENTRY-DIFF并明确接受后，才开放独立guarded submit。提交前须source/config/prereg均推送并核远程SHA，同名job、intent和锁检查通过；未知结果不重复提交。新启动跟首完整真实update双重重放、独立CP1九状态CPU验收、连续5+next。此候选仍只执行预注册938单pass预算，保留完整恢复状态；真实终态和同固定评测验收后再判断，不能自动扩大外部筛分。

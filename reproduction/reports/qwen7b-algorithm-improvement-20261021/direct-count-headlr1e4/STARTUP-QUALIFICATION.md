# headLR-only候选：实际启动资格通过

新作业743809，normal/acd_u、1H100/8CPU/64G/3h、无依赖；原743619不变。此报告只确认真实执行和控制因素，不报告科学提升。候选仍为原30016题、938次单pass。

- runtime immutable `c7947e478cb96e99f0ebe4b08c9a5bb202945cae`，新独立namespace `direct-hidden-count30016-headlr1e4-v1`；config17000B SHA71c3ee90f0232b889d4ced2d0cb50d69e852eb1ebbdbb55fdff03eaefdbbcc71。
- 首真实joint update的raw/EMA/两Adam/原buffer/RNG全9组件双重GPU重放PASSED。新namespace自己的CP1独立fullSHA+九组件CPU回读PASSED；CP1 3754035012B，SHA `eb49229e1a0ee52c56817d9aa70a65e4e49e19afb1c115233fcfaf73749fb99e`。CP内绑定digest和外部config文件SHA是两个不同字段，未混用。
- 与原control已接受CP1直接比对：legacy raw/EMA/Adam/native两buffers/RNG全部完全相同；新head Adam矩也相同。仅head raw和head EMA不同，符合只改head LR后的第一步预期。首步loss2.4544334411621094与control相同。
- 两次真实进展为cursor2(legacy10466/head2)→cursor6(10470/head6)；实际1..6逐步连续，每步micro replay exact。之后已确认cursor7继续，最新实测cursor16(legacy10480/head16)，无failure。
- 初始化同PID的rchar24.40GB→52.87GB为必需原bank reader验收和读取推进，并未因初始化未产step而重提。有限startup observer已成功正常退出，不存在新自动化/heartbeat。
- 后续只验本新namespace及head Adam LR1e-4、同fixed256/512/768/938；不得将原control审计冒充本作业验收，不新增GPU或外部筛选。

| 更新 | legacy Adam | head Adam | count NLL | trajectory BCE |
|---:|---:|---:|---:|---:|
| 1 | 10465 | 1 | 1.816862 | 0.637572 |
| 2 | 10466 | 2 | 1.707853 | 0.798121 |
| 3 | 10467 | 3 | 1.819372 | 0.611113 |
| 4 | 10468 | 4 | 1.701990 | 0.680216 |
| 5 | 10469 | 5 | 1.471568 | 0.551558 |
| 6 | 10470 | 6 | 1.345000 | 0.487624 |
| 7 | 10471 | 7 | 1.448820 | 0.625507 |

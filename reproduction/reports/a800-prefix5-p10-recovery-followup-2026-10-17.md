# A800 prefix5 P10：恢复后完整11–15轮与同进程16已实核（2026-10-17）

实际观测：2026-10-17T05:47:36.052844+08:00。**已完整15/200轮；恢复后11–15连续五个正常outer及同pipeline第16轮的真实推进已核实。实验200轮尚未完成。**

Actor累计300次成功Adam＝父执行保留200＋新执行100；探针累计50次Adam，仍冻结在第10轮。四rank副本只计一次。第16轮前缀12800条已完整、score实际exit0，续写正在运行；捕获时未启动actor16，不计第16轮完整。

## 实际身份和完整恢复链

- Run `387d3f6e228a488f90c418e2ba47e28e`；execution `cb12bf6ab10b46b0a72f1f0c2e753d78`；运行源码 `858381986d3332898c4e8aa8e5c31e966363ceea`，未热改。
- Pipeline PID1861870 / birth626397110与恢复launch及前两次实测一致，实际alive。
- 父source31f／ex90ae保留完整actor10／200Adam、scheduler10和四rankRNG；新执行只补probe10／50Adam形成joint10，没有重训actor10、重复采样或重置optimizer。
- 现有完整11、12、13、14、15的actor阶段均实际exit0，四rank分别累计220、240、260、280、300，scheduler推进至15。CPU full-SHA/finite资格本身仍不作为CUDA验收；当前正常更新证据来自这些实际原生训练完成记录。

## 五轮真实scalar记录

| Outer | Actor累计Adam | Probe累计Adam | Reward均值 | 实际混合题 / 512 | 混合比例 | 真EM选题 | Joint SHA256 |
|---|---:|---:|---:|---:|---:|---|---|
| 11 | 220 | 50 | 0.7937674565 | 358/512 | 69.9219% | 否 | `f3a7431cc200501935b5fac00ff61f84ebedd4fe75f138109bb50aa6fe8e84c9` |
| 12 | 240 | 50 | 0.7901032522 | 327/512 | 63.8672% | 否 | `58b03601b3552b1d5b89fb95a1b55e643c0cef3f10428342cadd119048ae6da4` |
| 13 | 260 | 50 | 0.7953783684 | 338/512 | 66.0156% | 否 | `d040fc7d3572ae288529cdf4ac3de44cf2b7a0d7048582fead9f18dfed701bd8` |
| 14 | 280 | 50 | 0.8118473483 | 318/512 | 62.1094% | 否 | `cbd7332f493e61d73cb494f0c5ee50b4daa888a9a6f2d32047b711a37b0469ae` |
| 15 | 300 | 50 | 0.8525460059 | 333/512 | 65.0391% | 否 | `afdd631ea17ed87316ab45cc9cd38e01d7b64c8259c1c881d8406cb3fa93b15f` |

完整窗口严格为anchor10之后的11–15：reward均值 **0.8087284862**，outer均值范围 **[0.7901032522, 0.8525460059]**；实际混合 **1674/2560＝65.390625%**，全对虚拟0归一化实际 **624组**。

11–15均为普通预测Top512轮，未用真实EM选题，也未更新探针。第10轮真实EM优先选题的100%混合结果和父执行reward窗口不混入上述平均。

## 第16轮真实推进

| Rank | 已完整prefix轨迹 | 应有prefix轨迹 | 生成阶段 | elapsed秒 |
|---|---:|---:|---|---:|
| 0 | 3200 | 3200 | NATIVE_RANK_PHASE_COMPLETE | 34.104311 |
| 1 | 3200 | 3200 | NATIVE_RANK_PHASE_COMPLETE | 35.438532 |
| 2 | 3200 | 3200 | NATIVE_RANK_PHASE_COMPLETE | 34.702362 |
| 3 | 3200 | 3200 | NATIVE_RANK_PHASE_COMPLETE | 38.583859 |

四rank共12800条前缀，actor版本一致为actor0015；随后score实际exit0，pipeline已进入outer16续写。续写progress尚未发出，保留unknown，不报0完成或提前报2560完成。actor16没有开始，不报告额外Adam或完整outer16。

## 已有QA与探针指标时钟

- QA仍是第5轮既有dev512：EM118/512＝23.046875%，F1 0.2888516865；11–15没有新QA评分，50/100/150/200尚未生成。
- Probe仍是第10轮已保存的50次Adam结果，11–15未产生新的探针更新或固定512前后诊断评分；正常选题打分继续进行。第10轮历史普通混合准确率：train74.2969%→80.0781%，test58.5938%→57.4219%；历史balanced accuracy：train72.2374%→79.5989%，test56.0721%→58.1743%。这些不是第15轮新测值，也不等于逐轨迹准确率或QA EM。
- 第10轮历史两loss（轨迹BCE／混合BCE）：train0.429205／0.576476→0.360586／0.439025；test0.788956／0.827688→0.706205／0.873174。原值和总loss均在JSON明确标记历史记录。

## 状态hash与监控窗口

- Joint15 SHA256：`afdd631ea17ed87316ab45cc9cd38e01d7b64c8259c1c881d8406cb3fa93b15f`。
- Probe10 SHA256：`2579b5ec44ff079ce1a9ec22c54bd03824c50136dc9ad63a14a28fc74b682eaa`。
- CPU资格 SHA256：`bfbb0da038bbf5877bd0838ffeb515c5d8b60c33514075c3e7e10355b85aded4`。
- 已有QA5 metadata SHA256：`05520a85768d35ced42ef730d9789af31c1195e128c055f1923596b1d3245aa6`。
- 本次actual primary SHA256：`b5fdd952edc5fdb797437cb98aa2e4db5ce04d99e31977288b9860f2590c90e6`。
- 已送达switch anchor SHA256：`a835e6889197c797e5a7115230446aa6478f354f8c008d8053e159176f892f3a`；仍保持joint10／actor200／probe50，未由本报告推进。

本execution累计新增actor100、probe50；从已送达anchor10起新增actor100、probe0。下一次常规实际观察不早于 **2026-10-17T06:17:36.052844+08:00**；小时报告到期时间为 **2026-10-17T06:09:09.622933+08:00**，按已送达anchor计算。本次是恢复后的连续五轮里程碑记录，不单独推进小时anchor。

已有实际primary未记录新terminal error，五个actor阶段exit0。GPU当次/本execution窗口峰及累计峰未投影，保持unknown，不借用父执行统计。后续继续同进程至200及既定最终验收，当前不标记实验完成。

本报告仅由本地既有实际原件投影scalar/hash；未SSH、Git、改运行源码、追加评测、review或模型/tensor读回。不含probe URL、环境全文、原QA/token/rollout或完整job进程列表。

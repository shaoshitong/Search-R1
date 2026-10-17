# Prefix5 I1–I6恢复与候选规模切换：实际第6轮（2026-10-17）

本次只修正式actor实验，保留已有探针初始化；不重跑独立warmup。运行source `75d536575593891876df339467da73b3c4cbdc7c`，只读操作入口 `c511f0ae9d5bcc12d129431ff214279b4ade1551`。实际metadata时间为2026-10-17 15:45:48 Shanghai。

- 当前同run `54ac33dcebe64e62a11fe4177f41f116`，新execution `a4a0c2a18cca40bc8c3850d40eed8554`。
- 从完整outer4／80次Adam全状态恢复；已真实完整结束outer5和6，累计120次成功actor Adam，四rank副本只计一次。保留Adam、scheduler、RNG、探针与题池游标。
- 第5轮原2560题前缀与评分复用，未重采或重复actor更新；第6轮起新普通轮规则真实生效：5120题、25600条前缀→Top512题→2560条完整actor轨迹→20机会、20次返回Adam、0skip→joint6。
- 第6轮选中真实Mixed为396/512=77.34375%。这是已筛选子集的真实比例，不是探针balanced accuracy，也不是QA成绩。
- 现成固定512题QA5：strict EM136/512=26.5625%，F1=36.06585167224031%；NQ256 EM25%、Hotpot256 EM28.125%。新共享BF16交互协议已绑定，评分器没有放宽。未新增评测，不能据此宣称I1–I6带来确定的因果收益。
- 探针仍0次在线更新。自然P10更新尚未到达：预期2560题×5全标签、LR5e-6、global平均256、1epoch、50次probe Adam；不得为验证额外插入更新轮。第200轮不更新probe。

[I1–I6代码和配方](../../docs/a800-prefix5-i1-i6-fresh200-20261017.md)已保留：训练／QA共享交互与BF16、原micro2等权目标、grounding生命周期、安全监测和流式action终止。grounding50/100/150实际拟合与采用尚未发生，仍是未来验收项。总目标200不变，未完成最终评测／checkpoint终验／资源收尾。

原始权重、Adam、token、QA、rollout和完整日志均留远端；本目录只发布现有标量、身份和哈希。

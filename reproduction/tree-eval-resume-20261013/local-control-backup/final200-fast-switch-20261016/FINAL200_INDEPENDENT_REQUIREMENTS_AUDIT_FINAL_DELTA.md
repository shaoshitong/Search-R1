独立最终增量验收：PASS（保留如实声明的限制）。本任务可标记 goal complete；reader 未改变 goal 状态。

42项最终状态：35 proven、7 limitation、0 incomplete、0 missing。旧 DRAFT 和 TRACE_PRESERVATION_DELTA 原字节保留；本次仅补 R32/R37/R38/R40 与最终 CSV/plot/SSH 收尾证据。

全量51713七集评测 COMPLETE / worker exit0。809份SHA绑定chunk连续覆盖0..51712，8rank身份一致，HF200全部文件SHA与已有CPU重载验证一致。F1 0.538153839337549；EM 0.4693984104577186。七集题数 {"2wikimultihopqa": 12576, "bamboogle": 125, "hotpotqa": 7405, "musique": 2417, "nq": 3610, "popqa": 14267, "triviaqa": 11313}。原始score留远端；terminal只读collector逐score重聚合，本reader本地核对chunk/ref/标量汇总和加权关系。

两帧无本任务PID/cgroup/lease-holder/18082监听；CPU审计单元两帧inactive/PID空，本任务SSH ControlMaster两次socket absent。用户仅停旧final的授权范围被保留，旧final FAILED不篡改成训练失败或正常退出。

成本：新final终态journal CPU28190.250s（1ms报告精度），GC后的精确ns及monotonic仍null；batch CPU26537.642854711s是子集，不另加；supervisor wall3636.757847548s不覆盖更早preflight。旧778全服务CPU1116210.663921s包含训练/dev/partialfinal，不重复加回；旧final直接失败control600.934673652s单列，完整final-only CPU未知。HTTP/scoring缺失保留null+known_sum；GPU共享capacity不称own/billed/energy。

完整200轮CSV的6000个原标量逐值一致；最终图200训练/40dev无平滑插值，其PNG/PDF SHA与receipt一致。

接受的限制：选择偏差未被token级μ保证消除、有限搜索不保证差异；outer1八rank抽样delta0不能称历史样本全非零；native5真实恢复与native200 CPU验收不等于完整GPU executor exact-resume；历史rolling非全200永久tensor保留；原h1009八PID当前身份不可观测；无合格匹配GRPO效果对照；成本不可恢复的细分缺失明确保留。全部限制不伪造为零或精确证明。

关键本地证据及SHA：

- outputs/tree-nested-branch-handoff-20261013/FINAL200_FAST_TERMINAL_DELTA.json — 84471 B / ec28a24fa267a51ae84cd64f7be85df82004fa36f774a9000169c6d54da2cbf9
- outputs/tree-nested-branch-handoff-20261013/FINAL200_FAST_TERMINAL_DELTA_READBACK.json — 3459 B / 4b4640994ec031f85a50b16f4c7191b8ab82050582e09c3dc8256e370b7065f3
- outputs/tree-nested-branch-handoff-20261013/FINAL200_FAST_TP1_ACTUAL_OBSERVATION_14.json — 1649265 B / 20c4ca439035fe1462f07dc1c4c8d0c3f37a8dd6e45509f591677d16d0fd93a0
- outputs/tree-nested-branch-handoff-20261013/FINAL200_FAST_TP1_ACTUAL_OBSERVATION_15.json — 1649268 B / 4964d62e33f867b65eafea52f4e486eb85f9409cecb8d0401e6c371639fc8592
- outputs/tree-nested-branch-handoff-20261013/FINAL200_FAST_TERMINAL_JOURNAL_CPU_SIDEEVIDENCE.json — 1311 B / 02d4be7423a0b717a3e0aa3054441361ab56195d9df66ec806e31fed37871348
- outputs/tree-nested-branch-handoff-20261013/ACTUAL200_COMPLETE_SCALAR_SERIES.csv — 63907 B / f0152a04ed1fe4dab31894defbf56b6e056169aa4b293660bd47f2aa2e32a33a
- outputs/tree-nested-branch-handoff-20261013/FINAL200_LOCAL_OWNED_SSH_CONTROL_CLEANUP.json — 732 B / 3c81406668c77484d81fceced45a0bca09ca767e6ff9acbf2b78e31d6115cbb2

JSON记录全部42项最终状态、精确refs/SHA与两次实际输入读回。没有新远端调用、重评测、重扫或模型读取。

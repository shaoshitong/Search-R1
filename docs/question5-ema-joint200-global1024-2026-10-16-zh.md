# Question＋prefix5：全局batch1024续接（2026-10-16）

用户明确选择将全局Adam batch从平均256条轨迹增至1024条，并接受Adam次数和EMA样本窗口变化。此指令替代此前全部epoch固定8700更新的执行参数；完整数据、联合BCE权重1:1、EMA0.99、200个完整epoch不变。

新全局G5 pattern为204/204/204/204/208，平均每次Adam1024条轨迹；每个完整新epoch为2175次Adam。训练每卡micro52个完整G5（最多260条轨迹），评测每卡micro64个G5（320条）。同题五条轨迹不拆分、不补重复样本。

先部署隔离源码，再由原生停止流程保存当前raw、EMA、Adam、四rank RNG、LR和数据游标。若封存的是partial epoch，该epoch余下部分沿用原8700步采样计划完整收尾；下一完整epoch起切换新batch。设封存得到的旧配方结束epoch为N，则实际总Adam预算为 `N*8700 + (200-N)*2175`，N由真实封存记录确定，不能预先伪造。

学习率按已处理数据阶段连续推进：边界B=N*8700之前沿用原clock，之后虚拟clock为 `B+4*(实际Adam-B)`，对原1740000步cosine计算；最终实际更新精确使用终点3e-6，不重置Adam或学习率。EMA仍在每次成功Adam后更新0.99，因batch增大，对应样本窗口随之增大。

当前父执行为run `8af926a49630428097b94027864da68b`／execution `141a9ada1d464f0d9925d8397a194944`／source `ed2a997e5162cd3a6e1dff0bd1d91fcc659caf2a`。新入口为 `a800_question5_global1024_resume_control.py`，调用顺序 `stage → stop → seal → launch`；只读入口为 `a800_question5_global1024_resume_observer.py`。旧入口、源码、日志和状态保持可追溯，不热改活跃源码，不恢复旧actor目标。

实际源码 `ddf36cca14ac7aafb0b5395ffa88d57ed8c35e74` 已推送并独立核对四个关键文件；A800实际934个源码文件完整SHA一致。23:23:19原生停止请求精确发送一次，四rank退出和完整状态封存通过。恢复点累计17403次Adam／EMA，前2轮完整，第3轮已做3次更新；状态1242421600字节，SHA `7de3176b54f3f72b3bdf259c15eb7dc6bdb31ce8238929d5d32b3a58a5476803`。

因此真实N为3：第3轮余下部分沿旧batch收尾，第4轮起全局1024；实际总预算454575次Adam／EMA。23:27:42一次启动run `8af926a49630428097b94027864da68b`／新execution `bcfa1de0a90a4fcda70263a75153fcee`，wrapper1551397／birth624405375。handle为 `preparation/question5-global1024-resume-deployment-20261016.json`，base为 `/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-8af926a49630428097b94027864da68b-resume-bcfa1de0a90a4fcda70263a75153fcee`。

实际stage／stop／seal／launch回执分别为 `question5-global1024-resume-stage-1792164146335709000.json`、`question5-global1024-resume-stop-1792164198951087000.json`、`question5-global1024-resume-seal-1792164357776413000.json`、`question5-global1024-resume-launch-1792164462378931000.json`。CUDA恢复、新增更新与真实吞吐继续核对；启动回执不等于训练或提速成功。旧执行的实际epoch2指标按恢复链读取，不以新执行指标冒充。

23:34:20实际回执 `question5-global1024-resume-observation-1792164860126803000.json` 已确认四rank完整CUDA恢复并新增822次Adam／EMA，累计18225，当前第3轮游标825、仍为旧batch阶段。未记录错误、退出或nonfinite拒绝；新全局1024阶段尚未开始，不能报告其吞吐。恢复链完整核验了epoch1与epoch2原生metrics／checkpoint／complete记录。

23:40:24第二次实际核对已到20250次Adam／EMA、第3轮游标2850；四rank一致、身份连续、无记录错误。两次实测窗口新增2025次更新，证明持续推进。当前旧batch阶段约1428条轨迹/秒，不是全局1024的吞吐；1024仍将在第4轮自动生效。后续正常每30分钟一次实际检查，保持同进程继续200轮。

继承epoch2的EMA混合balanced accuracy为训练76.1967%、测试57.3314%；逐轨迹BCE为0.416559／0.592635，G5 BCE为0.484883／0.779791。测试BCE较首轮变差，不能仅凭balanced accuracy上升宣称泛化改善。epoch2记录的985.782秒只覆盖恢复后的1363次Adam和全量EMA评测，**不是完整epoch耗时，也不能与首轮41.66分钟直接比较提速**。完整标量／哈希见[实际续接报告](../reproduction/reports/question5-global1024-resume-2026-10-16.json)。

不新增rollout、编码、评分、GPU验收、额外评测或review；当前probe目标继续200完整epoch，旧actor目标保持暂停。

## 2026-10-17实际进展：第4轮全局1024已生效

00:11:04实际回执`question5-global1024-resume-observation-1792167064807867000.json`确认epoch3的metrics／checkpoint／complete全部闭合，完整进度3/200；四rank累计26175次真实Adam／EMA，当前第4轮75/2175次更新。第4轮实际global G5 pattern为204/204/204/204/208，平均1024条轨迹；最近一次每卡52个G5／260条。仍为同一run／execution／runtime源码与连续进程，无记录错误、退出或nonfinite拒绝。

epoch3同一EMA0.99权重的全量训练／测试混合balanced accuracy为80.1052%／56.3273%；逐轨迹BCE为0.381939／0.570881，G5 BCE为0.433038／0.777596，联合loss为0.814977／1.348477。训练／测试混合balanced accuracy差23.7778个百分点，测试较epoch2下降1.0040个百分点。固定测试逐轨迹accuracy78.2422%，仅比全判错误78.1641%多判对2条，当前泛化仍弱。

epoch3记录2293.261938秒只覆盖本执行恢复后的8697/8700次Adam与全量EMA评测，不包含恢复前3次更新、构造／恢复及checkpoint写入；该轮仍属旧global256配方。全局1024的稳定同阶段速度窗口尚未形成，不能将这段耗时解释为完整1024轮耗时或报告固定倍数提速。

epoch3最新完整状态1242424288字节，SHA256`4fccd80f9c0761e0b613f8fbf45f0b562c08ec9d007fa868aa746bc49b4f4437`；`latest.pt`会被后续完整轮替换，不是epoch3不可变权重副本。每10轮／200轮的不可变保存策略不变。详见[2026-10-17实际里程碑报告](../reproduction/reports/question5-global1024-epoch003-phase-start-2026-10-17.md)及[标量／哈希JSON](../reproduction/reports/question5-global1024-epoch003-phase-start-2026-10-17.json)。

## 2026-10-17 00:41：第4轮训练窗口实测

第4轮四rank全部2175次Adam／EMA已返回，累计28275次，进入既定同EMA训练／测试评测。第4轮metrics／checkpoint／complete仍缺失，`fully_committed`未知；即使原生进度中的`complete_epochs`已写4，**正式完整进度仍只计3/200**。身份连续、无记录错误或退出，保持同进程推进。

同epoch4两份已有原生进度的游标75→2175，`observed_unix`相差1336.247471秒，真实新增2150400条轨迹／2100次Adam，得到**1609.28条轨迹/秒**。计算没有使用observer墙钟间隔；它不含全量EMA评测和checkpoint保存，也不是完整1024轮端到端耗时，不能据此宣称4倍提速。此次没有新EMA评分，最近混合balanced accuracy80.1052%／56.3273%仍属epoch3。详细证据见[第4轮训练窗口报告](../reproduction/reports/question5-global1024-epoch004-training-window-2026-10-17.md)及[标量／哈希JSON](../reproduction/reports/question5-global1024-epoch004-training-window-2026-10-17.json)。

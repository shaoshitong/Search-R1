# Question＋prefix5：全局batch1024续接（2026-10-16）

用户明确选择将全局Adam batch从平均256条轨迹增至1024条，并接受Adam次数和EMA样本窗口变化。此指令替代此前全部epoch固定8700更新的执行参数；完整数据、联合BCE权重1:1、EMA0.99、200个完整epoch不变。

新全局G5 pattern为204/204/204/204/208，平均每次Adam1024条轨迹；每个完整新epoch为2175次Adam。训练每卡micro52个完整G5（最多260条轨迹），评测每卡micro64个G5（320条）。同题五条轨迹不拆分、不补重复样本。

先部署隔离源码，再由原生停止流程保存当前raw、EMA、Adam、四rank RNG、LR和数据游标。若封存的是partial epoch，该epoch余下部分沿用原8700步采样计划完整收尾；下一完整epoch起切换新batch。设封存得到的旧配方结束epoch为N，则实际总Adam预算为 `N*8700 + (200-N)*2175`，N由真实封存记录确定，不能预先伪造。

学习率按已处理数据阶段连续推进：边界B=N*8700之前沿用原clock，之后虚拟clock为 `B+4*(实际Adam-B)`，对原1740000步cosine计算；最终实际更新精确使用终点3e-6，不重置Adam或学习率。EMA仍在每次成功Adam后更新0.99，因batch增大，对应样本窗口随之增大。

当前父执行为run `8af926a49630428097b94027864da68b`／execution `141a9ada1d464f0d9925d8397a194944`／source `ed2a997e5162cd3a6e1dff0bd1d91fcc659caf2a`。新入口为 `a800_question5_global1024_resume_control.py`，调用顺序 `stage → stop → seal → launch`；只读入口为 `a800_question5_global1024_resume_observer.py`。旧入口、源码、日志和状态保持可追溯，不热改活跃源码，不恢复旧actor目标。

实际源码 `ddf36cca14ac7aafb0b5395ffa88d57ed8c35e74` 已推送并独立核对四个关键文件；A800实际934个源码文件完整SHA一致。23:23:19原生停止请求精确发送一次，四rank退出和完整状态封存通过。恢复点累计17403次Adam／EMA，前2轮完整，第3轮已做3次更新；状态1242421600字节，SHA `7de3176b54f3f72b3bdf259c15eb7dc6bdb31ce8238929d5d32b3a58a5476803`。

因此真实N为3：第3轮余下部分沿旧batch收尾，第4轮起全局1024；实际总预算454575次Adam／EMA。23:27:42一次启动run `8af926a49630428097b94027864da68b`／新execution `bcfa1de0a90a4fcda70263a75153fcee`，wrapper1551397／birth624405375。handle为 `preparation/question5-global1024-resume-deployment-20261016.json`，base为 `/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-8af926a49630428097b94027864da68b-resume-bcfa1de0a90a4fcda70263a75153fcee`。

实际stage／stop／seal／launch回执分别为 `question5-global1024-resume-stage-1792164146335709000.json`、`question5-global1024-resume-stop-1792164198951087000.json`、`question5-global1024-resume-seal-1792164357776413000.json`、`question5-global1024-resume-launch-1792164462378931000.json`。CUDA恢复、新增更新与真实吞吐继续核对；启动回执不等于训练或提速成功。旧执行的实际epoch2指标按恢复链读取，不以新执行指标冒充。

不新增rollout、编码、评分、GPU验收、额外评测或review；当前probe目标继续200完整epoch，旧actor目标保持暂停。

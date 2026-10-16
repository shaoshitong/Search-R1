# Question＋prefix5：全局batch1024续接（2026-10-16）

用户明确选择将全局Adam batch从平均256条轨迹增至1024条，并接受Adam次数和EMA样本窗口变化。此指令替代此前全部epoch固定8700更新的执行参数；完整数据、联合BCE权重1:1、EMA0.99、200个完整epoch不变。

新全局G5 pattern为204/204/204/204/208，平均每次Adam1024条轨迹；每个完整新epoch为2175次Adam。训练每卡micro52个完整G5（最多260条轨迹），评测每卡micro64个G5（320条）。同题五条轨迹不拆分、不补重复样本。

先部署隔离源码，再由原生停止流程保存当前raw、EMA、Adam、四rank RNG、LR和数据游标。若封存的是partial epoch，该epoch余下部分沿用原8700步采样计划完整收尾；下一完整epoch起切换新batch。设封存得到的旧配方结束epoch为N，则实际总Adam预算为 `N*8700 + (200-N)*2175`，N由真实封存记录确定，不能预先伪造。

学习率按已处理数据阶段连续推进：边界B=N*8700之前沿用原clock，之后虚拟clock为 `B+4*(实际Adam-B)`，对原1740000步cosine计算；最终实际更新精确使用终点3e-6，不重置Adam或学习率。EMA仍在每次成功Adam后更新0.99，因batch增大，对应样本窗口随之增大。

当前父执行为run `8af926a49630428097b94027864da68b`／execution `141a9ada1d464f0d9925d8397a194944`／source `ed2a997e5162cd3a6e1dff0bd1d91fcc659caf2a`。新入口为 `a800_question5_global1024_resume_control.py`，调用顺序 `stage → stop → seal → launch`；只读入口为 `a800_question5_global1024_resume_observer.py`。旧入口、源码、日志和状态保持可追溯，不热改活跃源码，不恢复旧actor目标。

本文件先记录新配方及恢复约束，实际部署、封存边界、CUDA恢复、真实更新和吞吐待对应回执补齐。启动回执不等于训练或提速成功。不新增rollout、编码、评分、GPU验收、额外评测或review。

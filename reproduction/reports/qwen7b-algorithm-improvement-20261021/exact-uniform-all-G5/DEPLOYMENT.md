# 已授权实际执行：唯一 all252 G10 作业

父任务最新明确GO覆盖此前“等待G20最终证据后才可启动”的预备约束。本次只新启动这一个独立同源G10对照，不更改或停止既有任务。

实际提交时间：2026-10-21T05:21:41.553789+08:00；作业 **743417**，normal/acd_u、1H100/8CPU/64G/2h、无依赖。先实际核对账户同名作业为空、本任务锁与intent，再hold提交记录ID后release。观测已RUNNING于ACD1-28，但首个完整更新验收尚待后续观测，不能仅凭RUNNING称训练成功。

- 运行代码固定 `48eb944281d2e3d6fa5ad66e4d6f0f9eaf18aeec`，完整闭包20文件逐文件SHA通过，未热改活动源。
- HPC3实际10 focused tests通过（5.552秒）；typed CPU gate确认真实6528题/65280轨迹、ownEM和训练/评估隔离、原136×48/epoch0顺序、固定E1 fullstate和原生L26/P20配置。
- 实际config 9164B，SHA `66866b17f374cee5384582806000038f0bbd1cbe2d3d4830f74478c05e1d812f`。
- 原random4 config SHA `224558c829cb4b1d63330d0008d3fdaa57acb2cba624cc3fb56cb9b9deba49ec`；原plan 573412B SHA `3bff8f5331bd60634456b1f24a106a752f84b196bc0a0cd8961c5d17405e86d9`，descriptor和顺序原样继承。
- E1 source SHA `02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724`，Adam10464，raw/EMA/Adam/RNG均由实际GPU启动验收；未在CPU prepare加载checkpoint tensor。

后续必须确认首真实更新的六组件双重放、完整CP1及第二次真实增量，然后跟完136更新、最终CP验收和同一原生固定G5验证；不根据训练loss或队列状态宣称提升。native E4基线为BA55.529%/AUC57.610%/Top200；相同G10random4完整136为BA55.148%/AUC57.322%/Top194，不能混用另一HF backend的E4数值。

所有模型、Adam、原始预测和数据均远端。这里仅备份代码/配置/小型进展与SHA。有限只读observer每60秒记录一次，出现failure/complete/异常终态即退出，不会提交、重启或改任务。

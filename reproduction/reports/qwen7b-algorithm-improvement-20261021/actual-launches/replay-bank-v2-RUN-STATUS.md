# R2 多层capture实际运行验收（尚未完成）

controller92bf7117404fb2b6c4dd23794a49773b777b67ed；prepared/capture65f08f72b4e61cf72e3c43a202bda0508830a160保持不动。准备SHA a7fa015dd655ef40d371172375ae30e4280503375485bd61ed6ab49baf5c1fd9。

每片仅提交一次、无dependency。先normal；pending1/2/3在实际预计>30分钟后按授权用原ID逐级到acd_ue，未用emergency。原始preflight、submit、schedulerbefore/after及每次只读observer脚本/JSON均在本目录备份，未git提交。

|shard|job|partition|node|真实轨迹|完整chunk|实际feature文件字节|
|---|---|---|---|---:|---:|---:|
|0|743087|acd_u|ACD1-11|6611/11360|20|3521551992|
|1|743088|acd_ue|ACD1-8|2333/11360|7|1242083070|
|2|743089|acd_ue|ACD1-36|4141/11360|12|2209697565|
|3|743090|acd_ue|ACD1-17|2700/11360|8|1443868754|

四片都已通过至少两次真实轨迹、chunk和字节增长核对，run/capture锁均HELD，sacct/scontrol仍RUNNING，无failure回执或异常退出。日志仅有预期的HF模型先CPU初始化再to(cuda)提示，不把该提示当作失败。

这只是运行推进验收。完整capture、每片全部ReplayFeatureReader验收及四片full-SHA merge尚未完成。仍需持续跟进失败和最终bank-complete，不能当成层对照训练已完成。没有启动student训练，所有模型/特征/原始响应保持远端。

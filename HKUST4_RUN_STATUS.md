# HKUST4 启动验收

交接文件 status=accepted，gpu_indices=[2,3,5,6]，四张 UUID 均匹配；启动前没有 compute 进程。

- systemd 服务：resnet50-noise-20261026-hkust4-163715.service
- 主 PID：3536189；GPU2/3/5/6 的 worker PID：3538281/3538282/3538283/3538284。
- 运行账号 shaoshitong（UID1001），sudo 创建服务；该账号可直接写入现有 SSHFS 持久目录。
- batch=2048，micro-batch=128，四卡、累积4次，随机224×224输入与随机1000类标签，BF16/SGD。
- 开始：2026-10-26T16:37:26Z；24小时停止期限：2026-10-27T16:37:26Z（洛杉矶 10月27日09:37:26 PDT）。RuntimeMaxUSec=1d，Restart=no，KillMode=control-group，停止宽限15秒。
- 16:38:03Z：全部rank step30，loss6.9262695，参数最大变化0.00758037。
- 16:38:15Z：全部rank step60，loss6.9313812，参数最大变化0.00633708。
- 每卡进程显存7256MiB，整卡7274MiB，利用率98–100%；四个PID与指定UUID严格对应，无其他计算进程占据这些卡。
- 服务active/running，NRestarts=0，无Traceback/OOM/NCCL错误。
- 日志：HKUST4 /weka_3/shaoshitong/lse-sparse-repro-20261026/shared/resnet50-163715/train.log。
- 持久路径：HKUST3 /backup/shaoshitong/lse-sparse-repro-20261026/resnet50-163715/；已从该节点独立读回step60和日志3544字节，源码备份SHA与本地一致。
- 源码commit：f261ee3f69ce5ec7a7c4fb2597785add72f2e7a1；状态工具commit：6dd99c8。
- 备份分支：shaoshitong/Search-R1 的 codex/resnet50-noise-20261026。
- 本次未修改/重启HKUST2/3训练，也未修改Wan代码、输出和挂载。

注：首次部署的副本校验输出被控制器重写了文件名，因此本地比较拒绝通过；未启动训练。已改为比较SHA列表并重新部署，旧源码副本保留。训练仅启动一次。

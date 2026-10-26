# HKUST4 随机 ResNet50

仅 GPU 2、3、5、6，固定 UUID。有效 batch=2048，每卡 micro-batch=128，累积4次。训练复用原 train.py，随机输入与标签、BF16、SGD不变。

启动前读取 shared/WAN_ACCEPTED_GPU_HANDOFF.json 且 status 必须为 accepted，再确认四卡没有 compute 进程，显存不超过32MiB且利用率0。不得只按空卡启动。

launch_hkust4.py 通过 sudo 创建独立 systemd 服务，RuntimeMaxSec=86400，KillMode=control-group，Restart=no；启动24小时后自动结束，停止宽限15秒。User=shaoshitong 用于访问现有 SSHFS，无需更改挂载。所有日志、进度写入独立 shared/resnet50-时间戳 目录，源码和配置备份在相邻的 -source 目录。其真实持久位置为 HKUST3 /backup/shaoshitong/lse-sparse-repro-20261026/ 下的同名目录。

部署入口 deploy_hkust4.py 核验本地、HKUST4本地运行副本、共享源码备份的 SHA256，并生成 deployment-hkust4.json。此部署不启动训练。

进度检查需读取四个 progress/rank-N.json、train.log 和 systemctl show，至少两次 step 增加且参数更新非零、loss有限，并对照 nvidia-smi 的进程 PID 与 UUID。尚未交接时保留 prepared 状态，不重复创建健康作业。

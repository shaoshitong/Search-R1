# 运行验收记录（2026-10-26）

本会话任务：HKUST3 的 4–7 卡、HKUST2 的 4–5 卡，随机输入和随机标签 ResNet50，batch 1024，运行 24 小时自动退出。

## HKUST3：已运行并两次确认真实更新

- 服务：resnet50-noise-20261026-hkust3-093218.service；root UID=0。
- 启动：2026-10-26 09:32:31 UTC。
- 自动停止期限：2026-10-27 09:32:31 UTC（北京时间 17:32:31）；systemd RuntimeMaxSec=86400，停止宽限 15 秒，KillMode=control-group，Restart=no。
- 09:34:30 UTC：四个 rank 均完成 730 次更新，loss=6.9549866，参数最大变化=0.0024404。
- 09:34:50 UTC：四个 rank 均完成 860 次更新，loss=6.9715576，参数最大变化=0.0015009。
- 每台有效 batch=1024，每卡 micro batch=64，累积4次；各卡利用率98–99%。
- 两次均 active/running，NRestarts=0，日志无当前错误。

## HKUST2：未启动，指定卡已被他人占用

- 部署目录存在且源码哈希通过验证；启动前空闲检查拒绝运行。
- 09:33:22 UTC 起，用户 zhuzeyu 的 score.py 进程 3851056/3851057 占据 GPU4/5。
- 两次查询均约23485 MiB/卡、100%利用率；未停止或修改这些进程。
- 093218 服务未创建，没有启动后台等待器或自动重试器。

## 初次失败记录

092855 尝试未完成迭代：HKUST3 旧环境分布式 rendezvous 崩溃，HKUST2 旧 PyTorch 对 GPU UUID 处理导致 Invalid device id。失败日志保留。
新版本使用已安装且验证 BF16 前反向的 Python3.10/PyTorch 环境，以及静态 rendezvous。没有重启其他健康作业。

## 源码与部署

源码 commit：71d69dc719140484a1bb1ef7e1c9a7d243a7ffe5。
备份仓库：git@github.com:shaoshitong/Search-R1.git，分支 codex/resnet50-noise-20261026。
文件清单与远端 SHA256 验证见 deployment.json；两次进度见 verification-second-attempt-1.json 与 verification-second-attempt-2.json。
控制器会将审计输出中的用户名和机器别名脱敏；deployment.json 中保留实际启动路径。

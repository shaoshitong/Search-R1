# HKUST ResNet50 随机数据训练

用户指定：HKUST3 GPU 4/5/6/7，HKUST2 GPU 4/5；通过 su 运行，24 小时自动停止。

- 每台机器独立 DDP 训练，有效 batch size=1024。
- 每卡 micro batch=64；4 卡累积 4 次，2 卡累积 8 次。
- torchvision ResNet50，随机初始化；每次重新生成 224×224 高斯噪声、0～999 随机标签。
- BF16、SGD；每次执行完整前向、反向、参数更新。进度记录参数实际变化和 optimizer step。
- systemd RuntimeMaxSec=86400，整个进程组自动终止，Restart=no。
- 不保存权重、张量或数据集，仅远端小型日志与进度；代码不包含登录凭据。

启动入口为 launch.py，必须通过 su 以 root 调用。启动前核验指定 GPU 无计算进程、利用率为零、显存占用不超过 32 MiB，并用 GPU UUID 固定设备。

状态：`systemctl show UNIT.service -p ActiveState -p SubState -p Result -p RuntimeMaxUSec -p ActiveEnterTimestamp`。

进度：读取远端运行目录的 `progress/rank-*.json` 和 `train.log`。

只有收到用户明确停止指令时，才通过 root 执行 `systemctl stop UNIT.service`。

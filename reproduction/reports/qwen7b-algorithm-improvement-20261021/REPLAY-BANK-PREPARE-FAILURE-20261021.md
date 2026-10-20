# 实际CPU准备失败：matched REF缺少bytes

已部署源码：`fd83d55e27416028bdc9232201c87f8ccbb1f5ee`。
隔离根：`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-matched9088-bank-20261021`。

本次按父任务指令，通过现有balanced-control只读读取该根的`cpu-result.json`、`cpu-controller.json`、`cpu-prepare.log`、`cpu-worker.log`及是否存在`prepared/progress.json`；没有修改远端目录、重启CPU准备、删partial或接触GPU任务。

- worker PID2140087/start_ticks122851739；child2140246/start_ticks122851810，父任务已确认退出。
- 实测`cpu-prepare.log`为1938B，完整内容SHA256：`d792465f70f716b2c3fafae8122e8fbe596c404ec2ce72276dc4e33efabc6e32`。
- 实测`cpu-result.json`为215B，SHA256：`829dca17e69ad22568f66bbd19801e5f8af8edc32443d4ac4dcf2fc2103dfb6d`，状态`CPU_REPLAY_BANK_PREPARATION_FAILED`。
- `prepared/progress.json`不存在。失败发生在读取matched config中的`plan`引用，尚未进入真实prefix扫描。

首要错误链：

```text
replay_bank_prepare.prepare: read(checked(c['plan']))
replay_bank_prepare.checked: checked_file(ref)
replay_probe_layers.checked_file: descriptor['bytes']
KeyError: 'bytes'
```

真实matched config的REF由已运行matched suite生成，采用`{path, sha256}`；旧qualification helper的REF采用`{path, bytes, sha256}`。新preparer错误地将旧helper的严格descriptor格式直接用于matched引用。这是本地新适配代码的schema错误，不是Slurm、GPU、模型数据或校内源故障。

本地修复仅修改新module：

1. `replay_bank_prepare.checked`保持remote绝对路径/无symlink限制，始终流式计算实际SHA；若descriptor带`bytes`则同样严格核对，缺少时不猜大小，使用本次实测文件信息。旧18c46aff模块完全不改。
2. 新增SHA-only引用回归测试，覆盖正确SHA、错误SHA及显式bytes不符三种情况。
3. `deploy_replay_bank`的CPU失败回执增加child returncode、实际log路径/字节/SHA及最后一条日志行，避免只得到笼统“prepare failed”。

21项CPU合成/closure/脚本/replay/reader测试通过，包含隔离代码包实际CPU import及quoted控制命令<16KB；最大仍8789B。测试没有读取真实模型或生成特征，也不表示新的远端CPU准备已经成功。

原失败目录及日志保留。下一步由父任务提交推送修复，再使用新的隔离attempt重新执行CPU准备；不可覆盖`code-fd83d55e2741`，不可将launch成功或上述合成测试计作真实9088题fixture验收。

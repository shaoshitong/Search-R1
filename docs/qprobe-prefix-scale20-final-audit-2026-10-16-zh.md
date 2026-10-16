# Scale20 prefix 数据最终收尾验收（2026-10-16）

入口 `reproduction/operations/a800_qprobe_prefix_scale20_final_audit.py` 只读核验实际完成的 0850 数据构建，固定 run `62407ad0ed404586998439ca1f55b902`、execution `dd3032e313e14ce2b9c8fea996aaf721`、runtime `fcb77204f48d5cb8ec21c68df7d4a2c937b3557a` 和原 operator `0277a4b0b5466b5f8767624b544c0001b47a89d6`。保留原 controller、worker、日志及失败历史，不启动或停止任务。

发布最新入口并核验远程源码后，在本地运行一次：

```bash
python3 reproduction/operations/a800_qprobe_prefix_scale20_final_audit.py \
  --handle /Users/shaoshitong/Documents/Search-G1/preparation/qprobe-prefix-scale20-dataset-deployment-20261016.json \
  --finished-receipt /Users/shaoshitong/Documents/Search-G1/preparation/qprobe-prefix-scale20-native-observation-1792140512182782000.json \
  --output /Users/shaoshitong/Documents/Search-G1/preparation/qprobe-prefix-scale20-final-audit-20261016.json
```

验收通过要求：200 个封存 chunk 共 102,400 训练题、512,000 轨迹；190 个新编码 shard（ordinal 10–199，每个 2,560 条）共 486,400 条，加原有训练特征 25,600 条；固定测试 512 题、2,560 条保持复用。逐一核对小型 manifest 的 SHA、源与科学协议绑定、文件大小和 producer 已写出的完整 SHA 证明；四个 encoder 的完整 36 层/layer27、无 optimizer、参数及 buffer 的 before/after 完全相等证明须一致。

这是 **元数据与 producer 完整 SHA 证明的复核**，不重复扫描约 300GB 特征，不打开 npy、tensor、模型权重或原始 rollout。compact 小文件的额外全字节校验由后续 HF export 过程承担。输出仅包含计数、标量与哈希，不含题目、答案或 token 内容。

同一次最长 65 秒的 SSH 读取还核验 wrapper、teacher supervisor、四个 encoder 和 owned ledger 的精确出生身份已消失；按已记录 UID/SID 扫描本任务 session，并用 NVML compute PID 判断本任务 CUDA 进程为空。其他任务可以继续占用 GPU，不发送信号。只有实际返回 `accepted: true` 的新 receipt 才表示本入口验收通过；本文件不预先声称已执行或已验收。

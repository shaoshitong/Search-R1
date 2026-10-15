# 已有 G5 文本轨迹 HF 交付（2026-10-16）

已交付至 [Sutoonq/Trajectory_in_Search](https://huggingface.co/datasets/Sutoonq/Trajectory_in_Search)，保持 **private**，原历史 archive 文件全部保留。配置为 `official_qwen25_frozen_g5_20261016`，实际 commit 为 `bf55ef1a67282772162994d0ff4ca0e21a086e42`。

| split | 问题数 | 轨迹数 | 二元 EM 正确数 |
| --- | ---: | ---: | ---: |
| train：原 teacher warmup | 5,120 | 25,600 | 5,627 |
| test：2026-10-16 新随机诊断集 | 512 | 2,560 | 559 |
| 合计 | 5,632 | 28,160 | 6,186 |

test 不是官方 NQ/Hotpot dev512。题目 ID/hash/归一化内容跨 split 隔离；逐题实际五条响应、唯一 trajectory ID、response_index 0..4、EM 二元值和 G5 correct_count 均已核验。

字段包括原文问题、实际 prompt、完整 response 轨迹（含检索 observation、继续生成及 special tokens）、full context、独立 policy-only text、最终原始严格 EM 及源身份/hash。不重新生成，不重算分数；mixed 描述一次实际 G5 事件。

CPU 导出在远端完成。两个 Parquet 总计 40,518,460 bytes：train 为 36,813,716 bytes，SHA256 `468d6e18fa5d03163b7c0e99e4b7cc5e97b616ad8fcae7c2c3fa68ce9fc60276`；test 为 3,704,744 bytes，SHA256 `9716c7113712f93f97a2c5e3d4ec1343289b855c61b3ab86bbbe9fa255a16993`。

上传后已核对精确 commit 下五个新增/更新文件的大小、LFS SHA（适用时），并从该 commit 重新流式读取所有文件，下载内容 SHA 全部一致。远端 exporter 已退出，源码 SHA 保持原值；上传进程实际 exit0。

A800 HF 直连不可用，实际使用 Mac **仅内存**中转约 40.52 MB；没有 Mac 数据文件、原始 tensor、模型或凭据落盘，无需磁盘 staging 清理。token 仅无回显 stdin/内存使用，没有 login 或认证配置修改。并行 scale20 teacher 及 GPU/检索任务未受改动。

实际源码为 `b44441fccff3a59d416f01645ddfbcf4f3de0411`，exporter SHA256 `e99d3b9d6535d6d06732ab6c5c6646a06c611a5c294cb3f350ce55ba6c96560e`。完整去凭据 scalar/hash 证据在同名 JSON 报告中。

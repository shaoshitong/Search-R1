# 已有 G5 文本轨迹导出至 Hugging Face（2026-10-16）

目标仓库为 `Sutoonq/Trajectory_in_Search`，新增配置为 `official_qwen25_frozen_g5_20261016`。仅导出已完成且不可变的旧预热 5,120 题及 2026-10-16 新随机 512 题，不导出正在构建的 102,400 题，不生成新轨迹，不重算 EM，不启动模型或训练。

每道题保留实际采样的五条响应，每条轨迹一行：train 为 25,600 行，test 为 2,560 行，共 5,632 个不同问题、28,160 个不同轨迹。train 正确 5,627 条，test 正确 559 条。test 是随机诊断 512 题，**不是**官方 NQ/Hotpot 各 256 的 dev512。

导出入口为 `reproduction/operations/qprobe_hf_trajectory_dataset.py`。CPU build 沿既有 SSH 物理路由在 A800 远端读取已封存原始文件，复用完整 SHA、G5 标签与轨迹 ID 行顺序、模型/采样/检索/评分协议、题目内容隔离证明。问题原文来自未改变的 canonical train.parquet，轨迹文本由官方 tokenizer 对实际 token 解码；attention mask 只去掉 padding，保留完整 search、observation、继续生成及特殊 token。另列 `policy_only_text`，仅此字段应用 sampling mask；它不能代替全文。

上传字段包括问题 ID/hash、response_index、trajectory_id、问题原文、actual_prompt_text、response_trajectory_text、full_context_text、policy_only_text、最终二元 EM、该实际 G5 的 correct_count/mixed、模型 revision 与源文件 SHA。mixed 仅描述本次五条响应的标签组合，不是问题恒定属性。不按类别重采样上传数据。

A800 直连 HF 已实测网络失败，因此使用入口的 `upload-memory`：只把封存 Parquet 和小型来源文档通过既有路由读入 Mac 进程内存，然后提交 HF；不下载 7GB 原始 tensor，不在 Mac 保存轨迹文本文件。设置 256 MiB 总 Parquet 内存上限。token 只经无回显 stdin 传入 HfApi，不 login，不放参数、日志、Git、配置、缓存或磁盘。未改变 SSH、VPN 或认证配置。

上传保留仓库现有可见性、文件及配置；新增两个 Parquet、来源证明、配置说明，并合并 README 的配置列表。不存在的仓库才默认创建 private。以原 commit 为 parent，避免并发覆盖；配置路径已经存在时拒绝盲目覆盖。上传后核对实际 commit、全部新增文件大小及 SHA/LFS SHA，重新从该精确 commit 流式读取并核验内容 hash。远端原始构建还读回 Parquet，核对二元 EM、正确总数、唯一问题/轨迹数、每题 response_index 0..4；HF 字节一致证明沿用这些行级核验。

全部大产物留在远端 `/data/shaoshitong/Search-G1-hf-trajectories-*`；Mac 仅保留源码、此执行说明和小型去凭据验收记录。此导出任务不停止或改变并行运行的 scale20 teacher、检索或 GPU 分配。

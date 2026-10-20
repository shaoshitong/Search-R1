# H100 peer new-seed G5：只读兼容性核对

本次来源是 h100-10 的独立新 seed 运行，绝非此前从 HPC3 逆向迁出的旧 lite 数据。

- 实际源：`h100-10:/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020`。
- 源读数 epoch 1792522051.895：244 个 sealed batch，31,232 问题、156,160 条真实轨迹；当前声明 raw 载荷 1,107,173,409 B。全量目标 169,028 问题尚未完成，不能将部分 sealed 快照称全源完成。
- 运行 70188b8e50fd4b45bc420ce58014f46e，seed 20261021、attempt independent-seed20261021-g5-20261020；官方初始 Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28；actor 更新为 0。
- 九个核心 sampling / interaction / strict-EM 源文件与本线程逐文件 SHA 完全一致，参考 frozen commit 82062fc12c6c0bdb50b9e8681e1af9e6e763af10。实际记录 capture Prefix20、question 上限 273、BF16、上下文 4096、action 500、start 2048、observation 500、最多 4 次搜索、保留 answer 500，temperature=1/top_p=1/top_k=-1。
- 此源没有 hidden tensor。`feature_layer26` 是配置声明，不能当成已有 L26 特征；导入后仍须用统一 causal replay 提取。保留真实 Prefix20、full response tokens/行为、journal 与同轨迹 own EM，可为后续 G10 配对提供独立新 G5。
- 来源 namespace 与原 seed1729/run bdbb550fd9324cefab979728acb0f5e1 以及本线程 extra-G5 run fe15f2bffd1a4287840c6ce20a311164 不同。未复用输出；未证明全部 31-bit 整数 request seed 集合无碰撞，不能声称随机数集合数学上完全不交。
- 原始 prompt token IDs 来自同一 canonical 训练输入；source ordered-qid SHA dd2a29d71bb36f918c46ab26932183c344e3b47b7f1b2bc2731c44ee111a7553（源 plain JSON 编码口径）。同 qid 的新 G5 应按真实轨迹 ID 与 own EM 合并，不能把共享问题误称未参与 warmup。

## 实际验证范围与限制

源独立 CPU auditor 当时对 244 batch 报 VERIFIED、0 failed；其缓存绑定包含 full/prefix/own-EM 的 SHA 与 journal/token 对应。我们另 fresh 核验四个已 sealed batch 的 prefix 与 own-EM SHA、qid/轨迹 ID、每题 0..4 五条二值标签和 prompt IDs；并核对其缓存文件 stat 仍匹配。我们没有在这轮重读全部原始 full 文件，也未重新计算所有 strict EM，不能扩大为全量新鲜语义复核。

检索声明为相同 E5/fullWiki18、top3 语义，但没有在这轮逐字节对比两端大索引/语料 SHA；必须保留此不确定点。H100 源 TP1 pooled/XFORMERS/vLLM0.6.3 与 HPC3 生成后端不同，协议兼容不代表 bit-exact 生成等价。

未发现该新 seed 的现成 HF/HKUST 接收回执；这只是已查看目录的结果，不是全 HF 仓库不存在的证明。迁移只接受新独立快照，源不清理。

## 本地小证据

- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/actual-launches/peer-newseed-G5-source-inventory.json`：29716 B，SHA256 `b9c9b4fddd35a7de8089c0e82a81f3b013e8c23631799c63466fdcf449b2f581`。

- `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-algorithm-improvement-20261021/actual-launches/peer-newseed-G5-code-contract-comparison.json`：2724 B，SHA256 `fc734b3754559dd6f1fe557a86393579e8c6c1c51ae98527e94a85af5cef6089`。

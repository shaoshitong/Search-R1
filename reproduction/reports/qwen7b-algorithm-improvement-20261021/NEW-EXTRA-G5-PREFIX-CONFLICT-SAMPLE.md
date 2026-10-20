# 新真实 G5 的 Prefix5/10/20 重复与标签冲突：封存小样本

实测时间：2026-10-21T02:43:02.175538+08:00。743115_0 的 shard0 首四个已封存批次（batch-000000～000003），512 道问题、2,560 条新真实轨迹；每题响应索引5～9。初始官方7B revision `a09a35458c702b33eeacc393d103063234e8bc28`，actor未更新。

这是固定 matched6528 训练 cohort 的前四个分片批次，不是169,028题的独立均匀样本。批次按既有身份顺序取样，不根据本次EM或重复率挑题，不能直接外推总体，也不能直接与旧批次或不同prefix口径比较。K=0…5题数为 `[193, 58, 55, 65, 59, 82]`；真实Mixed共237/512＝46.29%。

所有complete、prefix gzip与own-EM小文件先绑定SHA、读完再次fresh核验；通过trajectory ID、qid、prefix SHA逐一join own EM；同题五条prompt完全一致。未读取tensor、完整响应文件、原始旧G5、final或改动采样进程。

| 指标（明确分母） | Prefix5 | Prefix10 | Prefix20 |
|---|---:|---:|---:|
| 题目有至少一对相同prefix（/512题） | 460（89.84%） | 232（45.31%） | 32（6.25%） |
| 五条全部相同（/512题） | 33（6.45%） | 5（0.98%） | 0（0.00%） |
| 相同prefix无序pair（/5120对） | 1475（28.81%） | 494（9.65%） | 47（0.92%） |
| 相同prefix且EM相反pair（/1188个correct/wrong pair） | 307（25.84%） | 95（8.00%） | 9（0.76%） |
| 真实Mixed题含相同prefix/相反EM pair（/237题） | 146（61.60%） | 61（25.74%） | 7（2.95%） |
| token-exact轨迹BCE经验下界（nats/全部轨迹） | 0.110995 | 0.039834 | 0.004537 |
| 同一下界仅对真实Mixed题（nats/轨迹） | 0.239786 | 0.086055 | 0.009801 |

每题五条轨迹的 distinct-prefix 数量分布：

| distinct个数 | Prefix5题数 | Prefix10题数 | Prefix20题数 |
|---|---:|---:|---:|
| 1 | 33 | 5 | 0 |
| 2 | 101 | 20 | 1 |
| 3 | 177 | 69 | 6 |
| 4 | 149 | 138 | 25 |
| 5 | 52 | 280 | 480 |

下界按同题内 token-exact 簇计算：`Σ_c n_c H(k_c/n_c) / N`，其中H为自然对数二元熵，N为本样本轨迹数。它只是本观测样本中、仅依赖题目与这些前缀token的确定性单轨迹判别器的BCE最小值，**不是Mixed balanced accuracy上限，也不是当前神经探针能够达到的承诺**。若native hidden中包含不同工具上下文或位置，这个token-only下界不能不加条件地套给更丰富输入；本样本另核相同token簇的实际prefix上下文与position，在5/10/20三个长度均未发现差异。

本样本无短于20的自然终止轨迹；按真实未padding token tuple统计，未补造token。Prefix20的“重复pair仅0.918%”与Prefix5的“61.60%真实Mixed题至少有一对标签矛盾”可以同时成立：长度、分母和事件都不同。延长到20后，本样本该Mixed冲突题比例降至2.95%，但这不等于后续probe BA或Top512会自动提高，仍需真实训练和独立评估。

证据：`actual-launches/new-extraG5-prefix-conflict-aggregate.json`（仅聚合数和来源refs），SHA256 `9bc928493cf10a2de3cb36a8b5876aa49c2afc2c3ca67c2da90a035edd5d08fc`；只读控制脚本为同目录 `new-extraG5-prefix-conflict-diagnostic.py`。远端样本根目录：

`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/training-extraG5-matched6528-20261021/materialized/sampling/shard-00/batches`。

四份封存complete SHA：
- `batch-000000`: `d0878c1dc3cd2ae7e1d5cd8f018041cce676daa48dd2afaffe8be3b08547a75c`。
- `batch-000001`: `150af37e0dac4e9836331112bc44b73088a5ef5c49a88c6f8026918720016065`。
- `batch-000002`: `53340ab8eb1b787418e3a68857080897c443c9d1f0b0c526b3f826692e27a7c5`。
- `batch-000003`: `e450cdd12f448f60863c08978d6e4157fd4c019d2eeccc938dcb3e53d2eb33f1`。

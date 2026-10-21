# Direct-hidden count head：938 完整更新终局

本分支已完成原定 30,016 题、938 次真实更新和四个固定验证节点；科学结论为负。新 G5 同 HF 验证集上，Mixed balanced accuracy 为 **50.00%**，Top512 为 **183/512=35.7422%**，没有同时超过预先冻结的 E2 BA 56.2505% 与 E4 Top196 前沿。不晋升、不调外部阈值、不打开 final4096。

| 节点 | Mixed BA | Mixed recall | nonMixed recall | AllWrong FPR | AllCorrect FPR | AUC | Top512 Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | 49.8917% | 0% | 99.7834% | 0.0948% | 0.6061% | 0.526156 | 171/512 |
| 512 | 49.9639% | 0% | 99.9278% | 0% | 0.3030% | 0.543864 | 177/512 |
| 768 | 50.0000% | 0% | 100% | 0% | 0% | 0.550092 | 182/512 |
| **938** | **50.0000%** | **0%** | **100%** | **0%** | **0%** | **0.553854** | **183/512** |

固定验证池为2048题，真实Mixed663题（32.3730%），K0..5=[1055,181,136,132,214,330]。最终 Top512 的 K0..5=[217,42,33,40,68,112]；其中183Mixed、217全错、112全对。阈值固定0.5时，全部663道真实Mixed被判为nonMixed；全错/全对没有误报Mixed。排序仍有少量信息，但未达到既有同后端前沿。原512题诊断 BA=0.5、AUC=0.548169，单独保留，未拿它替换主验证池。

本分支使用真实Prefix20 hidden，经block27、可训练masked token attention、128维投影、G5对称mean/variance/max和六类count head。q为count分布中K1..4的概率和；本实验不使用五个p的乘积来定义Mixed分数。旧block/trajectory头与Adam自E1连续继承；旧Adam10464→11402，新head Adam0→938，二者LR均5e-6。每题四个冻结真实G5视图、各权重1/4，保持同30016/B32×938计划，没有新actor更新。

## 实际运行与验收边界

- 原743619在2小时时限前，按实际最高完整cursor770保存并正常退出；新743872从完整CP770精确续跑至938，Slurm主job **COMPLETED0:0、33m03s、ACD1-6**。无重置、无回退、无重复sbatch；旧743619的终态不替代新job终态。
- 首个恢复事务771通过真实双次九状态重放、独立完整CP771验收，并连续771..776后继续777及以后。最终938保存先于固定评估，评估前后九状态hash一致。保留历史四rank archive，当前续跑为一个rank；不声称四rank bitexact。
- 最终producer在自己的allocation内完成CP938全文件SHA及九状态CPU重载。正式终局外置验收运行于 **ACD-Login-2、stdlib-only**：只流式读完整文件SHA，核对producer latest/complete/eval的九hash、计数、archive、冻结ID/ownEM与q_direct，并独立重算指标；**没有torch导入或checkpoint tensor反序列化**。实际训练锁已空闲。
- 在此之前曾在login CPU运行旧外置auditor，mmap反序列化CP并完成了重载检查；该执行位置不符合大型tensor不得在login载入的边界。原始成功回执保留，没有改称allocation内独审，确切旧hostname当时未记录。随后新增独立terminal-inherited回执作为正式终局验收，继承producer allocation的重载证据，避免再次载入tensor。两份记录不互相覆盖。
- finite observer已在实际新job终态与complete938齐全后自然exit0；只本任务allocation正常结束，没有处理外来进程。所有权重、optimizer、原始预测及checkpoint继续留在远端。

## 固定产物引用

远端基目录：`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2`。

| 产物（相对基目录） | Bytes | SHA256 |
|---|---:|---|
| `checkpoints/cursor-000938.pt` | 3754035012 | `8b97518ef0fadf5591625f82abbfe2f39d5c80093f16cfe7eaf03701164fa2ce` |
| `complete.json` | 3371 | `6e3fe7e9869d3244e6460084a4ffbc3db5ccd4ab10e3962095ceb1541cc94ad4` |
| `evaluations/cursor-000938/complete.json` | 9477 | `fecd65fd2b1629d68485dc107c2bb92cb03dcaf21420c42d4ea5aed3dd27b246` |
| `evaluations/cursor-000938/selection2048FreshG5HF.jsonl.gz` | 280575 | `4d4be474873cd7f6a1d55b47bce6f006b10cbd7bf13cb5be926e236b1fd090fb` |
| `evaluations/cursor-000938/terminal-inherited-independent-audit.json` | 4952 | `6a8b730441f23b1acbc69b67bc1d52e246ec03999d312d14730f199b1ddb00e4` |

原始训练runtime固定`501cf80478b77f68c3e13a96d4659e3da0971b2e`，精确续跑operator固定`31e930ce8327078612c3ee9dd98aec2188a78826`，terminal stdlib验收源码`806e3bd487ba9ba7281118db33500cba33261200`。最终checkpoint、全部四个节点及所有原失败/恢复记录均保留。

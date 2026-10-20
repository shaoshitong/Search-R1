# 参考线程“70% Mixed@512”的只读口径核对

记录日期：2026-10-21。参考线程：`01a15856-0269-7ce0-a310-793566f280d4`，用户给出的 host 为 `remote-control:env_e_6ac8dfc1a2f08320aa24a9ad9f5e129b`。`read_thread` 可读取，返回标题为 **U-Search-G1 (3)**。本核对仅阅读该线程及其引用的本地代码、小型历史报告和本地 Git 对象；没有向该线程发消息，没有 SSH、Slurm 查询、远端文件读取或源码修改。历史记录未被穷尽，因此“未找到某证据”不等于断言该证据不存在。

## 可核实的真实口径

1. **70.0390625% 是跨五轮已选样本的汇总。** 2026-10-17 16:55 报告记录 run `54ac33dcebe64e62a11fe4177f41f116`、execution `a4a0c2a18cca40bc8c3850d40eed8554`：普通 outer 7/8/9/11/12 合计 **1793/(5×512)** 道真实 Mixed，并不是一批“2048 道独立问题中选512”。其中 outer 7/8/9 分别为 388/512、364/512、366/512，即 75.78125%、71.09375%、71.484375%；另一个同 run 报告明确这三轮候选各为 **5120** 道。outer 5 候选2560，outer 6 起普通轮改为5120。
2. **另一个 run 早期普通轮确有较高比例。** run `387d3f6e228a488f90c418e2ba47e28e` 的 outer 1–9 汇总为 **3570/4608＝77.4739583%**，逐轮标记 `predicted_Top512`、`actual_em_selection=false`。对应 P10 启动模板声明候选2560、选512、Prefix5、官方3B初始 actor 和第5 epoch EMA探针；模板的 `source_commit` 是占位零，不能将模板单独当成实际部署回执。该比例来自 RL 中变化的 actor 和训练候选池，不能当作固定独立2048题的评估结果。
3. **周期更新轮的100%被单独标记为真实EM优先选择。** 同 run 的 outer 10/20/30/40/50 记录 `true_EM_priority`：完整2560题G5中真实 Mixed 分别911/835/792/679/651道，均选出512道Mixed，fallback为0。这些100%不能当成探针预测正确率；也不应混入上述普通预测轮统计。
4. **普通轮是同一G5的前缀筛选及完整续写。** 在历史 runtime commit `75d536575593891876df339467da73b3c4cbdc7c` 中，`rollout.py` 明确声明 label-free prefix scan、continuation of the SAME G5 samples；续写按 trajectory/prefix/selection 身份绑定。已查看的普通轮报告未见事后按真实EM重排的标记。此结论是代码和历史小回执层面的核对，未重新读取或审计远端原始轨迹。
5. **历史高分另有训练集口径。** `qprobe-native-qwen-warmup10-096c-training-scalars.json` 明确 `metrics_scope=same5120_training_only_no_independent_validation_or_formal_rollout`。该报告复用25600条已有teacher轨迹、没有新生成轨迹；Top512从初始41.40625%升至后期100%，属于同5120训练题上的拟合，不能替代未见问题测试。

## 问题隔离：已经证明什么，尚未证明什么

历史 commit `75d5365` 的候选池代码要求 canonical train 为169028行，并排除固定512诊断题的 qid 与 normalized-question hash；它没有声明排除全部 warmup 训练问题。候选池来自训练语料，不能仅因“本轮新生成G5”就称问题本身未见过。

3B交接记录声明：原始 train/dev 按稳定来源ID及标准化问题hash隔离；完整合并 warmup 每epoch有 **445440个G5采样事件**，而且明确警告这不是445440个独立问题，也不是给169028题每题只生成一次G5。其 feature manifest 为 `e9ed944ed244513ab55bdd1b13aae745c5b7902228294fd62ef70ea349b6f246`，data manifest 为 `8ac391ba26466e9fdbe9220b157bec81294ead69a09a0b7c2c4994cc7873a2c7`。

**本次没有取得“70%对应候选qid/问题hash全集 ∩ warmup所有事件qid/问题hash全集＝空集”的逐qid交集证明。** 因此不能把这些70%记录描述为严格未参与warmup的独立2048题，也不能据此作出数据泄漏指控。尚缺该准确批次的候选身份清单、warmup覆盖交集审计、完整actor/采样协议绑定及未筛选候选Mixed基线。

## 与本7B结果的可比性和责任边界

本7B selection 配置明确为官方 `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`、actor更新0、独立NQ-dev来源8757题中冻结2048题、G5、按预测Mixed概率选512，角色为 **checkpoint-selection validation，非untouched final**。源文件SHA为 `10d5ee25e813f3e2092247a5d2797793dd47ef5853959989cb34bf80a110d607`。经过多次checkpoint比较后，这批2048题更不能重新宣称为最终未触碰测试集。

已封存7B epoch2报告：2048题原始Mixed为 **641/2048＝31.2988%**，Top512为 **182/512＝35.546875%**；同固定512诊断的Mixed平衡准确率 **53.5235%**，而训练平衡准确率73.8221%。Top512 Mixed数未较前一checkpoint增加，不能用选中轨迹EM提高来替代Mixed选题收益。

参考70%包含3B/训练问题池/在线变化actor/2560或5120选512，本7B是初始7B/NQ-dev/2048选512，选择比例分别可能为20%、10%和25%，不能直接作性能等价比较。**这些口径差异并不解释或免除本7B低平衡准确率与低Mixed选题增益的问题。** 本7B仍须继续检查训练和推理一致性、组内可辨识性、目标和采样机制，并使用预先冻结的新确认集验证改进；不能用本报告代替该诊断。

## 来源及本次本地SHA

下列文件只记录路径、大小与SHA，不复制原始rollout、概率向量或全量metrics。SHA是本次对已存在本地文件重新计算的值，不能冒称本次已重新核验远端。

| 本地绝对路径 | 字节 | SHA256 |
|---|---:|---|
| `/Users/shaoshitong/Documents/Search-G1/actor50-grpo-mixed30-20261018/reproduction/reports/a800-prefix5-hourly-20261017-1655.md` | 3047 | `a3555cd28311c78fbb23ca770d9549f29864b2a9390ae9198edb59577c50a273` |
| `/Users/shaoshitong/Documents/Search-G1/actor50-grpo-mixed30-20261018/reproduction/reports/a800-prefix5-i1-i6-restored5-through9-20261017.json` | 11132 | `9796bda6accef3a0a95affd93d18b1c7b8c3303773df777f322556c83246be50` |
| `/Users/shaoshitong/Documents/Search-G1/actor50-grpo-mixed30-20261018/reproduction/reports/a800-prefix5-p10-qa50-mixed1-50-2026-10-17.json` | 45913 | `5dca482eb7197fdf4625e2086efb545abe5dd2f834df060463f83ef55c3c5dc2` |
| `/Users/shaoshitong/Documents/Search-G1/preparation/3b-probe-warmup-handoff-for-7b-20261019.json` | 13727 | `9466479503147928e49a15ca208f2802e794d0e9520292a4183b82053e19dd9b` |
| `/Users/shaoshitong/Documents/Search-G1/repo/reproduction/reports/qprobe-native-qwen-warmup10-096c-training-scalars.json` | 124832 | `6ae4191fb2ca6bb35b79c1d7c8fdee405d69463e046dfec1f9923e231692d971` |
| `/Users/shaoshitong/Documents/Search-G1/actor50-grpo-mixed30-20261018/reproduction/prefix_selection200/a800-prefix5-p10-launch-config-template-20261017.json` | 7863 | `49116218c994836326da9ece33093e20a111be0871a0fe55ff1e03e77fd8487e` |
| `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-four-epoch-unseen2048-20261020/selection-config.json` | 3256 | `2ed21976cb09983ed855fa2fd8d3ac7e866c13067552427325cd1765653fce45` |
| `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-prefix20-balanced-rank1-20261020/epoch2-complete/REPORT.md` | 2040 | `e2eb282d4bf20ace2d81c06f0628db8b87032716350311a86be15a63f164acc4` |
| `/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/qwen7b-probe-warmup-20261019/reproduction/reports/qwen7b-prefix20-balanced-rank1-20261020/epoch2-complete/independent-prediction-audit.json` | 4301 | `1602133a0faa1dbac2600237075e20de58fe512ce6b85f60430e693a38f9ec52` |

可核实的历史commit与本地Git对象：

- 5120候选逐轮报告绑定 runtime **`75d536575593891876df339467da73b3c4cbdc7c`**。从本地 checkout `/Users/shaoshitong/Documents/Search-G1/actor50-grpo-mixed30-20261018` 读取该commit的 `reproduction/prefix_selection200/pool.py`：11385B，SHA256 `0e40d8ae51aa2599ffad0e4b4114efd8f863c0b03f8ec0d73ef224193033c712`，Git blob `d2530b76e1e6ef273904d846aad21369b59d8bfb`；同commit的 `rollout.py`：52741B，SHA256 `04604592bb4590a0108047d6c39251da73a5320d63d6c0cf50678b4b0b53422a`，Git blob `817ead2e0963a6c9fc606bb9dbe69235a407a351`。
- 另一run的1–50汇总报告声明 runtime **`0de9565c9b1c735635744f7e80a2526a2852e568`**；不能把不同run、execution或runtime的计数拼成同一次固定评估。
- 3B完整合并warmup交接声明code commit **`ddf36cca14ac7aafb0b5395ffa88d57ed8c35e74`**；训练集only标量报告声明source commit **`e81445905bf8b15dbb0bdec3afa29d99ab0b11d8`**。
- 7B selection配置保留plan commit **`99432e5d88ea2c42adc341f460438296556d8686`** 与sampling commit **`96df5a23f23f6da0f24e79e67dcf0e3039f22de0`**。本次只核对已有配置内容，不将其表述为新远端发布验收。

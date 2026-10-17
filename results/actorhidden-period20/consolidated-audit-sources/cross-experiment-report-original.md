# 新 Prefix 选题实验与早期 Easy Selection＋Soft Reward 的实现审查

审查日期：2026-10-17。范围：G1 Prefix5/P10/2560候选、Gi Prefix20 actor-hidden（probe LR=1e-5）、Co Prefix20＋virtual0。三个子审查并行执行，只读实际源码、运行配置、完成账本和现成轨迹；没有联系原实验agent、修改训练或启动新推理。完整轨迹留在远端，本地只保存源码、摘要与少量诊断片段。

## 主要结论

**已经发现实际训练／评测链路的不一致，并取得具体输出证据。不能把目前下降直接解释为 question diversity probing 选题失败。** G1最明确的问题是：训练保护检索外壳，评测原始截断检索文本；模型在评测中续写检索文本并补 `</information>`，随后正确答案仍被严格动作解析器拒绝。需要修复的是输入／交互协议并重新推理，不能放宽正式评分器。

此外，新G1/Gi的custom actor改变了loss归一化和安全监控调用；G1/Co的virtual0改变GRPO优势。它们都不是纯data selection。Co的训练／评测共用新预算，没有发现与G1/Gi相同的内部预算错配，且其第50轮只比旧Easy少10题。

## 真实源码与运行身份

| 对象 | 实际源码 | run |
|---|---|---|
| 成功 Easy Selection＋Soft Reward | `0553adec146537e07b84377ce31cba41a54f37cd` | `15bb15365346401da1d919e9cde8f813` |
| G1 Prefix5/P10 | `0de9565c9b1c735635744f7e80a2526a2852e568` | `387d3f6e228a488f90c418e2ba47e28e` |
| Gi Prefix20/LR1e-5 CP5分支 | `c22dbf86664bcc4da86aecd702fbaaea9c381551` | `a008d7ce983347bbb6e81051b25a7c4e` |
| Co Prefix20＋virtual0 | `478f46315147ebb0d4d1c556fcdd356c658e117e` | `e4cf46d47ae24434a177931df5530751` |

不能用旧baseline checkout HEAD或recipe默认FP32代替实际成功运行。成功Easy后续正式阶段和三个新实验训练均为BF16／FP32 master。Gi的1e-5是选题探针学习率，actor仍为1e-6。

## 同轮效果

固定验证集均为512题，NQ和Hotpot各256，文件SHA256 `3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb`。

| 实验，iter50 | 选中训练轨迹EM | 固定512题EM | 较旧Easy少正确题数 |
|---|---:|---:|---:|
| Easy Selection＋Soft Reward | 约49.60% | 40.82%（209/512） | — |
| G1 Prefix5/P10/2560候选 | 52.89% | 30.66%（157/512） | 52 |
| Gi Prefix20／probe LR1e-5 | 80.20% | 26.17%（134/512） | 75 |
| Co Prefix20＋virtual0 | 49.10% | 38.87%（199/512） | 10 |

训练EM来自选中题目上的随机G5，验证是固定题目greedy单轨迹；两者不是同分布测量。这个区别一直存在，不能仅凭80%对26%就证明实现错误。本报告的实现证据来自后面的实际token与代码路径。

## G1：已有逐样本证据的具体故障链

1. 训练通过 `context_budget.observation_tokens` 截断内容但保留 `<information>…</information>`。
2. 固定评测 `eval.legacy_observation_tokens` 直接取原始前500 tokens，常丢失 `</information>`。
3. 模型面对未闭合的检索文本，继续补检索句子、补闭标签，再输出answer。
4. `qa_protocol.parse_model_action` 明确拒绝answer前的游离保留标签。因此有answer甚至答案正确，仍被正式strict EM计0。

真实iter50例子（其余文本省略）：

```text
上一段检索文本在此截断：… In downtown Winnemucca

模型下一动作：
, US 95 splits from I-80 and continues north towards Reno.
</information>
<answer>
Fernley
</answer>
```

该条答案字符串匹配gold，正式EM仍为0。另一条 `Michoacán` 同样是补检索文本和闭标签后给正确答案。

现成512题输出的CPU解码统计：

| 诊断项 | 题数 |
|---|---:|
| 正式正确 | 157 |
| 严格解析未识别最终answer | 153 |
| 上述153中，实际含完整answer标签 | 139 |
| 上述153中，上段检索有开标签却无闭标签 | 119 |
| 上述153中，模型下一动作含 `</information>` | 106 |
| 仅诊断抽出终末单独answer后，额外匹配gold | 52 |
| 其中同时满足“检索未闭合＋模型补闭标签＋仅一个answer＋无search标签” | 50 |

这里52条全部来自上一段未闭合的观察；其中50条有完整双重外壳信号。50/512约9.77个百分点，与相对旧Easy的10.16个百分点差距接近。**这是已有文本中的机制证据，不是修复后推理成绩。** 诊断得到的209/512不能替换正式157/512：重新闭合输入会改变后续生成，仍须同checkpoint重跑。不能通过宽松正则接受嵌套search/answer来“涨分”。

G1训练iter50现成2560轨迹中，1054条发生观察截断；提前final触发数为0，natural-EOS终止仅1条。所以本次直接证据集中在观察外壳和格式，不能把该轮大差距归给提前final或EOS。

用同一个strict提取器核对，G1训练可解析answer为2551/2560（99.65%），固定QA为359/512（70.12%），不是用executor宽松answer终止标记冒充strict合法率。

对应小型证据：`g1/remote-qa-wrapper-diagnostic.json`、`g1/remote-trajectory-aggregate.json`、`parser-reproduction.json`。后者用真实解析器CPU复现了带游离闭标签返回None、单独answer可解析。

## Gi：同一故障链已由独立现成输出确认

Gi子审查在现有iter50训练／评测全8rank文件上远端CPU扫描，完整raw未落Mac。首次旧兼容握手连接在mgmt-host关闭，使用现有兼容连接重试后成功；未修改SSH配置。

| 指标 | 固定512评测 | 同轮训练2560轨迹 |
|---|---:|---:|
| 严格EM | 134/512＝26.17% | 2053/2560＝80.20% |
| 缺少严格可接受answer | 156 | 1 |
| 上一观察未闭合information | 131 | 0 |
| 模型自己生成 `</information>` | 124 | 0 |
| 未闭合观察→补关闭标签→给answer→严格拒绝 | **119** | **0** |

训练同轮855条观察发生截断，但外壳均保留。固定QA真实输出例如：

```text
PA) are not specified. Area code 700 is not assigned
to any specific geographic area. </information>

<answer>
Ohio
</answer>
```

**119条不是额外119道正确题。** 它证明输入协议差异大规模影响真实模型输出，不能直接加回EM。证据：`gi/remote-qa-train-behavior.json`，含远端文件SHA与6个精简片段。

## 不属于选题的额外实现差异

| 项目 | 旧成功Easy | G1／Gi | Co | 判断 |
|---|---|---|---|---|
| 检索观察与上下文协议 | 原始观察前500；legacy交互 | 训练保留外壳、总4096、回答预留500；QA却legacy raw500、无预留、模型上限4596 | train/QA共同采用新预算及外壳保护 | G1/Gi内部不一致；Co是相对旧实现变化 |
| 推理精度 | 最后成功训练及live QA基于BF16路径 | 训练BF16，独立QA FP32 | live trainer直接QA | 是执行差异，未测其分数贡献 |
| loss归一化 | localmicro2内token mean，再各micro等权累积 | local16整体policy-token mean；G1用virtual-local16保持该目标 | localmicro4 token mean再累积 | actor优化目标／长短轨迹权重改变 |
| KL与负优势tail监控 | 原native actor有before/after-minibatch monitor路径 | custom actor直调forward/loss/Adam，magnitude/ratio只诊断；仍有DualClip3、gradclip1、finite检查 | 保留原native actor体系 | 不能称完整保留旧防护；旧实际启用另见子审查证据 |
| soft reward grounding probe | CP50/100/150后刷新 | 固定round0版本 | 已核实CP50后刷新并在51–55采用 | G1/Gi后续reward机制不同；**不能用50之后缺刷新解释iter50差距** |
| 全对组virtual0 | 五条soft reward内部归一化 | G1有；Gi无 | 有 | 改变GRPO优势，不是纯selection |

长短轨迹权重例子：一个micro有100个policy tokens，另一个有1000。旧目标给两个micro相同权重；新整体token mean给它们约1:10。mini128、每outer20次Adam相同不意味着目标相同。

virtual0例子：五条全对且soft reward相等，旧GRPO五条优势均为0；加入虚拟reward0后，这五条优势约均为+0.408。它增加全对组的正向学习。这个设计可以单独研究，但不能和选题变化混在一起后称“只换data selection”。

Co已核实实际round50刷新：manifest指向同run的global_step_50，日志有BOTH_PROBES_FITTED／REFIT_ADOPTED，第48–50轮用round0，第51–55轮实际用round50，见 `co/reward-refresh-actual.json`。

## 当前未发现的错误与未证明的因果

- 未发现Co固定QA加载错误checkpoint；它直接调用live trainer。
- G1/Gi配置与账本绑定当轮actor路径；目前没有加载初始权重代替当轮权重的证据。
- 固定512题身份、strict EM核心及官方chat-template未见混用错误。
- Prefix续写保留原prompt、已生成prefix和其采样logprob，未找到“丢了前20/5 tokens”的源码证据。
- 没有证据表明Gi probe LR1e-5被误传给actor；actor仍1e-6。
- 账本每outer20次Adam；不能仅凭mini配置相同推断loss完全一致。
- soft reward固定round0、loss权重和virtual0的因果贡献尚未量化；不能直接宣布它们是下降主因。
- 旧Easy实际run manifest声明policy_safety_required=true、mode=inline_first_five；未逐条读取全部旧监控阈值和触发历史，因此安全路径变化不能冒充已实测归因。

## 修正和定位顺序（本次未执行训练／新推理）

1. **先统一G1/Gi训练与固定QA的工具交互实现**，尤其观察外壳、动作切分、EOS、context与回答预算；评分器继续strict。对已保存iter50同一checkpoint做协议交叉回放，保存逐题首次输入／动作分歧，旧正式分数保留。
2. 如果实验定义是“仅换data selection”，直接复用成功Easy的actor/reward/evaluator；selector仅返回题目身份和真实可续写状态，不重写actor损失。若因效率改变实现，需逐项证明mask、logprob、loss归一化等价。
3. G1/Co virtual0单列实验开关，与选题方法分开报告。Gi无需补virtual0来“对齐”。
4. 复原或明确记录旧micro目标、安全刹车、grounding probe刷新策略，逐项对照，避免一次改多个因素后归因探针。
5. 重跑验收同时看strict answer可解析率、未闭合观察数、模型续写环境文本数、EM、动作/搜索次数；保留原始输出远端，不能只看reward均值。

## 子报告

- [G1 Prefix5](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/g1/REPORT.md)
- [Gi Prefix20](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/gi/REPORT.md)
- [Co Prefix20＋virtual0](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/co/REPORT.md)

G1和Gi的发生率分别由各自原始输出独立统计，不能相互套用；Co没有取得同样的内部协议错配证据。

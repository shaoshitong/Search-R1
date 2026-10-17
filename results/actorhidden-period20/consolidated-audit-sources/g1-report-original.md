# G1 Prefix5 P10: selection之外的训练/评测审计

只读审计；没有联系训练agent、没有停止/热改训练、没有启动模型推理或GPU任务。现成远端轨迹只在远端流式聚合，Mac仅保存小型统计摘要。

## 实际运行绑定

- 新实验session `01a0fb1b-d3ff-74b3-aa13-49ca57871f92`；run `387d3f6e228a488f90c418e2ba47e28e`；当前execution `1bf0183c24c84abfa8a65c587089145d`；源码 `0de9565c9b1c735635744f7e80a2526a2852e568`。
- 实际输出 `/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-387d3f6e228a488f90c418e2ba47e28e-resume-1bf0183c24c84abfa8a65c587089145d`。
- 本地对应源码根 **R**：`/Users/shaoshitong/Documents/Search-G1/actorhidden-prefix5-p10-eager-resume27-20261017`。
- 对照最终Easy Selection+Soft Reward，而不是仅最初job3364：run `15bb15365346401da1d919e9cde8f813`；源码 `0553adec146537e07b84377ce31cba41a54f37cd`；实际输出 `/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012`。证据：`outputs/session-refresh-20261015-current/g1-mixed.json` 的 metadata/run-manifest及actual-updates。
- 对照源码从git精确提取到本目录 `baseline-0553ade/`；`baseline-source-proof.json`列文件SHA。不要将mixed-first-200工作树当前HEAD28aa93误作最终运行代码。

## 已确认差异

| 项目 | 最终Easy Selection | 新Prefix5 P10 | 判断 |
|---|---|---|---|
| GRPO优势 | 原G5样本均值/std，五条优势和约0 | 全对组额外在统计中加入虚拟reward0，真实五条优势整体为正 | **改变训练算法，绝非仅选题** |
| actor loss长短权重 | 每个micro的policy-token均值，再均匀平均micro（每卡micro2） | 完整virtual local16的policy-token均值，micro按token数加权；4physical模拟8virtual | **不同目标函数**；mini128/20Adam相同不能证明等价 |
| 训练上下文与终止 | 四次search-capable动作＋最后动作；无回答预留；原始检索首500token | 总上下文4096，预留500token回答；接近预算提前禁search；截断检索时保留information闭合wrapper | **不止selection改变** |
| 新实验内部训练/评测 | — | 训练采用上述reserve/wrapper规则，固定512却采用legacy无reserve/raw首500规则 | **确定train/eval协议分歧** |
| 自然EOS | 结束当前动作，episode主要由环境done决定 | 训练natural_eos直接结束episode，评测不以EOS直接结束episode | 确定代码差异；现有发生率很小，不足单独解释大gap |
| soft reward grounding | step50/100/150刷新，200轮实际记录使用round150版本 | 固定绑定初始round0版本 `6de29...:0:0dcad...` | 刷新策略不同；版本ID不同不单独证明初始权重不同，需head字节SHA对账 |
| 负优势尾部/KL刹车 | 原inline_first_five监控与阈值路径 | Dual-clip仍在；当前actor只遇非有限值才退出，ratio/KL幅度诊断不停止 | 不能说三项防护原样保留；需结合实际ratio数值判断影响 |
| dtype | 最终resume是BF16 compute/FP32 master，前5轮原作业FP32 | BF16 compute/FP32 master；新固定512 FP32推理 | **最终训练dtype相同**，不能归因新实验“变BF16” |

### 精确源码位置

- `R/reproduction/prefix_selection200/actor.py:414-447`：优势contract与虚拟0；`:451-475`依strictEM分类全对后启用；`:499-513`固定grounding版本。
- `R/reproduction/prefix_selection200/actor.py:532-545`：Dual-clip与micro token权重；`:790-813`实际反向；`:805`明确ratio幅度不停止；`:817-820`gradient clip1与Adam。
- 对照 `baseline-0553ade/verl/trainer/ppo/core_algos.py:111-153`：标准G5；`verl/workers/actor/dp_actor.py:537-590`：逐micro均值/累积；`verl/trainer/ppo/ray_trainer.py:1106-1108`：checkpoint后刷新grounding。
- `R/reproduction/prefix_selection200/rollout.py:187-196`：final_stage；`:247-278`：natural_eos/obs截断；`:596`500回答预留；`:674-676`temperature1。
- `R/reproduction/common8/context_budget.py:20-24`：提前final；`:27-50`保留闭合wrapper截断。
- `R/reproduction/prefix_selection200/eval.py:27-34`：legacy无reserve/raw首500；`:110-119`FP32/max_model_len4596；`:161-172`greedy参数；`:185-202`环境done决定续写。
- 对照 `baseline-0553ade/reproduction/common8/generation_behavior.py:125-127,200-225`：旧四搜索＋最后动作、原始观测及done。

## 已检查、未发现直接问题的点

- 新train与eval都调用同一 `reproduction/prefix20_online/rollout.py:30-34` prompt_tokens：官方 `apply_chat_template(add_generation_prompt=True)`＋`add_special_tokens=False`＋末2048原token；没有发现rstrip assistant header。
- 原RLHFDataset同样chat template；旧generation_behavior保留原始末2048；没有发现新增自定义prompt文本。
- strictEM的decode_segments、contexts、qa_em以及EcoRewardManager/signals公式源码在0553ade→0de9565 diff为空；contexts的prompt只影响grounding上下文，不改变answer提取。
- 新QA加载路径由 `R/reproduction/prefix_selection200/launch.py:427-431`明确传`actor['hf_model_dir']`，没有源码证据显示误加载官方初始模型。
- hook代码仅detach拷贝hidden+residual；lookahead token丢弃，不纳入policy；但采样分块/seed/eager/KV规则不同，尚无同输入同模型的行为等价实验，因此不能声称完全行为等价，也不能仅凭代码判hook改权重。

## 实际输出证据（不是只看代码）

远端只读聚合 `remote-trajectory-aggregate.json`：

| 样本 | 严格正确 | 结束为answer | natural EOS结束 | 至少一次观测截断 |
|---|---:|---:|---:|---:|
| train iter5，2560轨迹 | 1260/2560＝49.21875% |2552|7＝0.2734%|809＝31.60%|
| train iter50，2560轨迹 |1354/2560＝52.890625%|2551|1＝0.0391%|1054＝41.17%|

固定512 iter5：118/512＝23.046875%，有严格answer360/512，缺152/512＝29.6875%；至少一次非法动作37题。
固定512 iter50：157/512＝30.6640625%，有严格answer359/512，缺153/512＝29.8828125%；至少一次非法动作33题。

注意：训练集是偏置选择后的同问题G5、temperature1；固定512是独立题目greedy，不能用两列差值直接断言bug大小。iter50是P10真实EM选题边界，非普通预测Top512。自然EOS发生率很小，**不能把它当作大gap已证实根因**。观测截断非常常见，但截断本身不等于错误；问题是训练与评测采用不同截断/闭合规则，需以同checkpoint、同固定512的单变量评测隔离。

聚合字段`eos_before_last_turn`只检查native finish_reason='stop'，包括正常search闭合后EOS，**不能解释为非法EOS提前终止率**，故未用于上述结论。

## 同口径训练/评测的严格answer

新增 `remote-train-strict-answer.json` 用同一decode_segments/contexts提取器核对train50：**2551/2560＝99.6484%**存在严格合法answer，9条无answer；最后一次检索观测未闭合为**0/2560**，缺answer样本含模型补information闭合的为0。dev50同提取器合法answer只有**359/512＝70.1172%**。这消除了仅拿训练terminal:answer与评测strict answer_present混比的疑问；仍需保留选题/温度不同的说明。

EOS补充：first_action_cut的native_terminal由“raw最后token属于terminal_ids **或** native finish_reason=stop”决定，且在boundary_found=True时也可为True，不是只指完全没有action时EOS。当前已采聚合是训练器实际terminal_reason字段，不把所有finish_reason=stop（包含正常动作结束）当作非法提前退出。

## 现成QA文本定位：不是153题都“没输出答案”

`remote-qa-wrapper-diagnostic.json`在远端用CPU tokenizer解码现成dev50输出（无模型前向），并对照原512 gold：

- 153条严格answer缺失中，**139条实际上有完整answer标签**。
- **119条**前一段检索观测含`<information>`但没有闭合`</information>`；**106条**下一模型动作含补写的`</information>`。
- 正式评分仍为**157/512＝30.6640625%**。仅作文本诊断，去掉answer前缀后有105条恢复为合法独立answer，其中52条答案与gold相符；157+52＝209/512＝40.8203125%，**这不是修复推理后的成绩，不可替换正式结果**。
- 52条中全部52条前一检索观测未闭合、恰好一个answer、没有search标签；其中**50条同时明确补写`</information>`**。因此不是从嵌套search中抓取多个答案冒充正确。
- 具体已有样本：检索尾在`In downtown Winnemucca`处被截断，模型继续`, US 95 splits from I-80 ...</information><answer>Fernley</answer>`。另一个输出`ombra"" club.</information><answer>Michoacán</answer>`。两条answer内容与gold一致，但保留协议标签出现在answer前缀，`qa_protocol.py:24-27`拒绝，正式EM均为0。
- 这给出**输入观测截断→模型补完观测→严格动作评分拒绝**的现成证据链。但去前缀的诊断上限并不是单变量对照推理，不能断言修复闭合后必然追回10.16pp。

soft reward刷新补充：旧在CP50后才refit，其影响从51起；**不能拿刷新周期差异解释当前CP50差距**。新旧初始grounding版本名称不同也不能单独证明head参数不同，需比较实际head文件SHA。

## 建议修正顺序（本次未修改）

1. 先统一训练/评测交互状态机、检索观测处理和EOS约定；保留原结果，新命名重评。使用一个公共函数而不是两个近似实现。
2. 若实验声称仅data selection，把虚拟0、loss reduction、grounding刷新周期恢复最终Easy Selection的原设置；要保留时作为单独消融明确命名。
3. 用同一个已存checkpoint和固定512验证protocol差异，再判断剩余差距属于选择分布、过拟合还是优化。重评不能把不同checkpoint混在一起。

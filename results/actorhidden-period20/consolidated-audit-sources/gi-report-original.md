# Gi 会话 Prefix20 actor-hidden / probe LR1e-5 只读审查

## 审查对象与证据边界

- Session: `01a0fcbb-863f-7450-926d-db537f648265`。
- Current run: `a008d7ce983347bbb6e81051b25a7c4e`；job `3430`；base `/mnt/data/shaoshitong/search-g1-prefix-selection200-actorhidden20-lr1e5-20261017-a008d7ce9833-resume43-xformers`。
- 当前真实 state 指向 runtime commit `c22dbf86664bcc4da86aecd702fbaaea9c381551`。当前checkout HEAD虽已更晚，但本报告引用的actor/rollout/eval/launch/context_budget字节与c22提交完全一致，见 `source-provenance.json`。
- 对照早期成功 Easy Selection+Soft Reward 的 `0553ade` 源码及 run `15bb15365346401da1d919e9cde8f813`。baseline最后成功阶段不是FP32：native8 BF16 varlen / FP32 master。旧recipe的FP32默认不能代表最后成功运行。
- 本审查只读本地已回读的真实配置、源码、receipt；未启动GPU、未重跑评测、未改训练、未联系原agent。因果归因仍需同一checkpoint的协议交叉回放。

## 现有输出已证实的具体失败链（iter50）

只读在H100-8流式扫描现有8个固定512评测gzip和8个训练gzip，CPU tokenizer解码，无新推理，完整raw没有下载到Mac；文件大小和全文件SHA保存在 [remote-qa-train-behavior.json](remote-qa-train-behavior.json)。首次mgmt连接失败，使用既有兼容握手重试成功，没有改SSH配置。

| iter50证据 | 固定512评测 | 当轮训练2560轨迹 |
|---|---:|---:|
| 严格EM正确 | 134/512＝26.171875% | 2053/2560＝80.1953125% |
| 缺少严格可接受answer | **156/512＝30.46875%** | **1/2560＝0.0390625%** |
| 上一条观察含information开标签但无关闭标签 | **131** | **0** |
| 最后policy自己生成`</information>` | **124** | **0** |
| 缺严格answer但已包含`<answer>…</answer>` | **149** | 0 |
| 上一观察未闭合→policy补关闭标签并给answer→严格拒绝 | **119/512＝23.2421875%** | **0** |
| 曾截断观察 | QA旧协议固定raw500，见131例未闭合 | **855，但保持闭合外壳** |

这119例不是“完全没有回答”。真实示例：上一检索观察以`... (N`被500token截断，最后模型输出：

```text
PA) are not specified. Area code 700 is not assigned to any specific geographic area. </information>

<answer>
Ohio
</answer>
```

另一例：

```text
 Baylor, Jerry West, Wilt Chamberlain, Jerry West, Magic Johnson, Kobe Bryant, and LeBron James. </information>

<answer>
Los Angeles Lakers
</answer>
```

`search_r1/qa_protocol.py:24-27`见前缀含保留协议标签（`</information>`）时拒绝整条模型动作，`signals.py:20-22`随之返回None。这不是应当放松评分的理由：实际是训练与推理给模型的观察格式不同，引发模型补写环境文本，再触发合法的严格格式检查。

**119不是119道本应答对的题**；不能给官方EM直接加119，也不能宣称修复后必然恢复80%。80.20%是选题后的随机采样训练集，不可与固定dev greedy直接等价。但已有真实逐题行为证据，足以确认一个具体的训练/推理输入协议缺陷，优先级高于把低分笼统归因于选题探针拟合差。

## 已确认：不止改变 data selection

### 1. 训练和固定512推理使用不同交互协议（高优先级）

| 行为 | Gi正式rollout | Gi固定512评测 |
|---|---|---|
| 前向精度 | BF16 | FP32 |
| 上下文 | 总计4096，含后续动作和观察 | request可到4096，再生成最多500；模型上限4596 |
| 最终回答预留 | 500 token；上下文将不足时提前进入禁search final | 0；仅第5次模型动作禁search |
| 观察截断 | 先计算剩余空间，截断时保留 `<information>…</information>` 外壳 | 原始观察前500 token，可截断关闭标签 |
| 动作生成 | Prefix20暂停后原样续写，逐token观察首个完整动作并中止native request | 每次原生完整500 token结束后再裁到关闭标签 |

直接证据：
- `rollout.py:150-156` start_action调用final_stage；`rollout.py:208-216`观察空间与保留外壳；`rollout.py:479` BF16/max_model4096；`rollout.py:545` answer_reserve500。
- `context_budget.py:19-23`提前final；`context_budget.py:26-49`预留和wrapper修复。
- `eval.py:24-30`明确 legacy_pre_d00_async_fp32、无reserve、无early-final；`eval.py:33-36`原始观察前500；`eval.py:103-111`FP32/4596；`eval.py:154-157`greedy500；`eval.py:218-243`完整native结束后裁切。
- 当前实际qa5 state receipt中的 `evaluation_contract` 同样记录上述FP32/无reserve/原始前500，与源码一致，并非只有死代码。

例：request达到3200 tokens时，训练判断3200+500动作+500回答>4096，已进入final、不再执行search；测试仍允许search到第4次。另一个长观察，训练会补 `</information>`，测试可能没有该闭合标签。由此同一权重接下来的输入token及工具行为不同。这是确定的不一致；尚不能定量说它解释了多少EM落差。

### 2. soft reward 的 grounding probe 被固定为 baseline round0（高优先级）

当前真实config固定：
- `grounding_manifest`: `/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012/artifacts/round_0-attempt-22c55c34f646454f9c1f158ebdeb59c2/version-manifest.json`
- version: `15bb15365346401da1d919e9cde8f813:0:a72f2e62b8674787bee00266a8eef79e`
- SHA: `924e5487d19cf667dee56ae0a76eba7143cd6eb6325f1024989bac5d05efbb61`

`actorhidden-period20-resume43-xformers-config.json:53-55`实际配置；`actor.py:29-31`硬编码archive绑定；`actor.py:368-383`每轮创建EcoRewardManager并强制该固定版本；`launch.py:230-237`8个grounding服务从该manifest加载。没有调用common8 refresh_after_checkpoint的路径。

旧baseline `baseline-0553ade/reproduction/common8/runtime.py:406-433`在50/100/150调用refit，切换以当前checkpoint产生的新round probe；`ray_trainer.py` checkpoint后将reward_fn.expected_probe_version替换为新版本。**时间范围限制：baseline在CP50之后才刷新round50，因此固定round0不能解释iter50之前已经存在的Gi差距；它会影响51轮以后。** 因此公式相同不等于实际soft reward信号相同。Question-diversity head与grounding reward probe是两个不同模块，本差异不属于新选题探针自身拟合程度。

### 3. actor优化目标的长短轨迹权重改变（高优先级）

旧baseline本地mini16/localmicro2、8次累积。`baseline-0553ade/verl/workers/actor/dp_actor.py:547-587`对每micro先policy token masked_mean，再除accumulation8，属于8个micro token-mean等权平均。

新Gi全局micro128，即localmicro16，1次反传。即使OOM减micro，新 `actor.py:402-407,633-649`按micro真实policy token数加权，最终目标始终为本地16条轨迹的整体token-mean。目标receipt明确 `mean_of_eight_rank_local_minibatch_policy_token_means`，不是旧8个micro mean。

举例：旧同一mini前2条共100个token、另一个micro共1000个token，两个micro各占1/8；新目标这两部分按100:1000加权。长轨迹、搜索较多样本的梯度权重会增加。mini128/20次Adam相同不足以证明训练等价。是否这个差异导致性能不佳尚未经过对照验证。

### 4. 新actor未调用旧代码的KL/负优势尾部刹车（启用状态还需actual baseline配置确认）

旧 `dp_actor.py` `_run_minibatch`在safety_enabled时每次更新前后调用 `_monitor_old_policy`；旧`safety.monitor`包含old-policy KL、负优势高ratio尾部和log-ratio上界门槛。新 `actor.py:631-659`直接调用 `_forward_micro_batch`、`compute_policy_loss`、clip_global_l2、optimizer.step，绕过`DataParallelPPOActor.update_policy/_run_minibatch`。

新仍有DualClip3、gradclip1、finite检查，但`actor.py:643`明确 magnitude/ratio diagnostics never stop this run。`old_policy_kl`记录的是masked `-logratio`标量，不等于旧完整冻结batch上的k3与负优势尾部门槛。不能把“new仍import dp_actor”当成保留了旧安全更新逻辑。当前仅确认旧源码路径和recipe默认enabled，尚未在此子审查中拿到最终成功baseline的actual resolved safety配置及更晚撤销授权；因此不能直接称“丢失了旧已生效防护”。

### 5. 相同项及已排查未发现的错误

- actor初始化是官方Qwen2.5-3B-Instruct aa8e725，不是把probe模型当actor；source模型SHA在真实config中绑定。
- prompt生成 train/eval共用 `prefix20_online/rollout.py:31-35`，官方apply_chat_template(add_generation_prompt=True)，取最后2048原始prompt tokens；未发现train/test不同chat模板。
- `actor.py:355-365`是每题5条真实soft reward的unbiased sample std归一化，与旧GRPO correction1一致；当前Gi没有virtual-zero。
- `actor.py:368-383`reward仍来自EcoRewardManager，prefix正确性探针未直接作为reward；环境系数balanced0.6/no-search0.5/cost0.02与baseline一致。
- train strict_label和eval score_tokens共用 signals.decode_segments/contexts + qa_em.em_check；未发现把F1当EM或宽松答案当严格答案。
- `launch.py:250-258`每outer绑定上一个已完成actor HF路径；`launch.py:342-347`验证使用当outer actor HF路径。采样/评测并非加载official0常驻权重的明显路径错误。
- 旧和新最终PPO分母都是实际采样时sampling_log_probs，不能错误声称旧用HF重算分母。旧FP32阶段有HF parity gate；最终BF16 journal-only模式同样不要求该额外forward。
- 新最新本地rank账本94/95每rank20次Adam、1900累计/95轮、57outer warmup后1e-6；未发现“写了20个日志却只更新1次”的计数错误。

## 建议的定位顺序（尚未执行）

1. 不动正在运行作业，选已保存iter50同一个checkpoint，在同一512题用训练交互协议/legacy评估协议各一次greedy回放，保存逐题first-divergence token/action与截断位置；将协议本身影响从data selection影响中分离。
2. 用已有同批轨迹离线分别计算fixed round0与baseline匹配round的reward、优势方向翻转比例、全对组std和负优势tail；无需另采样，不先改新probe。
3. 同批token/同一actor分别比较旧micro2等权和新mini16 token-weighted objective及梯度方向，确认目标变化大小。
4. 原样审查旧安全刹车是否被用户后续显式撤销；若无撤销，新custom actor缺失原调用路径需作为实现偏移修复候选。不得未经授权热改训练。


## 可点击源码定位

### 训练与QA交互协议不同

- [4096total/500answer reserve](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/rollout.py:545)
- [final_stage guards search](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/rollout.py:156)
- [final_stage early when reserve insufficient](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/common8/context_budget.py:19)
- [wrapper preserving observation](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/common8/context_budget.py:31)
- [legacy pre_d00 noreserve/noearlyfinal](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/eval.py:24)
- [raw first500 observation](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/eval.py:33)
- [FP32 model length4596](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/eval.py:103)
- [BF16 model length4096](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/rollout.py:479)

### grounding reward probe固定round0

- [actualmanifestround0](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/actorhidden-period20-resume43-xformers-config.json:53)
- [actual versionrun15bbround0](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/actorhidden-period20-resume43-xformers-config.json:55)
- [each rewardmanager binds config version](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/actor.py:368)
- [services loadfixedmanifest](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/launch.py:230)
- [old refreshaftercheckpoint](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/gi/baseline-0553ade/reproduction/common8/runtime.py:406)
- [refit current checkpoint](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/gi/baseline-0553ade/reproduction/common8/runtime.py:424)

### 长短轨迹loss权重改变

- [tokenmassweight preserves local16 tokenmean](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/actor.py:402)
- [newminipolicymass](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/actor.py:633)
- [newlossweightedbymicrotokens](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/actor.py:649)
- [splitmicro2](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/gi/baseline-0553ade/verl/workers/actor/dp_actor.py:547)
- [8microtokenmeans equalaccumulation](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/gi/baseline-0553ade/verl/workers/actor/dp_actor.py:587)

### 旧KL与负优势尾部刹车未被新actor调用

- [customactorloop bypassupdate_policy](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/actor.py:631)
- [onlyfinitechecknotmagnituderatio](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/actor.py:643)
- [clipglobalnorm1thenAdam](/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/actor.py:654)
- [old before_minibatch monitor](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/gi/baseline-0553ade/verl/workers/actor/dp_actor.py:542)
- [old after_minibatch monitor](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/probe-training-eval-gap-audit-20261017/gi/baseline-0553ade/verl/workers/actor/dp_actor.py:597)


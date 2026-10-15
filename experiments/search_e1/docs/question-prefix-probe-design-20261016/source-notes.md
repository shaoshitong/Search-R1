# Question + 已生成5token：single-trajectory correctness probe 源码契约

2026-10-16。只读本地静态源码及已保存小型发布摘要；没有SSH、模型forward/tensor读回、rollout下载、实现修改、训练或外部线程消息。下面“可读取”表示**现有生产者代码保存了对应字段**，不表示本次实际打开/验证每一个远端数据文件。主任务应以实际原资产manifest完成逐行身份验收。

## 直接回答是否需要重新forward

- **取得原前5个token ID不需要重新生成或重新forward。** 原生成记录保留实际首轮prompt IDs、原生成 IDs、trajectory/run/policy snapshot及最终严格EM标签；从已完整保存的数据提取即可。
- **取得“prompt/question + 前5 token”的上下文hidden，现接口需要一次新的特征前向。** 当前缓存只存question-only layer27特征；vLLM生成接口只返回token/logprob/行为元数据，没有返回隐藏状态。旧question特征不含prefix，也不是可用于续算完整encoder的KV。
- 冻结encoder时可以每条prefix只前向/缓存一次，后续训练head的多个epoch复用；不需要重新采样25,600条完整轨迹。若未来生成端专门增加hidden capture或训练forward hook，可复用已有计算，但这是另一个实现合同，当前代码没有支持。

## 本地源码身份

现实现 `/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix`，publication source commit `fcb77204f48d5cb8ec21c68df7d4a2c937b3557a`。下列7文件fresh本地SHA均逐项匹配已保存publication.source_sha256：[发布记录](/Users/shaoshitong/Documents/Search-G1/preparation/qprobe-qwen-native-selection-metrics-fix-source-publication.json:1)。这是本地静态核对，非新的远端身份实读。

| 源文件 | SHA256 |
|---|---|
| `search_g1/qprobe_warmup.py` | `6950496796dfadaedc3418d2ad9bee269f5fb023aff8320bdf0a468141bcd617` |
| `reproduction/common8/generation_behavior.py` | `627ae042b0b68c9771cc5966d84afe11bd169e27ab6e8e489d27b3f15a0e0d8b` |
| `reproduction/common8/behavior_trace.py` | `b5912e7f764f9088c5360ccf6c82946dcd781c3fcaff875da5379767323b6020` |
| `reproduction/common8/sampling_capture.py` | `64181dc92baf737df4ea3f7592eae8a3f880ef175292cdaea8c13ed7474c5c0e` |
| `verl/third_party/vllm/vllm_v_0_6_3/llm.py` | `e5f4a67693ebf819edfca6f6939ff09281d281a00f2fcec72700e1cd4d6a3b68` |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | `15342e6aa59533e30122b80e1ba98bebd69072ff0263d2fc1a1e5561b76b1ed7` |
| `verl/workers/actor/dp_actor.py` | `139de8655f071b469bfbfc926d8973d2995740d2fa16481ac25898e2d92f173c` |

## 原资产能恢复哪些字段

### 生成ID、真实prompt及单轨迹标签

`qprobe_warmup`每chunk写 `rollout-%02d.pt`，内容包括完整 `output`、generation identity、contracts、frozen before/after、questionbindings；随后由 `strict_vectors(parent,output,tokenizer)`得到每题5个EM，组记录还保存五个trajectory IDs和policy snapshot。[保存及严格标签代码](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_warmup.py:224)。

因此名义样本量为5120×5=25,600**轨迹级**标签，target是该条完整已生成轨迹最终strict EM的0/1，而不是该组mixed标签。须按原trajectory ID和sibling次序一对一join，不能把组K>0给五条都当正例。25,600不等于25,600独立题，train/calibration/test须按qid/规范化内容分组，禁止五siblings跨split。

vLLM capture保存 `actual_request_token_ids`、`raw_generated_token_ids`、count、selectedlogprobs、finish/stopreason，要求request已经finished及n=1completion。[实际capture字段](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/verl/third_party/vllm/vllm_v_0_6_3/llm.py:32)。生产端校验实际request IDs等于提交的prompt，原response前count项等于raw generated IDs。[提交/校验](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/verl/workers/rollout/vllm_rollout/vllm_rollout.py:344)。

这里必须用token IDs截前5个，不能decode再tokenize、不能取5个汉字或词。第一轮若EOS/关闭行动边界早于5token，不能用之后的工具/forced文本或后续轮token补齐；应显式定义short-prefix处理、保留真实长度/mask。保留原模型生成的特殊token，哪些计入5必须预注册。最终有效样本量需据实际边界核数，不凭名义25600断言每条都有5个有效token。

### prompt、attention/position及环境边界

首轮请求采用初始input的真实左padding attentionmask去pad，再保留原2048-token prefix选择；后续request=prompt+先前response。用首轮 `actual_request_token_ids`最直接恢复实际完整模型prompt，不等同于question-only文本。[初始prompt](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/reproduction/common8/generation_behavior.py:153)、[每轮request](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/reproduction/common8/generation_behavior.py:203)。

最终output包含prompts/input_ids/attention_mask/position_ids/info_mask/sampling_mask/logprobs与behavior_turns；position按有效token从0递增，padding不占位置。[assemble](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/reproduction/common8/generation_behavior.py:280)。behaviortrace actor-row schema也明确保存原input/masks/positions/promptwidth，并核首轮request与actorcontext以及保留的原tokenprefix相同。[actor schema](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/reproduction/common8/behavior_trace.py:466)、[prefix一致性](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/reproduction/common8/behavior_trace.py:495)。

prefix重算可读取保存mask/position，或对已去pad实际首轮prompt+5 IDs按原规则重建0…L+4；不能搬整个轨迹的mask后让未来响应/obs继续可见。若要“纯原question文字+5 IDs”而非实际chatprompt，则属于明确不同输入表示，需重新tokenizer/模板合同，不应声称复现原actor状态。

## 当前没有可直接读出的prefix hidden

现fixed featurecache由官方冻结encoder对question-only文本编码，hook layer27输出，保存所有非paddingtoken；`model(...use_cache=False,output_hidden_states=False)`，利用hook收集question features。[question cache](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/search_g1/qprobe_token_features.py:72)。该缓存没有原prefix token、prompt全模板、每层KV或每条trajectory prefix表示；把现question feature直接拼5个embedding不等同于完整encoder的上下文hidden。

vLLM `_post_process_outputs`只返回 paddedtokens/logprobs和可选behavior；无hidden字段。[返回接口](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/verl/third_party/vllm/vllm_v_0_6_3/llm.py:206)。当前actor `_forward_micro_batch`用模型logits计算entropy/logprobs，最终仅返回这两项，没有向head传prefix hidden。[actor返回](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/verl/workers/actor/dp_actor.py:163)。因此不能说“生成已经算过所以现成hidden无需额外forward”；内部算过不等于已保留/可访问。

## 一次新的prefix特征forward应满足什么

推荐先在独立诊断采用冻结encoder、inference/no_grad、原首轮promptIDs+**仅已观察的**y1…y5，严格causal attention，选择明确定义layer/位置；输出detach并绑定featureencoder checkpoint、tokenizer、dtype/backend、模板、token前缀SHA及原label actor/protocol。

- 不含y6之后、完整答案、gold、未来检索内容或后续forced提示。完整轨迹正确性是训练**标签**，不作为输入。前5token恰好含答案片段时属于可见prefix信息，但要按实际早停规则检查，不能用未来结果选择一个有答案的“最有利prefix”。
- 如果同一snapshot下actor旧forward已经对完整序列做因果teacherforcing，理论上早期hidden不依赖未来token；但当前接口并未返回它，复用需透明hook、因果mask/索引、snapshotbeforeupdate与批几何数值验收。保守实现直接截短prefix，避免不小心pool到未来。
- 冻结官方encoder特征：稳定、可cache、head更新不改actor，但不是实际vLLM内部hidden。
- actor特征：若要行为actor的真实表示，必须用采样时冻结snapshot，而不是actor更新后再前向却仍叫原行为特征；原checkpoint/低精度/backend差异需记录。不detach或解冻encoder会改变训练范围与计算预算，不能继续称只训probe。
- 新prefix输入的最大长度不再由旧question-only244保证。实际chatprompt可到2048再加5；当前flatten readout固定244×2048不能不改合同直接接入。可用固定layer的prefix最后位置向量/明确的maskedpooling小head作为新结构对照，但本笔记不实施或擅选。

计算量是25,600个短prefix的前向，不是25,600条完整多轮轨迹重生成。token预算包含prompt+5，长prompt的prefill成本仍需实测；同题5个prefix重复prefill可用专门共享KV优化，但现缓存不能提供这种共享KV，不承诺免费或固定倍数加速。

## off-by-one：生成第5token与看过第5token不同

因果decoder在已经消费prompt+y1…y4的hidden上预测/采样y5；这个hidden**没有看见y5**。要获得“已观察5token后的state”，需消费y5，取其位置hidden（通常正用于预测y6）。

离线teacherforce `prompt+y1…y5`即可得到post-y5表示，不需实际生成y6。若未来在生成端capture，应区分“用于生成y5的hidden”与“输入y5之后的hidden”，记录absolute token position和已消费token数；不能把API第5个hidden tuple自动解释为post-y5。当前源码不返回hidden，无法用现有字段补出它。

## 生成5token→筛选→继续：现API没有完成这个合同

现common8批接口 `generate()`跑到请求结束才capture，其 `max_tokens`是完整行动预算，结束后可free cache engine。[完整generate及返回](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/verl/workers/rollout/vllm_rollout/vllm_rollout.py:344)、[freecache](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/verl/workers/rollout/vllm_rollout/vllm_rollout.py:435)。

已有async phase确实手动 `engine.step()`，但`advance()`忽略unfinished输出，只交付finished request；关闭时要求pending清空或abort，未提供“达到5token暂停、拿hidden给probe、按通过结果恢复同request/KV/RNG”的接口。[async adapter](/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/reproduction/common8/sampling_capture.py:398)。

因此：

1. **离线probe训练/验证**可用已完整完成的轨迹prefix和最终标签，不节省那些已发生的rollout成本。
2. **未来真实early screening**必须先生成至少5token才有输入，无法像question-only probe那样生成前筛题；筛掉的轨迹仍耗prompt prefill+5decode。
3. 将 `max_tokens=5`跑完再以prompt+5重发，不是自动复用原请求KV；会产生额外prefill，且RNG、stop、剩余行动/搜索预算、n和轨迹ID必须明确定义，不能声称bitwise延续原完整采样。
4. 要保证原续采轨迹，需要独立实现暂停/恢复与KV/RNG生命周期合同并验收；已有async engine.step基础并不等于已支持该功能。性能节省和模型收益均待验证。

## 目标与指标边界

新target是`Pr(该轨迹最终strict correct | question/prompt,observed prefix,actor/protocol)`，与原`Pr(一组G5mixed | question)`不同。单轨迹classifier可给五条分别评分，不能把它自动称为已修复groupdiversity。若要从五个prefix正确概率选择“mixed潜力题”，应另行定义组决策目标及剩余采样预算；有相关性的continuations不能未经检验套用独立概率乘积。

只有5token可能大量是`<think>`、换行或通用开场，内容差异是否足够预测最终EM应有question-only vs question+5token的同题holdout对照、按prefix唯一率/长度/格式分层。prefix标签仍是随机未来的单次观测，不保证能接近100%准确；也不能用这种不确定性忽略当前已证的std训练支路问题。

本笔记仅为来源契约与未来设计风险；未开启新训练、未生成新token、未改变已有probe、actor、采样器或运行状态。

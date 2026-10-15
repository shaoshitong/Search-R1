# Question + 实际前5 token → 单轨迹成功概率：设计审查

2026-10-16。只读Hugging Face官方文档/源码和已核原始概率评价论文；没有加载真实模型、访问训练节点、运行生成、重训probe或修改旧runtime。以下是独立设计审查，数学可行性不等于新增实测收益。

## 判断

这个设计有明确的概率目标：从题目和**这一条实际采样轨迹**的早期前缀预测其后续最终strict correctness。它比只看question多了当前actor的早期行动信息，可以成为有价值的下一步验证。它仍没有看到未来推理、检索结果和最终答案，不能称“已经知道该轨迹是否成功”，也不能将训练拟合接近100%视为预测确定性。

应先做独立offline proof：复用已经保存的实际generated IDs和原严格C，在独立目录里抽取question-only / question+5 / question+16，对同一批轨迹做概率泛化比较。初期不改正在运行的采样器、GRPO、既有probe或旧冻结runtime，不自动转为在线重训。

## 概率目标和数据单位

令 (x\) 是预测时原生prompt上下文（包含真实system/chat模板），(y_{1:m}\) 是同一条actual actor轨迹的前m个新生成token，C是其完整后续执行的strict correctness，则

\[
s_\psi(x,y_{1:m})\approx
P_\mu(C=1\mid x,y_{1:m},\text{actor snapshot},\text{executor protocol}).
\]

该C是**single-trajectory 0/1**，不是旧G5是否mixed的题组标签。相同qid的5条轨迹可能有相同的模板前缀而标签不同；这可以来自后续采样的随机性，不应强逼模型把所有单次C拟合到0/1极值。BCE的目标是条件成功概率，不是每qid永久成功标签。按概率评价的原始依据见[Gneiting & Raftery的strictly proper scoring rules，§3](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)；有限模型/有限重复数据不因此自动校准。

若有5120题×5轨迹，就是25600个**轨迹观测、5120个qid**，不是25600个独立新题。若数据也含多个actor快照，须保留actor和采样时点；固定teacher编码器只稳定特征映射，不会固定当前formal actor的条件成功分布。

## 官方API核查：hidden states不是刚选出token的表示

实际读了[HF GenerateDecoderOnlyOutput文档](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput)、[cache原理](https://huggingface.co/docs/transformers/cache_explanation)、[Qwen2 model forward文档](https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2Model.forward)。文档提供逐generation step的hidden-state tuple；cache保留先前token的K/V，后续只输入尚未处理token。它不保证第k个tuple就是第k个刚采样token的自身表示。

为确认顺序，读了文档所链接官方 **Transformers v5.19.0** 的[_sample源码2949–3002行](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/generation/utils.py#L2949-L3002)：先forward已有输入、取末位置logits、处理scores、保存本次hidden_states，然后才sample/argmax下一个token并拼进input_ids。本文只据此核标准非beam decoder顺序；**没有验证生产env就是v5.19.0，不建议升级生产环境来配合本设计**。真正实现时应固定已有部署的transformers/Qwen source SHA和对应返回语义。

设prompt最后token位置为L−1，生成新token为y1…y5，常规use_cache=true路径是：

| generate循环 | 本次真正forward的末token | 最后hidden用于预测 | 新选出的token |
| --- | --- | --- | --- |
| step1 / prefill | prompt最后token | y1 | y1 |
| step2 | y1 | y2 | y2 |
| step3 | y2 | y3 | y3 |
| step4 | y3 | y4 | y4 |
| step5 | y4 | y5 | y5 |

因此max_new_tokens=5返回5个step输出时，通常包含prompt及y1…y4的表示，**尚无y5自身处理后的表示**。第1step可能有完整prompt长度，后续通常为1个位置；不能简单stack成“5个generated token states”。speculative/beam/custom generate和提前终止需另审，不能按此表硬套。

需要y5自己的state时，可以将已选出的y5再做一次forward；这次不需要sample第6个token。若要从cache续算，必须按真实cache已处理长度喂入尚未处理部分，并保持总attention mask、position_ids/cache_position和padding对齐，不能默认generate返回cache已含全部5个输出token。应以token IDs/位置验收，不能用tuple数或字符串长度猜。

## 短prefix重算足够，不必整轨迹forward

对于固定参数、eval模式的causal decoder，在正确mask/position条件下，早期token表示只依赖它和前面的tokens，不依赖未来响应。因此可从已有完整rollout截取**原native prompt IDs + 原actual y1…y5 IDs**，一次causal teacher-forced forward拿到这些位置的states；不必把整条最终response、检索文档或答案输入编码器。这是causal依赖图上的结论，非保证不同kernel/cache/batch形状bitwise相等；实现验收仍需数值容差和相同token/位置身份。

推荐第一版离线特征接口复用已有prefix encoder，但输入只到预测边界。若它是冻结teacher编码器，应明确是“actor实际token + teacher重新编码”，不称“actor生成当时的native hidden states”。若取formal actor states，则绑定该actor权重身份，跨actor特征漂移另测；两者不能混用结果。

要明确所取“last block”是decoder block输出、最终RMSNorm后的last_hidden_state，还是LM-head前输入；它们未必相同。也要明确head收到prompt所有位置、question区域、5个prefix位置，还是pooling。添加5/16位置若同时改变flatten维度、参数量或截断question长度，会形成容量/输入范围混杂。先保持同一backbone/训练预算，记录各arm参数数目，必要时用等维pooling作为控制。

causal语义还意味着：只在输入末尾追加5token、却仍只取原question位置的states，不能让那些question states获得新增后缀信息；必须实际把新生成位置的states（或依赖这些位置的合法汇聚）传给probe。否则“+5”可能只是重算了同一question特征。

使用原生token IDs，禁止先decode再retokenize而不核对边界；token不等于词或字符。第一版把m定义为assistant第一次policy生成中的前m个native新token，包括真实special/协议token。短于m的EOS/终止以真实length和mask处理，不能填假token；不按未来是否成功选择哪些prefix入数据。若要改为“跳过think等模板后的5个内容token”，那是另一种预测时点与成本，应单独对照，不能静默替换。

前5token可能几乎只是`<think>`、换行和固定开场，甚至不含任何题目特异的决策；增加前缀并不保证增加有效信息。必须用matched数据比较0/5/16，而非只展示+5的train accuracy。

## 最小offline proof及评价

| arm | 特征 | label及其余设置 |
| --- | --- | --- |
| Q-only | question/native prompt，m=0 | 相同单轨迹C；同actor/样本/容量预算/优化设置 |
| Q+5 | 同一轨迹真实前5token | 同上；不含第6个及以后信息 |
| Q+16 | 同一轨迹真实前16token | 同上；额外前缀成本另报 |

原question-only **mixed-group** probe不是这里的matched Q-only baseline；目标从mixedY换成singleC也会改变任务，不能把旧指标直接相减来归因前5token。

训练/validation/test首先按qid分割，5个siblings及重复采样全部随qid；去除规范化重复/近重复泄漏。另列seen-qid的fresh realization测试，回答是否泛化到同题的新prefix/continuation。最终在较晚formal actor上做时间外评估，明确probe冻结还是允许多少refresh；不能把同一旧actor下random row split称current-actor泛化。

主指标为heldout single-C Brier/logloss，辅助AUROC/AUPRC、实际accuracy及预测概率校准分箱；按qid聚类估区间/差值，报告类别比例和常数概率baseline。两个概率模型都可能accuracy相近而NLL/calibration差很多，[Guo et al.官方ICML论文§§2–4](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)支持将这几个指标分开；校准验证假设也不能跨actor默默沿用。

校准或early stop只用dev，不用测试label/E0筛选结果调模型。问题prefix可以暴露有用轨迹信息，但禁止参考答案、future observation、最终stop status等进入预测特征。标签来自以后真实结果是合法监督，特征提前使用该结果则是泄漏。

如果最终仍按**题组diversity**选题，5个conditional成功概率不是直接平均就等于mixed概率。仅在给定5个实际prefix后各自continuation条件独立时，

\[
P(\text{mixed}\mid x,y^{(1:5)}_{1:m})
=1-\prod_{j=1}^5s_j-\prod_{j=1}^5(1-s_j).
\]

这不是 (1-\bar s^5-(1-\bar s)^5\) 的一般恒等式，也不是无条件qid mixed概率；共享检索/执行状态导致依赖时需joint验证。若只选高s轨迹，可能得到全对group而减少mixed；成功预测与diversity选题不是同一排序目标。

## 排名成本与KV不能忽略

若候选池5120题、每题5个prefix、每prefix实际生成5token，则是

\[
5120\times5=25600\ \text{prefix requests},\qquad
5120\times5\times5=128000\ \text{generated prefix tokens}.
\]

只把最终Top512题×5=2560条完成为full trajectories，candidate prefix请求量是最终完整轨迹请求量的10倍。这里**不是10倍整体训练耗时**，因为prefix很短且可能共享prompt prefill；但每个候选仍至少要处理prompt，不能只按128000短token估FLOPs而忽略prefill/调度/teacher重编码。可优化prompt cache共享、批处理或复用已有token，但都需要实际backend支持和测量，不能先假定免费。

KV的量级对普通GQA decoder可按每条序列近似

\[
B_{KV}\approx2\,n_{layers}\,n_{kvheads}\,d_{head}\,L\,b_{dtype}
\]

估算，乘实际保留的candidate序列数；分页/共享prefix可改变实占。若为全部25600候选保留KV以便选中后续算，不能承诺能放下或不影响formal训练。若丢弃KV后给选中轨迹重prefill，必须计入重复计算；若先生成prefix再从头fresh生成另一条完整轨迹，prefix的single-C预测已不绑定那条新轨迹，不能沿用原概率声称筛选有效。

正确的在线conditional continuation还需保留选中prefix的native IDs、已消耗生成/turn/search预算、executor state、必要随机状态及原processed sampling配置。继续生成可以保持条件分布，但仅固定seed不自动保证停/续与原无中断批处理bitwise相同。为拿hidden_states开启“保存所有层/全部step”也会额外占内存，不能把它当零成本接口。

第一版offline proof不用这些在线调度改动：已有rollout里slice前缀、冻结encoder、matched0/5/16；通过heldout/currentactor概率增益后，再单独做小规模prefix-candidate ranking与random对照，实测prefill/decode/encoder时间、峰值KV、吞吐和每GPU秒mixed收益。未通过前，不自动接入旧runtime或进行在线autoretrain。

## 验收界限

- 数据身份证明qid、actor、trajectory、原native prompt/prefix IDs、C一对一绑定，prefix不含未来数据。
- 表示边界证明y1…y5自身位置对齐，不能把预测y5的h(y4)误标为h(y5)。
- matched Q-only / +5 / +16 的qid-heldout及actor外proper-score差值，而非只报固定标签99%。
- 排名若实施，需同候选池、同actor、同完整生成预算random baseline与qid层CI，并报告prefix额外成本。

本次仅完成文档/源码语义与数学审查，没有运行上述proof，也没有声称5token方案已经提高正确率或question-diversity收益。

# Co Prefix20 + virtual0 vs Easy Selection + Soft Reward — 只读源码审查

审查对象：Co job727586，run e4cf46d47ae24434a177931df5530751，训练源码478f46315147ebb0d4d1c556fcdd356c658e117e。成功旧mixed-first按实际源码0553ade比较，未用旧checkout HEAD默认FP32配置做虚假差异。文件字节对比见source-comparison.json。依据已有2026-10-17午间远端实测快照与当前handoff；本审查未重新训练、未发消息给原agent。

## 结论

**Co并非仅替换data selection：all-correct virtual0确实改了GRPO优势。** 但当前没有证据证明Co的固定512评测读了错误模型/错误题集，也没有证据证明Prefix20被丢弃或logprob遗漏。Co dev50 199/512=38.8671875%，旧Easy dev50 209/512=40.8203125%，仅差10题，远小于Gi与G1的差异；不能用同一根因概括。

## 已证实的非selection差异

1. **GRPO全对组virtual0**。[search_g1/prefix20_advantage.py:25](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/search_g1/prefix20_advantage.py:25)：对strict EM五条全对的组，把soft score向量r改成baseline=[0,r1,...,r5]，真实五条A=(ri-mean(baseline))/sample_std(baseline)。旧实现仅在五条内标准化。若五条soft score相等，则旧A=0，新每条A=1/sqrt(6)≈0.408248（epsilon略减）。即全对组变成额外正向模仿梯度，不是纯选题；soft score不等时也改变优势符号与均值。路由已接入真实训练：[verl/trainer/ppo/ray_trainer.py:1098](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/verl/trainer/ppo/ray_trainer.py:1098) 和 [verl/trainer/ppo/core_algos.py:131](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/verl/trainer/ppo/core_algos.py:131)。
2. **同步完整动作改为独立async请求，选中后续写同一prefix**。训练[search_g1/prefix20_collection.py:62](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/search_g1/prefix20_collection.py:62)先5120×5、暂停20policy tokens，选512题再继续原状态。它是按实际采样前缀选择题+轨迹联合事件，旧mixed-first按完整G5真实EM过滤；这会改变受训轨迹条件分布，不仅改变题目列表。推理无prefix selection。
3. **总context与检索文本裁剪实现变化**。旧0553ade不存在context_budget.py；新[reproduction/common8/context_budget.py:18](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/reproduction/common8/context_budget.py:18)加入4096 total、500 answer reserve、提前final；新[reproduction/common8/context_budget.py:29](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/reproduction/common8/context_budget.py:29)保留information闭合wrapper裁剪内容，旧对batchedtokenized observation前500直接截取。新Co训练与QA均调用同一protocol/observation_room，暂无内部预算冲突证据；这一改动可能改善或改变行为，不能直接宣称导致降分。
4. **micro归一化**。旧实际native8/BF16每卡micro2，新native4/BF16每卡micro4，全局micro仍16。loss为每个localmicro内token masked_mean再平均（[verl/workers/actor/dp_actor.py:682](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/verl/workers/actor/dp_actor.py:682)、[verl/workers/actor/dp_actor.py:727](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/verl/workers/actor/dp_actor.py:727)），因此长短轨迹实际权重取决于micro拼组。globalmini128、ppo1epoch、每outer20Adam一致；它不是严格逐轨迹等权，也不能因全局micro一样断言数值目标完全一样。

## 已排查，未见错误的路径

- 固定512题：validation_identity.py与旧0553ade字节相同；实际validation.parquet SHA=3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb，NQ256/Hotpot256；QA评分qa_em.py、signals.py与旧字节一致。
- QA权重：[verl/trainer/ppo/ray_trainer.py:452](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/verl/trainer/ppo/ray_trainer.py:452)使用同一个live trainer/actor_rollout_wg，不是HF导出后重载；selector仅fit collect分支，_validate从未调用PrefixCollector。故导出遗漏/错误checkpoint不能解释当前inline dev50。
- 贪心区别：QA do_sample=False、1条；训练T=1、G5。这是旧Easy已有的差别，不能把它当新发现。
- Prefix没有丢：[search_g1/prefix20_generation.py:106](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/search_g1/prefix20_generation.py:106) suffix请求=prompt+已完成response+current_raw，剩余max_tokens=500-len(current_raw)；[search_g1/prefix20_generation.py:134](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/search_g1/prefix20_generation.py:134)拼接native logprobs并使所有modeltokens mask=1、环境mask=0。[search_g1/prefix20_split.py:70](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/search_g1/prefix20_split.py:70)逐chunk验证suffixrequest恰好=request0+raw。尚未观察到破坏此契约的真实trace。
- 原生sampling logprob追踪、masked概率比verification模块behavior_probability.py与旧0553ade字节相同。split新增独立源码验证，不等于已用GPU证明同一个随机seed分块与不分块逐token完全相同。
- 初始化官方Qwen2.5-3B-Instruct aa8e725；新不是从较高EM旧actor恢复。新/旧实际均BF16+FP32 master；不要拿旧recipe默认FP32当旧实际训练精度。
- grounding/soft reward probe刷新：Co训练代码仍在50/100/150刷新（[verl/trainer/ppo/ray_trainer.py:1240](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-prefix20-hpc3-4gpu-20261016/verl/trainer/ppo/ray_trainer.py:1240)）；reproduction/profile.py与旧字节相同。已进一步实际读取live manifest与不可变round50 manifest，均绑定本run global_step_50；model3片/index/config哈希与complete.json五项相同（不读权重）。日志6505/6506记录BOTH_PROBES_FITTED及REFIT_ADOPTED；actual-updates第51–55轮确实使用round50 version。详见reward-refresh-actual.json。

## 当前真实行为线索与待证伪

已有outer61实测：selected mixed fraction=0.11328125；selected hard EM=0.5109375；训练avg softreward≈0.768，advmean≈+0.184、mean absoluteadv≈0.272；actions≈2.001、search≈1、finish=1。优势正偏与virtual0机制一致，但需要逐组A/score/length分解才能量化占比。该步probe own-label BCE≈0.695、acc≈0.514只是分类器信息，不能用它替代GRPO/评测链路的排错。

优先用已有远端raw artifact只读核对：同一dev50逐题模型输出/检索动作与旧Easy dev50；selected allcorrect组的softscore和A正负质量；50refresh fit manifest sourceactor/encoder；prefix native chunk边界前后token/position/logprob与actor mask；按micro2与4对同一batch重算CPU权重（无需新GPU评测）。若要归因virtual0或context改动，需要单变量对照；当前只读证据不能证明它们导致38.87%。

## 补充实际验收：grounding reward probe 的50轮刷新

Co确已刷新并使用：version=e4cf46d47ae24434a177931df5530751:50:cf4a6e021ed5484e85b811f27f7cdb84。48/49/50轮使用round0；51–55轮使用新round50，每轮20次actor Adam。live manifest与round50/version-manifest.json一致，encoder指向同run actor/global_step_50；complete.json的config/index/三片模型SHA元数据与manifest一致。calibration 300独立题、2-fold，commit AUC0.8017537559658089、boundary AUC0.7340782288698955。这是实际远端只读证据，不是仅源码推断。未读取任何权重或启动推理。

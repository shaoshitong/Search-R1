# 非选题部分实现差异审计

2026-10-17，北京时间。目的：核对当前实验是否只修改了数据选择；不是因果归因实验，也不改动正跑。

## 比较对象与实际证据

- 当前逻辑run `a008d7ce983347bbb6e81051b25a7c4e`，3430/h100-8，冻结训练源码 `c22dbf86664bcc4da86aecd702fbaaea9c381551`。后续审计提交不是部署。
- 历史纯Soft Reward common8基线run `db474b6abf654ad9892b02ae67ab6f86`，实际manifest绑定源码 `fe59dc164490098054879141d5421e0b418b4056`。来源：`/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/latest-iter50-g1-current-raw-20261008-1954.json` 内原 `logs/run-manifest.json`；该观测发生在50轮之前，不能据此伪造基线50轮分数。
- 历史实际精度已是BF16 policy/FP32 master，且使用sampling-journal-only分母。不能只读recipe默认FP32就认定两者精度或PPO分母不同。
- 当前QA50实际512题、8worker exit0，EM134/512=26.171875%，F1 .32375104279401157。见 `qa50-checkpoint50-readback.json`。这是QA指标，不是探针混合分类准确率。

## 确定差异一：actor loss归约改变

旧actor `verl/workers/actor/dp_actor.py`：每2条轨迹micro内对有效policy token取masked mean，再对micro等权平均。当前 `reproduction/prefix_selection200/actor.py:630` 起：每卡16条轨迹构成local minibatch，按整个local minibatch的policy token总数归约；发生micro拆分时还按token质量加权，继续保持这个新目标。

对同一卡16条、8个二行micro，记第b个micro有效token数为T_b、逐token PPO项为ell：

```
L_old = (1/8) * sum_b [sum_t ell_bt / T_b]
L_new = sum_b sum_t ell_bt / sum_b T_b
```

T_b不相等时两者不等价。增大物理micro/noaccum可以保留旧数学目标：一次forward16条后，按原二行分组计算masked mean再等权归约，无需为了保持原目标增加模型前向。当前实现没有这样做。

来源：`4e480eb6c2c74c53d2a5ea05b44d32eb05c144b0`（2026-10-16 19:36，北京时间）新增独立actor路径；`af3b7f399`（同日19:45）显式固定local16 token-mean并为OOM拆micro按token质量加权。当前binding直接声明 `mean_of_eight_rank_local_minibatch_policy_token_means`；历史manifest声明 `original two-row micro masked_mean`。

上述是数学目标的确定差异；尚未完成当前实际轨迹的权重变化量读回，未测梯度或EM因果贡献。

## 确定差异二：训练交互协议改变，且与QA回退协议不同

旧 `reproduction/common8/generation_behavior.py`：原始500 token动作、原始前500 observation token，不预留答案预算，仅ordinal4是最终动作；request可4096再生成500。

当前 `reproduction/prefix_selection200/rollout.py:545` 写死 `total_tokens=4096, action_tokens=500, answer_reserve_tokens=500`，调用 `reproduction/common8/context_budget.py`：

- 截断检索内容时保留information闭合标签；
- observation按剩余答案预算进一步裁剪；
- request长度+500动作+500答案预留超过4096时提前进入最终动作。

因此实际可见检索内容、搜索机会和终止时机都可能改变。当前QA模块则显式使用用户要求的旧FP32/noreserve/noearlyfinal/raw500obs协议；存在确定的训练/评测交互协议差异。不能仅凭静态差异认定它解释了多少EM下降；提前final/observation裁剪实际命中次数尚待远端只读统计。

来源：`d00e4f8811cb4a2119c7d088280f7a97cd149d7e`（2026-10-14 20:38）新增context_budget；`29d81af65`（2026-10-16 19:33）新增prefix rollout时继承该规则。旧QA回退没有同时回退这个独立训练入口。

## 确定差异三：奖励用grounding探针刷新流程缺失

必须区分选题正确性探针与Search-G1奖励中的grounding探针。

旧common8训练路径在checkpoint50/100/150之后调用 `refresh_after_checkpoint`，用新actor重拟合并切换grounding版本，供后续轮次使用。当前launch仅启动配置中的固定round0 grounding manifest，后续只更新选题正确性探针，没有调用旧grounding refit流程。

当前配置版本固定为 `15bb15365346401da1d919e9cde8f813:0:a72f2e62b8674787bee00266a8eef79e`。启动路径来源 `e6cc96705`（2026-10-16 19:47）。这与基线规定的奖励探针生命周期不一致。该差异从51轮后的奖励尤其相关，**不能单独用它解释QA50之前的训练表现**。历史基线50后的实际refit成功与否，不由本次旧manifest快照证明。

## 已核对未改变的部分

见 `nonselection-static-audit.json`：

- `EcoRewardManager`整个类AST与基线相同，奖励公式未被选题探针替换；
- `core_algos.py`、`search_g1/signals.py`、`qa_em.py`逐字节相同；
- G5奖励组内样本标准差标准化、LR1e-6/57完整outer warmup、mini128/每轮20次Adam、epsilon.2、DualClip3、gradclip1、reference KL与entropy系数0保持；
- 历史实际BF16和采样日志分母已由真实manifest确认，不应将这两项误报为新发现。

## 结论与修复边界

当前结果不能被描述为“除数据选择外完全等价于旧Soft Reward”的干净对照。已确认实现/协议差异，尚未确定哪项造成多少EM变化；混合比例本身也不能证明EM必然提高。

要建立严格只改选题的对照，需要保留旧二行归约数学权重、统一训练与既定旧交互协议、恢复奖励grounding探针生命周期，再冻结统一科学配置。已有checkpoint包含过去更新的影响，修后续训不能抹去这段历史或冒称fresh干净对照。此报告不授权停止、重提或热改当前任务。

远端CPU统计工具 `nonselection_audit.py` 只读原轨迹和rank日志，不加载权重、tensor或执行GPU前向。普通SSH与既有兼容握手尝试本次超时，定量影响读回保持pending；没有以连接失败推断训练退出，也没有编造命中数量。所有完整轨迹留远端。

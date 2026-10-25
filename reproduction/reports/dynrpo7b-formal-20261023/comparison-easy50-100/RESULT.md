# DynRPO 与 Easy：第50～100轮实际差异

本次只读核对真实运行配置、部署源码、408份四rank优化器回执，以及第91～99轮保存的实际训练rollout。未修改、停止或重启训练。

## 1. 超出选题器本身的数学差异

以下 diff 表示两次运行的生效配置差异，`-` 为 Easy，`+` 为 DynRPO，不是建议直接应用的补丁。

```diff
- actor_rollout_ref.actor.ppo_logical_micro_batch_size = 4
+ actor_rollout_ref.actor.ppo_logical_micro_batch_size = 2
- algorithm.all_correct_virtual_zero = False
+ algorithm.all_correct_virtual_zero = True
```

### 损失归一化

两边实际物理micro均为8；`verl/workers/actor/dp_actor.py:723`读取logical micro，`search_g1/logical_micro_loss.py:41`把物理batch分成logical micro，分别做masked token mean，然后对这些均值取平均。

因此4和2并非单纯吞吐参数：不同长度轨迹的相对梯度权重不同。调用实际函数的CPU例子：4条轨迹长度10、10、100、100，前两条每token loss为1、后两条为0；logical2得到0.5，logical4得到0.090909。这是示例，不是实际训练loss数值。

DynRPO绑定点：`reproduction/dynrpo7b/train.py:105`。Easy生效值见runtime-configs.json和history-evidence.json。

### 全对组优势

共享的`verl/trainer/ppo/core_algos.py:132`按开关进入不同分支；DynRPO调用`search_g1/prefix20_advantage.py:35`：

```python
baseline = torch.cat((real.new_zeros(1), real)) if bool(labels[positions].all()) else real
normalized[positions] = (real-baseline.mean()) / (baseline.std(correction=1)+epsilon)
```

Easy关闭该开关，均值和标准差只取真实五条奖励。对于奖励全部为1的全对组，Easy优势为0；实际CPU调用DynRPO函数得到每条0.4082473。虚拟零仅参与基线，不新增训练轨迹；Mixed及全错组不添加零。这改变了目标，不是infra加速。

## 2. 选题相关的实际代码差异

### 补齐512题时，全错题是否允许入选

Easy `search_g1/mixed_selection.py:87`：

```python
correct = [x for x in ordered if x.category == 'all_correct']
if len(mixed) + len(correct) < self.target:
    raise InsufficientEligibleGroups('INSUFFICIENT_ELIGIBLE_GROUPS')
members = mixed + correct[:self.target - len(mixed)]
```

DynRPO `reproduction/dynrpo7b/selection.py:70`：

```python
return sorted(rows, key=lambda row: (row.outcome != "mixed", *score_order(row)))[:k]
```

后者将全对与全错混在一起，按探针分数补齐。这与奖励代码存在具体冲突：两边`reproduction/common8/train.sh:17`均设`ECO_WRONG_GROUND_SCALE=0`。`verl/trainer/main_ppo_eco.py:297`中有效错误答案奖励均为0.05，无效答案为-0.1；全错组若均有效，优势严格为0。

实际CPU重读第91～99轮原始训练rollout，按生产`decode_segments`、`contexts`、`em_check`重新核对，错误标签无分歧：4608组中1132组全错，其中1129组五条均有效，优势为0，占24.5009%；另3组仅有效/无效格式差异产生奖励差异。实际anchored_grpo函数对五个0.05返回五个0。以上是样本信号占比，不能直接换算成最终EM损失或Adam更新幅度。

同区间Easy选入的全错题为0。DynRPO的Backstop仅保底154道Mixed；第50～100轮没有触发，不能防止上述补齐行为。

### 候选数据流

`verl/trainer/ppo/ray_trainer.py:913`：DynRPO使用`QProbeCandidateStream`，Easy带eligible pool使用`Prefix20CandidateStream`。前者从训练loader读取并在当前pool去重；后者按eligible pool及其跨epoch延迟策略读取。原始训练数据SHA相同，但Easy eligible pool为168510题、原始训练池为169028题，候选排列与游标并不配对。

### 生成路径与预算

DynRPO通过`reproduction/dynrpo7b/generation.py`执行prefix暂停/原轨迹续写，普通轮5120题前缀→1024题完整续写→512题训练；每10轮完整生成2560题用于探针更新。Easy完整生成2560题后Mixed优先选512。两边并非同一候选池、同一轨迹的配对实验。

## 3. 基础设施的历史差异

| Easy轮次 | 部署commit | 与DynRPO本次50～100轮主要差异 |
|---|---|---|
|50～52|b76d7a05245a5eb7adc75385e2db803c67acce43|尚未启用后续rollout FA3/compact GQA/快照复用组合|
|53～54|48b87a7399a25f78b4fcf077e946a5a2b75ecace|已启用上述组合，尚无71轮后的search boundary stop/async proxy组合|
|55～60|c7976da3afc863a1c20b99daa7cc99de3bb61f15|native prefix相关运行版本变更|
|61～70|2d5183455551145c52d1783a38663e2fcf170ceb|lean metrics版本|
|71～100|dde1452d2ec65fb3f4a0464d4beacf3b8136414c|主要优化开关与DynRPO相同|

DynRPO部署commit为ee74bfa3c6d1141859abb19e882b65c93ea6f0ba。当前两边`dp_actor.py`、`core_algos.py`、`async_rollout_runtime.py`实际文件SHA一致；相同源码不等于相同数学，因为上述生效参数与分支不同。初始化适配、采集与恢复编排存在代码差异；未发现这些适配本身引入额外奖励公式变化。历史推理调度变更也不能宣称逐token等价。

## 4. 奖励资产差异及排查边界

Soft Reward评分公式相同、初始reward head相同，但第50轮分别用各自actor重新拟合后，boundary/commit head及encoder资产不同，不能认为50～100轮奖励函数数值完全相同。这是训练资产差异，不是MSE/BCE公式或GRPO代码差异。

408份实际优化器回执全部存在、Adam及参数有限，每rank累计写入符合20×outer，两边同一outer消耗的学习率相同。未发现漏更新或学习率偏差。

固定512题验证：DynRPO从257/512→259/512；Easy从259/512→280/512。差距存在，但不能从以上非配对实验给每个差异分配因果贡献。当前最明确的训练信号缺陷是全错补齐与零错误grounding系数的组合；logical micro及virtual zero是必须单独控制的数学变量。

## 证据与复现

- runtime-configs.json：两边远端实际manifest、resolved config、源码SHA。
- history-evidence.json：50～100轮跨恢复目录的日志、选题类别及四rank回执。
- aggregates.json：区间汇总，类别比例均以实际选中512题为分母；非更新轮按51～59等9轮汇总。
- wrong-signal-audit.json：实际rollout只在远端CPU读取的汇总；不保存原始题目、token或tensor到Mac。
- easy-training-metrics50-100.json：对方已验收验证结果及训练指标。
- 复现入口：`python3 -m reproduction.operations.compare_dynrpo_easy_50_100`，`python3 -m reproduction.operations.audit_dynrpo_wrong_signal`。需要现有私有RAM凭据，不在代码或输出中提供凭据。

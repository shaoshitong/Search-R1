三个真正多层训练分支的新增入口是 `replay_layer_train.py`，模型为 `replay_layer_probe.py`。本次实现没有启动GPU，也没有替换已有模型、特征reader或活动运行源码。Matched6528 G10 random/global-balanced对照仍是独立的数据构组轴，本入口不取代它。

- **初始化**：zero-based block21/24/27分别加载官方初始7B对应block，输入对应postblock20/23/26。三者均为Q273+Prefix20、flatten维度`293*3584`，全新head使用共同seed1729；EMA初值精确复制fresh raw，Adam从0开始。不是继承E1/Adam的resume。
- **数据及预算**：相同6528个fit问题，G5原轨迹不重采样；`build_plan(reference_plan)`生成共同4epoch shuffle计划，每epoch128批×51题，合计512次真实更新。LR5e-6、EMA.99、trajectory BCE+three-event NLL、micro2不变。四轮训练预算不同于之前128步fresh head筛查，不将两者称作等预算。
- **后端**：fit、512诊断、2048 viewed selection全部通过同一层的同一个`ReplayFeatureReader`。每个进程启动只初始化一次reader并做完整文件SHA；后续更新和评估仅按reader逐条检查重构SHA。没有native特征fallback，不宣称HF/native bit-exact。
- **持久状态**：CP0支持真正空Adam；首更新执行两次真实训练并从同CP恢复raw/EMA/Adam/缓冲/RNG，逐组件hash一致后接受一次。保存cursor、数据位置、plan/config绑定、raw/EMA、完整Adam、RNG及phase。每16步保存；128/256/384/512节点必须先保存完整checkpoint，再固定512+2048评估。评估失败可从边界checkpoint先补评，再继续；未持久化的更晚训练事务仍明确拒绝静默回滚。
- **资源**：每分支1张H100，`render_sbatch()`默认normal/acd_u、8CPU/64G/2h，没有依赖和自动提交。模型shard SHA及对应block权重只在GPU节点读取，login仅准备配置和小metadata。退出记录实际CUDA allocator peak。

独立配置schema如下。所有`REF`均为真实远端`{path,sha256[,bytes]}`；不得编造尚未完成bank的SHA。

```text
block_layer: 21 | 24 | 27
epochs: 4
update_budget: 512
global_groups: 51
micro_groups: 2
save_every: 16
lr: 0.000005
ema_decay: 0.99
head_seed: 1729
training_seed: 1729
loss_kwargs: {trajectory_weight: 1, three_event_weight: 1}

source_matched_config: REF  # accepted matched-loss128 event/config.json
reference_plan: REF         # its original128/B51/6528 plan, copied exactly
plan: REF                   # build_plan(read(reference_plan))
bank: REF                   # exact source_matched_config.bank
selection_bank: REF         # exact source_matched_config.selection_bank
replay_bank_complete: REF   # merged four-shard bank-complete.json
replay_features: REF        # exact complete.features_by_capture_layer[20|23|26]

model_dir: /remote/verified-official-snapshot-view
model_config: REF           # model_dir/config.json
model_inventory: REF        # model_dir/official-inventory.json, revision checked
model_index: REF            # model_dir/model.safetensors.index.json
official_block_shards:
  <every shard named by selected block in official index>: REF

code_hashes: {repository-relative runtime file: actual published SHA256}
```

完整code closure应覆盖runner/model、ReplayFeatureReader、replay_probe_layers、probe_ablation_model、probe_ablation_train、probe_ablation_loss、runtime及它们实际本地import依赖。每层源block从官方shard严格加载后，运行回执另保存实际FP32 block tensor SHA、官方shard checksum、共同head初始SHA和geometry。配置SHA与checkpoint绑定。

6项CPU小测试覆盖：三block独立/共同head/Adam0和EMA初值、padding及梯度、四轮全cohort计划、首更新实际toy双重放、CP0/CP1及下一更新完整恢复、边界先checkpoint后评估且RNG不变、normal单GPU两小时脚本。真实GPU first-update/replay与四轮最终指标仍须执行后验收；本次不提前声称通过。

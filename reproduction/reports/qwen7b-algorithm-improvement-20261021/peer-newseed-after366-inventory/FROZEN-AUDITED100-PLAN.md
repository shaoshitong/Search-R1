# 固定已有独立审计的100批源数据

此次只冻结源端metadata；实际export、HF上传、下载和GPU均未由此操作启动。

权威NAS10计划为 `/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/metadata-plans/audited100-after366-20261021.json`，170,253 bytes，SHA256 `00e5021284cb8f6614670452b749c7add10c028e50578822432600762074662a`。本目录 `frozen-audited100-plan.json` 是同SHA的metadata副本。

- 固定原观测时已独立审计、当前fstat一致的100批：12,800题、64,000条真实G5轨迹；HotpotQA 12,672题、NQ 128题。
- `selected_batches` canonical digest为 `636015f5b3cf84ad49688a1cee6b8d15959ccbc59451e91f6dae5d625df103ce`。
- 对已接受244+122=366批的batch、qid、question hash、trajectory ID交集均为0。
- 原28批不进入此次计划。之后其中25批取得了新proof，也没有扩大这次成员。
- 这不是原warmup未见问题，不是完整169,028题源采集完成，不是新增数据已接入HPC3。

## 两次实际验收

`freeze-created.json`记录第一次原子创建并回读（9.72秒，`existing_preserved:false`），执行源码和远端SHA均为 `eee0a9a8c9208995488e1692227206fd54e9b894`。`freeze-receipt.json`记录第二次独立调用的小型回读（7.79秒，`existing_preserved:true`），执行源码和远端SHA均为 `291eae8f1f78a06d65b9f84d4ac1d3cb52943764`。两次计划路径、bytes和SHA完全相同；第二次没有重写或扩大计划。

最初尝试因全128批的audit摘要变化而拒绝，保存在 `freeze-attempt1-rejected.json`。定位结果是未选28批中新增加25个proof；原选100批的input/seal/科学payload声明SHA及原proof没有变化。修复按原seal时间与原proof时间cutoff重构原128记录，得到原digest `d2453003c9061af5309e674c6dcefe38ba19dae24b27da734093275b6e7e8edb`；当前128记录digest为 `3deed4f541a5a57919baa4be1e7ff4c6a08097daf5156e570c1b8124400a2e81`。计划明确记录了这种差异，未忽略选中payload变化。

## Export消费接口

计划schema为 `FIXED_AUDITED_SOURCE_G5_SUBSET_AFTER_ACCEPTED366_V1`，`subset='existing-audited'`。下游应显式绑定上述整个计划文件SHA，使用固定 `selected_batches`，不再重新发现或扩大成员。

计划保留 `source_base/source_run_id/seed/source_seed/model_revision/sampling_attempt_id/source_config/input_manifest`；每个selected batch包含精确input/seal/proof descriptor、counts/domain及ordered identity digest。`prior_snapshots`包含旧244和122两个真实source complete descriptor及expected batch/question/trajectory计数；`accepted366_identity_membership_refs`与零交集结果绑定这366批。后续通用固定subset exporter应验收这些prior memberships，不能直接使用硬编码122批的旧validator。

当前proof的科学payload SHA继承已有独立源审计，仅fresh验证metadata/fstat和选中input身份。它不替代未来export时逐archive/member fresh SHA、HPC3接收后的fresh readback或strict EM/seed接入验收。原peer服务与进程未改变，也未向peer线程发消息。

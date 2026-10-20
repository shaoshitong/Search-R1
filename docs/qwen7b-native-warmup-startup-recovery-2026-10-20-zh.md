# 7B 原生探针 warmup 启动恢复记录

## 范围与版本

目标为官方Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28` 的原生探针warmup；actor冻结，不启动studentRL。zero-based layer26 post-block原生特征，训练layer27+flattened Linear；question+Prefix5拟合，同轨迹Prefix20及真实位置/mask保留。

远端base：`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019`。

- 采样源码：`f2d5ae7de92d198692abf5bf82143e309d91c271`，不可变`base/repo`。
- 当前拟合源码：`96df5a23f23f6da0f24e79e67dcf0e3039f22de0`，不可变`base/repo-postprocess-96df5a23f23f`；22文件已核对部署SHA。
- 运行作业：740358，普通acd_u/normal，4张H10080GB、16CPU、128G；09:47:53 CST启动于ACD1-52。此文档不声称20epoch已经完成。

## 数据已完成

模型11文件共15242788168字节，目标`assets/qwen25-7b-instruct-a09a35458c70/complete.json`实际`MIGRATION_ALL_SELECTED_ASSETS_VERIFIED`且fresh_all_files_SHA256=true。

八份采样739426均完成，共169028训练题/845140轨迹，512独立诊断题/2560轨迹；总847700轨迹。`bank/complete.json`实际`FULL_CANONICAL_7B_G5_NATIVE_FEATURES_AND_OWN_EM_VERIFIED`。run_id `bdbb550fd9324cefab979728acb0f5e1`不变，未重复生成。

`bank/groups.json` SHA256 `1dbec76bab828a391de66f01fdaabfafc1557f699df174e82d81952263278bf8`；`features.json` SHA256 `6d2ff5e6f522c4be37a01d9af130be1ab656d7e903d813d48ad0cf29de6ec7ff`。

训练K=0..5题数：[77725,17561,14930,15599,19109,24104]，真实Mixed39.7561%，轨迹EM34.4528%。独立诊断：[228,63,50,44,48,79]，Mixed40.0391%，轨迹EM34.4531%。这些是冻结actor的真实标签统计，不是探针准确率。每epoch平衡233175问题组，补采64147组，4573个真实Adam更新；20完整epoch。

## 首次拟合739951故障

该作业FAILED1:0，运行26:59。在拟合前全量train评估末尾，rank0/2/3各完成42257题，rank1仅29081题，超过默认NCCL600秒，最终20项指标ALLREDUCE超时，后续出现次生assert/SIGABRT。生产Adam更新数为0，不能声称已经训练、额度不足或丢失已训练优化器。

失败预测文件、初始化及提交回执均保留；`control/fit-startup-failure-739951.json`及`control/retired-fit-739951`记录该事件。所有采样bank和模型保持原样。

## 修复与实际验收

- G5问题基底共享解码/哈希，但每轨迹delta仍独立读取；逐行重建BF16 bitSHA继续验证，未假定五条特征完全一致。
- forward-only评估micro8组，每32步匹配collective并写各rank进度，包含不均匀尾批。训练micro2/全局51组、全G5损失与梯度权重不变。
- 初始完整step0状态先保存，随后独立shadow模型/Adam执行真实四卡恢复测试，生产模型与RNG保持原样。
- CPU打包/未来15token隔离、DDP不均匀全局均值、Adam/RNG下一步一致性均通过；真实两批256题/1280轨迹所有文件SHA、20行原生特征独立恢复通过。
- `probe-qualification/proof.json`和`qualification-740358/proof.json`实际为`FOUR_RANK_GPU_RECOVERY_QUALIFICATION_PASSED`：原生GPU Prefix5未来15token隔离、51题不均匀评估、shadow第1步保存后恢复并重算第2步，所有4rank的raw/EMA/Adam **bit_exact=true、最大误差0**，生产更新0。
- 正式`probe/initial.pt`1872501278B，SHA256 `1f3d271f8baed6f4cac9b642b3626df3a90690617e7e9678ad70d4429ead178a`，CPU重载及四rank状态校验通过。

## 尚需完成

当前已进入全量拟合前评估，逐rank计数持续推进且保持接近；正式Adam更新尚待确认。必须完成baseline、20epoch逐epoch训练/诊断与raw/EMA/Adam/RNG checkpoint验收、最佳EMA选择及最终交付，才能报告目标完成。不可把shadow资格测试的更新计入warmup预算。

权威进度：`probe/predictions/eval-*-rank*-progress.json`、`probe/progress.json`、`probe/curve.jsonl`、`probe/latest.json`、各epoch回执及最终`probe/complete.json`。本地仅保存代码/小摘要，无模型、tensor或完整rollout。

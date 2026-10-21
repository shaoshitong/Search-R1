当前执行实测（2026-10-21；目标保持active）

主训练745501：1025/15528真实更新，Adam15443，完整可恢复CP1024；原PID300719/birth116543024在ACD1-28推进，无failure/interrupted。124194题G10冻结，原生block27+readout联合训练，LR5e-6，每batch64Mixed/64nonMixed真实G5视图。当前首个新增epoch尚未结束，没有本轮独立效果提升可报告。

全量canonical169028题计划逐qid与sortedSHA闭合：aa14bce45fc6a793789369d6f33c84edf3c1d79e601fec45882ec193b1e4c58d，缺失/多余0。最后生产实测205/241批=26240own题；与main/donor去重联合164514题生成、4514待生成。真正full rawG10验收现162210题（95.9663%），剩6818待验；不能用生成数代替通过数。新14592题old5/new5各72960条strictEM/causal/seeds/fullSHA均0错，346份batch metadata回执freshSHA通过、audit/launch锁释放。

特征745731（normal1H100）已生成93440/93440行、292/292完整chunks，仍等待完整逐行retained/sourceSHA最终bank接受。donor745832（独立normal1H100）最新可见38732/140800行、121chunks；启动后480->2656行/1->8chunk已实测，无GPU热改。可变progress存在NFS可见性窗口，仅只读control做窄修05f28e2b；缺失时标unknown，不报0或停训。下一14592独立L26入口准备中，只有完整raw与CPU投影/physical末次绑定通过后执行。所有模型/特征留远端，当前fit未扩容。

旧E0 EMA独立2048：MixedBA55.7714%，Top512196Mixed/209全错/107全对，实际Mixed38.28125%，原池32.37305%；全对题进入Top51232.42%仍高于Mixed29.56%。这些是旧模型短板，不是新训练收益。新epoch必须对比固定阈值MixedBA/真实Mixed@512及三类误选，不能以训练loss代替泛化。

完整目标还要求把全169028 ownG10+同合同L26数据实际用于独立数据scaling联合训练；不能仅备齐数据就宣告完成。当前124194实验保持不变、不丢Adam/RNG。新增全量实验需独立source/config/output/locks及实际启动恢复验收，比较matched-update和epoch节点。最终选择完成前不读取untouched4096结果。当前head-only高LR分支仍延后。既有terminal-only精确续跑控制保留，不在原writer活动时重复提交，不静默回滚。最后还需实际收益确认、fullcheckpoint和资源收尾。

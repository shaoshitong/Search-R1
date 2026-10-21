CPU-only三bank元数据合并与内部诊断预冻结。

合并30016+15616+12800=58432真实G10，严格核qid/hash/十条ownID互不重复。整池必须与固定diagnostic512、selection2048、final4096身份无交集；只读已嵌入identity-only排除，不读outcomes。后两批28416中，从holdout候选排除controlled6528 qid或hash，按SHA256([peer-new28416-internal5120,9137,domain,qid,question_hash])排序，每域各2560。域配额不是Mixed标签均衡；所有标签、模型分数与参数均不用于选题。原warmup已见全部169028，绝不宣称问题未训练。

内部5120主G5固定每题真实newseed5，即原G10索引5..9，使用这些轨迹自身strict EM，不按标签择组、不平均多组。阈值固定0.5。报告pooled与NQ/Hotpot各自BA、Mixed/非Mixed召回、AW/AC误报、AUC；主Top覆盖25%，pooled1280/5120、每域640/2560。可选pooledTop512仅10%，不可冒称与2048选512相同覆盖。此集只作新增内部诊断，最终4096 outcomes仍不读取。

holdout之外的所有新23296题（包括controlled重合题）保留作fit tail，以SHA256([peer-next-append-only-qid,1729,qid])排序，728个B32，四唯一G5 view/题总权重1。fit总53312=旧30016+新23296。actual config必须fresh绑定原938 plan SHA；字节复制原plan，原938 ordered batches前缀逐字保留，仅追加728，未来目标1666。此阶段不绑定模型CP、不选LR/recipe、不验证或宣称无损模型恢复、不提交训练GPU。

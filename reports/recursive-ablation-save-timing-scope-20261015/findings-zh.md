# 实际保存耗时与计时范围

C1 前5轮的四次非评测 rolling 保存累计 **152.564 秒**，各次约33.35–43.34秒。该计时包含八rank保存RPC、driver/manifest及旧rolling回收，发生在原生step计时之外。

原生 `timing_s/save_checkpoint` 的微秒数是受保护评测路径里的空钩子，不能解释为真实保存只需微秒。第5轮的testing已包含HF导出、评测前native保存、512评测和评测后native保存；不能再把这些分量重复加到step。

C1第5轮评测前各rank保存加reload为52.38–55.84秒，评测后保存为33.96–35.82秒。A的5/50/100/150节点逐rank计时见summary.json。八rank并行，不能将其和当成wall。

HF导出和关键节点coordinator保存没有独立完整wall记录；A的rolling保存prefix也未恢复。这些项保持未知，不用减法或相加制造端到端耗时、GPU活跃时间或计费。此报告只投影已有小收据，没有改变训练、评测或旧报告。

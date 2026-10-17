# I1–I6 修复与官方actor fresh200

按2026-10-17最新人工指令，本次执行清单仅I1–I6。旧3430已人工取消，last joint99；18888原进程birth两次核对全部消失，cgroup与自有GPU占用均空。错误实验a008／5cba的12个源命名空间仅删除自有权重／optimizer／log，代码、必要小审计与原始轨迹gzip保留；warmup、官方权重、真实round0 grounding、数据、FAISS及fixed recovery assets保留。

|项|修复实现|必要验证|
|---|---|---|
|I1|train/QA/后续grounding校准共用新interaction contract：4096total、500action/answer reserve、闭合information、预算early-final|wrapper/预算/动作CPU边界；正常QA5起验证实际输出|
|I2|physical local16不减，forward后8个原micro2 token mean等权；OOM拆到8/4/2/1仍同目标|真实PPO函数loss/梯度CPU等价、mask与空EOS组|
|I3|真实round0导入；50/100/150用当轮actor真实fit两head、8replica adopt；未adopt不得下一轮|版本时钟／恢复／CPU保存点测试；真实refit到50才发生，不冒称已GPU验收|
|I4|恢复每mini真实before/after原KL与负优势tail monitor；固定rank4条；after可作下一before|原统计CPU检查；有限阈值按已授权diagnostic-only告警、真实非有限拒绝；无独立GPU gate|
|I5|train/QA统一BF16，FP32 master/loss保持，原题greedy及strict scorer不变|新QA contract/dtype与8worker实际输出绑定|
|I6|train/QA共用首个完整action的stream停止、EOS与原IDs裁剪|prefix20/21flush不变，同prefix续写及LP/mask/position保留；CPU stream mock|

新actor：官方Qwen2.5-3B-Instruct aa8e725；旧错误actor／Adam不导入。Warmup不重跑，仅导入已实际保留最佳CP100探针模型，online Adam0。最新显式goal将探针学习率设为1e-6（替代先前1e-5）；每20轮全5120G5自然一遍100Adam，其他轮冻结。Actor仍200outer、20Adam/outer/rank、globalmini128、LR1e-6/57outer warmup；8卡大batch。固定512在5/50/100/150/200及final51713沿用原题与strict评分，新interaction contract统一执行，不把旧QA分数混入本run。

正常checkpoint/backup/archive已改为动态绑定本run，含grounding版本／encoder／head依赖；没有旧a008固定身份。旧round0实际拟合资产来自原协议，本次明确复用；后续正常校准显式启用新交互合同，不更改其他实验默认路径。初期验收须真实连续5轮＋第6轮实际开始，不能以CPU测试或RUNNING代替。

未进行额外GPU测试、基线重生成或重复warmup。新run只在源码正常commit/push及独立源码SHA后一次部署提交。当前记录不宣称新训练已完成。

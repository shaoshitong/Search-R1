新分叉 Tree-GRPO 已完成 **200/200 个完整 outer、每 rank 4,000 次有效 Adam 更新，以及七集 51,713/51,713 题最终评测**。七集按题数加权 F1 **53.8154%**，EM **46.9398%**。最终成本、资源收尾和独立最终验收均已完成；42项要求中35项直接证实、7项如实记录验证/测量限制，无缺失项或未完成项。

| 数据集 | 实际题数 | F1 (%) | EM (%) |
| --- | ---: | ---: | ---: |
| NQ | 3610 | 54.7124 | 46.2050 |
| TriviaQA | 11313 | 70.1270 | 62.4856 |
| PopQA | 14267 | 51.4466 | 47.2209 |
| HotpotQA | 7405 | 54.6933 | 43.1330 |
| 2WikiMultiHopQA | 12576 | 46.4068 | 40.7363 |
| MuSiQue | 2417 | 26.2041 | 18.1630 |
| Bamboogle | 125 | 49.2467 | 35.2000 |
| 全部（按题数加权） | 51713 | 53.8154 | 46.9398 |

用户批准停止当前 778 的旧慢最终评测并换 8 路 TP1。原终态 FAILED/exit1 和中断成本保留；这不是训练200轮失败。新评测在原 H10018（h100-10/10.33.0.18）执行，8 个常驻 TP1 worker、每 worker64题、BF16/XFORMERS、本机 GPU 检索；共享8张物理卡，Slurm预约0卡，不声称独占。保留同一HF200、固定split、四轮搜索、2048/4096/500/500及greedy评分协议。批次/kernel形状变化不保证逐bit相同。

8路TP1实际最终流程 3636.757847547531 秒（60.61分钟），8worker exit0，809个提交批次覆盖唯一连续 index0..51712，七集计数均完整。两次实际终态读取一致；run-result COMPLETE，metrics SHA775c209ff0c88d37543c6e7437fa160a611559ca3f095695bc66e85506ec2a20。最终collector两帧确认自身cgroup/PID为空、8个UUID租约无holder、本机18082无listener；两个CPU审计unit及本任务独立SSH连接也已清理。systemd已回收，精确ns CPU计数不可用；journal终态记录28190.250 CPU秒，精度为毫秒，不能将活态asof替代终态。

训练有效来源为0bef outer1、ba9 outer2–5、778 outer6–200；实际1600条直接rank记录和各rank4000次有效Adam更新。历史失败、丢失更新和回退重算分列，不等同全部物理历史仅4000次。第200轮每rank20次成功、无跳过；固定57轮warmup及实际LR已核对。训练从官方 Qwen/Qwen2.5-3B-Instruct revision aa8e72537993ba99e69dfaafa59ed015b17504d1开始，reference同基座；同一新实验链恢复保留，不继承旧3360权重/Adam/scheduler。

第200轮dev512（NQ/Hotpot各256）F1 58.4719002785409%、EM49.21875%，40次每5轮dev全部完成。dev512与全量七集题目组成不同，不能把二者分差直接解释为效果退化。5/50/100/150/200 native/HF存档仍在远端；native200、HF、tokenizer完整CPU重载及全部有限性通过，最终HF对比3,085,698,985/3,085,938,688个参数实际变化。第1轮8192坐标样本差为零的事实保留；有有限梯度/实际Adam写入不等于该样本变化。native5已做8rank冷恢复；不声称native200 GPU世界恢复或完整executor精确resume。

A树随机分叉形成2叶；B树按奖励差异从尾向前回退得到B1，再从B1递进得到B2，共3叶。全部200轮/102400题实例/512000训练叶/620594候选的祖先、分叉动作、继承前缀哈希、每叶质量.2、配额2/3、policy权重和原shared_prefix_credit数值关系已通过独立审查。不是独立重算全部逐token优势向量。该目标是奖励筛选后的surrogate；token μ校正不能消除完整候选选择偏差，有限回退不能保证混奖或树间差异。第200轮仍有209/512五叶全对、214全错、89混奖；原树内奖励方差.023193359375、纯树内优势方差.06201151347109296、实际总优势方差.3770195396024971，均为实际记录。

200轮原始采样成本：生成236862900 tokens、prefill1351202404 tokens、检索880769次、评分825394次；额外候选620594，其中accepted307200、discarded313394，fallback185191为accepted子集，不重复相加。outer wall235453.225841秒，包含dev24760.471602秒；generation119142.712781秒、actor update55496.759776秒为阶段子集，不再次加总。失败历史和观察器另列。8×wall只能表示共享物理卡容量上界，不是独占账单、实际GPU活跃秒或能耗。

完整30项×200轮scalar/方差CSV：[ACTUAL200_COMPLETE_SCALAR_SERIES.csv](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-nested-branch-handoff-20261013/ACTUAL200_COMPLETE_SCALAR_SERIES.csv)。训练/dev/方差图：[PNG](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-nested-branch-handoff-20261013/ACTUAL200_TRAINING_AND_VARIANCE_FINAL.png)、[PDF](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-nested-branch-handoff-20261013/ACTUAL200_TRAINING_AND_VARIANCE_FINAL.pdf)。旧图在评测完成前制作，原样保留，当前图已更新说明。200train+40dev完整原始curve、成本、树结构证据均三副本双读，原始rollout/全量逐题输出/模型/optimizer仅在远端。

旧3360 native/HF200、源码manifest和独立验收小SHA仍保留；90MB旧composite仅stat不传Mac。当前固定28条公开产物路径范围没有匹配完成的同硬件/后端/split/协议GRPO控制，不作公平算法增益结论，也不拿旧Tree、GiGPO、H20或跨集群运行替代。

远端主训练目录：/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:778b2b43dad90fc0c1c5d95d77c2326286011b73。最终逐题结果位于该目录/output/final-eval-fast/scores。当前已完成CPU审计两个unit清理；最终主评测成本与双读cleanup已完成，独立最终delta已PASS；未重训、未重复评测。

最终评测成本已闭合：SearchRollout记录generation31,697,148 tokens、prefill135,907,135 tokens、检索119,053次；逐题保留输出统计11,453,564 tokens/119,067个search tag事件另列，不把两类计数混同。批次process CPU26,537.642854711秒是服务CPU子集；批次wall26,702.112292856秒跨8rank重叠，不与3636.757847547531秒supervisor wall相加。服务journal终态CPU28,190.250秒；精确ns/完整unit execwall/独占GPU账单/自身GPU活跃秒/能耗/未记录评分及HTTP调用均为空，保留覆盖限制。旧778慢final直接control记录600.934673652秒及trainer CPU191.472203417秒，为可分离子集；整个778段CPU1,116,210.663921秒含训练/dev/旧partial final，不再次加账。资源guard3526个有效样本，平均间隔1.030734865秒、最大1.408735752秒，无坏行/大于2秒间隙；整卡峰值是采样下界，不是自身持续利用率。唯一terminal collector main1/helper1，不重复训练/评测/旧成本采集。

最终独立验收：[FINAL_DELTA](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-nested-branch-handoff-20261013/FINAL200_INDEPENDENT_REQUIREMENTS_AUDIT_FINAL_DELTA.json)；所有42项要求完成，无未完成工作。旧DRAFT及TRACE_PRESERVATION_DELTA保持原始SHA。未知的历史GPU活跃/能耗、退休节点旧背景PID、候选选择偏差、完整executor恢复边界、未匹配GRPO等限制已公开，不伪造补证。旧30分钟automation已PAUSED，保留本线程与所有远端产物。

114个新增封存批次、14,592题的原5+新5已完整通过CPU校验，共145,920条真实轨迹。两步分别用原745518/745519 allocation的2CPU/GPU0，570.5和571.9秒退出0。每条重新计算tokenizer strict EM，校验原始payload SHA、Prefix20到full因果链、own IDs及原/新native seeds，差异和同题种子冲突均0。

346份batch验收小回执已fresh SHA复读，输出组与事件清单SHA相符，audit/launch锁释放。qualified raw联合题数因此到162,210；最后54批/6,818题另行封存和验收。当前124,194训练未扩，未生成L26特征或申请新GPU。whole-G5来源须先选定，禁止混合相同protocol tid的不同执行payload或提前union特征manifest。

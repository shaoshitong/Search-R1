# Native Qwen 启动隐患专项静态检查（2026-10-16）

结论：`fcb77204f48d5cb8ec21c68df7d4a2c937b3557a` 的最小修复可用。508项source map为 `19edc0fd3382948decd0e06074e4e82d0f29ef269163131ef07472d29c6e2b1b`。仅检查本地源码和既有标量/哈希回执，没有远端调用、GPU测试、模型/tensor读取、作业操作或源码修改。

## 确定问题与修复

- **已观察的致命错误**：`qprobe_runtime.py:881` 的native/token-binary producer输出 `selected_predicted_binary_mixed`，原trainer在 `ray_trainer.py:940` 读取 `selected_predicted_diversity`。实际512×5生成已完成，四rank optimizer文件及attempt/update账本尚未产生。新提交按 `token_binary` 读取真实字段，保留count6语义。
- **已确认的下游漏计**：`qprobe_selection.py:241` 输出 `predicted_binary_mixed`，原trainer分组诊断匹配 `predicted_diversity`，会漏掉全部2560条native轨迹。新提交同步使用真实来源；未添加假的diversity别名。

## 周边核对

- collect验证512唯一qid、512×5整数严格标签及2560条实际轨迹绑定，binary输出为两个一维向量。Top512/E0仍只按pMixed排序。
- 四rank形状一致：每rank640条、mini32、micro4；global micro16、accum8、20机会。Head只在全部真实rank返回一致complete后更新，不完整attempt不fit。
- Head仍为每complete outer两epochs、batch256、最近8个formal outer、raw age weights `[2,1,.5,.25,.125,.0625,.03125,.015625]`；teacher排除。Adam预计outer0/1/5分别200/204/260。
- qualified旧bank和本实验warmup10/Head200采用接口保留；native layer35 exact-copy、layer27全部有效token2048、pad244/flattenLinear2、BCE+.3 normalized std均未变。
- birth UUID/SAME Ray/CPU RNG在历史实际run已通过，本次driver源码未改。外层driver EXIT1仅包装本次下游KeyError。
- checkpoint typed保存/加载绑定actor id、Head版本、SHA与complete计数。既有验收边界保留：nativefive先于post-eval completed-pair写出；仍需CP5/Head回执、completed-pair及same-process6，不能把row5单独当最终验收。

本专项范围内未发现额外确定致命问题，也未将未验证猜测列为风险。实际新run完成情况由主线按既有进展验收规则实测。详细行号、source hashes与evidence hashes见 [专项审计 JSON](qprobe-native-startup-hazards-2026-10-16.json)。

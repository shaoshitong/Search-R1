# A800 prefix5/P10：从完整joint27恢复，续写改为eager

日期：2026-10-17，Asia/Shanghai。

## 故障与保留边界

同一科学run `387d3f6e228a488f90c418e2ba47e28e` 的父执行 `cb12bf6ab10b46b0a72f1f0c2e753d78`、源码 `858381986d3332898c4e8aa8e5c31e966363ceea` 已完整提交outer27。四rank各540次actor Adam、scheduler27；探针最近完整更新outer20、各100次Adam。父actual config SHA为 `3c11e74d379b7a95163aacf86bd9554b7165aa8dd8bafb3bc2fc013f337e0e17`；joint27 SHA为 `5fe53575eea47c2393b2925e0c108ba0164787429a0528f637bfeb34a2341679`。

outer28的12800条prefix与probe-score已完整，continuation发生真实CUDA illegal memory访问，未产生完整continuation rank receipt、未发生outer28 actor更新。原始部分产物及日志留在父执行，不伪称回滚，也不把outer28计为完整。原件终态证明3041个owned births均已退出、cleanup为true；新入口不发送任何停止信号。

## 本次必要改动

仅将后续生成执行配置 `rollout_enforce_eager=True`，避免沿用出错的CUDA graph执行路径。模型、数据、采样温度、Top512/真实EM筛选、联合loss、奖励、全对组零锚点、PPO与LR配方保持父配置。eager不会被描述为新算法，也不保证未经实测的加速。

恢复完整actor/probe/Adam/scheduler/四rank RNG、候选pool与有效历史链。不得回到fresh0，不再次导入warmup CP5，不生成teacher或重编码特征。只复用CPU全SHA合格的prefix/score28与完整continuation rank；本次实际whitelist为空，失败partial不复用，继续生成使用原前缀。

新执行唯一ID `6cc73ac131734c9e8e4522c99f503c90`，新namespace：

`/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-387d3f6e228a488f90c418e2ba47e28e-resume-6cc73ac131734c9e8e4522c99f503c90`

源码发布commit、远端source-file SHA、CPU验收与CUDA恢复结果须分别记录；本文件不把尚未执行的准备或launch当作恢复成功。

## 发布、CPU验收与唯一提交

新控制入口 `reproduction/operations/a800_prefix5_p10_eager_joint_resume_control.py`，只允许 `stage / prepare / launch / observe`，没有stop动作。原有control.py物理路由不变，不改SSH/VPN/auth/knownhosts，不global ray stop，不操作他人任务。

1. 先Git提交/push代码与必要配置，核验远端commit/blob。stage逐文件验证source inventory、SHA与archive SHA，不允许symlink、越界路径或旧namespace覆盖。
2. prepare启动独立CPU worker，一次调用 `checkpoint_native_a800_joint`，完整SHA/finite验收joint27与原始状态、prefix/score28。输出 `preparation/native-a800-joint-qualified.json`；CPU通过不是CUDA恢复验收。
3. 从父实际config深拷贝，只变新base/source/execution、执行路径、eager、`resume_joint.adapter=native_a800_joint_v1`及明确qualified continuation复用映射。删除已完成的旧 `resume_pending_probe`，避免再次走补probe10分支。
4. CPU入口 `native_joint_resume.import_complete` 导入合格完整状态和元数据；同run历史1..9仍绑定31f来源，10..27绑定858来源，各自原路径/内容SHA继续保留。不会为满足新source守卫重写旧历史。
5. CPU导入后，通过原GPU inventory/compute-process metadata选择实际idle的四训练卡（排除5/6），辅助5/6也必须实际idle；若不足，worker本地30秒等待，不占位、不停他人、不从SSH连续轮询。最终配置记录indices/UUID；真正CUDA启动仍由native allocation在OFD UUID lease下再次准入。
6. 记录CPU实际exit0及final config/qualification/import SHA后，launch再次核实父全部gone、完整资格链与新source，唯一submit。启动本身不证明训练推进。随后至少两次真实增量与首完整正常恢复outer28，继续同执行至逻辑200。

各mutation的once claim不可重复使用。SSHtimeout只是观察失败，不证明退出；失败namespace、原始日志、claim、CPU proof和partial状态全部保留。

## 只读观察与完成条件

observe仅ONE有界SSH，65秒/64MiB metadata/128KiB输出，读取已有JSON/scalar、birth、四rank Adam、checkpoint元数据及既有QA。完整历史1..27通过CPU seal和import `history_provenance` 的原source/path/SHA绑定；28起必须为新source。FAILED资格文件只返回实际error与未知边界，不把缺字段当0或遮蔽为新KeyError。无模型/tensor readback、概率扫描、额外GPU验收、review或评测。

候选2560题×G5；普通轮预测Top512，10/20/…/190更新轮按真实混合优先512、不足按探针分数补齐；probe LR5e-6、每周期50 Adam；outer200不更新probe。actor每完整outer20真实Adam；全对G5零锚点公式不变，全错和混合按原公式。QA/checkpoint保存点5/50/100/150/200及final51713保持。

真实1..200完整链、最终canonical有限值/SHA、final51713 ordered IDs/二元EM/F1/七源NQ3610与only-owned cleanup全部通过才可完成新实验；旧暂停goal不因本次恢复而完成或删除。大产物只留远端，Git仅代码/config/docs/scalar/hash。

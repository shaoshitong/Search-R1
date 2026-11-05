# V8 给编辑的实验章节证据矩阵

2026-11-05。只读整理 V1—V7 已有文档和最新保存结果；不联网、不重跑、不挑选重拟合结果。本文是写作输入，不是新实验。下文路径均相对仓库根目录；列表键采用 `[字段=值]` 表示筛选唯一记录。

## 先确定正文到底回答什么

**建议主研究问题：平台曝光的持续性，主要来自目标作品自己的历史，还是来自同账号其他作品的共享信息？这种区分在新作品上是否改变？**

解决的具体问题是：仅凭“历史份额预测得准”无法判断应该给博弈模型加入什么状态。本实验通过分开来源作品和目标作品，比较作品记忆、公共账号信息及其组合；它验证的是预测信息的适用对象，不是识别算法的历史奖励、不可转移资本或 AI 降本效应。AI 与身份竞争保留为建模动机和条件性应用，不冒充已发生的处理实验。

一条完整实验叙事即可：**同一信息截止和候选池 → 四个嵌套评分族 → 全体作品主比较 → 新作子池与来源历史分解 → 线性候选选择 → 小型外部作品记忆复制 → 结论及未回答的问题。** 不按 V1、V2、V3 的工作顺序讲故事。

## 术语必须先用人话定义

| 正文用语 | 实际定义与来源 | 不可替换成 |
|---|---|---|
| 作品历史记忆 m | 目标视频过去曝光 / 年龄时长基准期望曝光的收缩比；分子分母按过去日期衰减 | 实物库存 B、内在质量、算法因果反馈 |
| 共享账号信息 h | 从不相交来源作品计算的同类曝光残差比；同日同账号全部目标作品共用 | 已识别账号资本 H、h−1 的非负资本、owner 能力 |
| 新作品 | 源表实际有 `date==upload_dt` 行的视频；上传自然日，不是完整24小时 | 新注册账号、真正新创作者 |
| 账号 | 数据中的 author_id；经济 owner 及跨号归属未知 | 同一真人、策略玩家的真实账号数 k |
| 曝光 | KuaiRec show_cnt；外部源为其公开 impressions 字段 | 独立人数、质量、收入或福利 |
| 条件交叉熵 | 在同日同方向实际观察候选池内，按曝光加权的负对数预测概率，单位 nats；越低越好 | 创作者效用、均衡偏离收益 |
| 联合 / 组内条件损失 | 联合保留分给其他组的概率；条件损失在指定子池内重新归一化 | 两个可互换指标 |
| α | 当前有限预测族的账号总权重指数，校准网格候选 | 平台公布规则或已识别生产回报 |
| Q / F | 截止预测日前已出现的目录数 / 其中近期上传数 | 完整产量、真实活跃库存、理论本期生产 x |

定义依据：`docs/V7_EMPIRICAL_PROTOCOL.md`、`src/v7/empirical/analyze_states.py`；数值设置由下列 E 的 `protocol` 给出。正文保留直观定义和一个评分式即可，不需要在实验章重讲 KKT。

## 来源缩写与优先级

- **E** = `docs/V7_AUDIT/empirical_summary.json`：当前 KuaiRec 正式评分结果、原比较区间、窗口、参数选择。优先于旧报告中的手写近似数。
- **EA** = `docs/V7_AUDIT/empirical_acceptance.json`：从保存预测独立重构结果、样本计数和执行验收。
- **D** = `docs/V7_AUDIT/empirical_new_work_group_diagnostic.json`：读取正式结果后增加的固定预测诊断；没有重拟合，必须标记事后。
- **Y / YA** = `docs/V7_AUDIT/openchannelstats_empirical_summary.json` / `openchannelstats_empirical_acceptance.json`：独立 opt-in YouTube 小源及重构/集中度诊断。
- **L / LA** = `docs/V7_AUDIT/dynamic_alpha1_summary.json` / `dynamic_alpha1_verification.json`：结果后追加的线性评分理论算例；不是经验估计。

以下表中每个保留数字均有键。正文只需选取显示精度，不再重新计算区间；不能用报告里不同四舍五入值拼表。

## 一、正文必须保留

### A．实验实际做了什么：同一池的受控预测比较

| 项目 | 准确事实 / 设置 | JSON路径与键 | 池、时间及解释 |
|---|---|---|---|
| 输入规模 | 343,341 视频日、10,728 视频、8,369 账号；2020-07-05—09-05 | E `source_profile.rows/videos/accounts/date_range` | KuaiRec 来源记录；不是全平台普查 |
| 测试规模 | 每模型190,568视频日；28个不同测试日；4,315首日新作；8,107观察账号 | EA `rows_per_model/unique_test_dates/new_works_per_model`；E `influence_dual_minus_work.accounts_observed` | 两方向的目标作品互斥；曝光计数仅作权重，不当独立样本数 |
| 分割与截止 | 固定视频哈希分两组，反向复核；4起点，每次训练14日、校准7日、测试7日 | E `protocol.split_salt/source_directions/origin_start_indices/training_days/calibration_days/test_days`；各日期见 `origins_and_directions[*].training_dates/calibration_dates/test_dates` | 无未来存活筛选；当天实际有行才进候选池，不把缺行补真零。两方向不是独立复制 |
| 状态更新 | 7日半衰期、7个item-day当量先验；新作m=1 | E `protocol.memory_half_life_days/prior_item_days`；EA `new_work_memory_equals_prior` | 在t日先生成特征再吸收t日结果；测试期可用已实现前日反馈，是一步滚动预测 |
| 对照公平性 | 四族共享年龄/时长基准和α网格；无状态/仅作品/仅账号/双状态分别3/9/9/27候选 | E `protocol.alpha_grid/work_gamma_grid/account_beta_grid`；`origins_and_directions[*].models[model=…].grid_size` | 仅校准选择；复杂族更易过拟合，不因嵌套就必然测试胜出 |

正文评分式可写 `z=(e/μ)m^γ，T_a=Σz，s_v=h_a^β T_a^(α−1)z_v`。账号聚合为 `h_a^βT_a^α`，因此有明白的模型—测量接口。它是评分形式对应，不是理论库存/资本转移得到识别。来源 E `maximum_relative_account_score_bridge_error` 与实验协议；极小数值误差放附录，不占正文。

### B．主结果：自己的作品记忆很有用，额外公共信息增益很小

| 四族视频交叉熵（同一全目标池） | 准确值 | 源键 |
|---|---:|---|
| 无状态 | 7.341394896 | E `pooled_models[model=no_state,subset=all].video_joint_log_loss` |
| 仅作品记忆 | 6.626575301 | E 同键 `[model=work_only,subset=all]` |
| 仅账号共享 | 7.288812582 | E 同键 `[model=account_only,subset=all]` |
| 双状态 | 6.626126895 | E 同键 `[model=dual_state,subset=all]` |

| 要回答的比较 | 损失差及95%区间 | 源键与可写结论 |
|---|---|---|
| 作品记忆是否有信息？ | work−none = −0.714819594；日期区间 [−0.783920890,−0.649869501] | E `comparisons[model=work_only,reference=no_state,subset=all,evaluation=joint].delta/paired_date_bootstrap_95`。当前预测任务中作品过去记录增加信息 |
| 控制作品记忆后共享信息是否仍增量？**预定主比较** | dual−work = −0.000448407；日期 [−0.006534892,+0.004934549]；起点 [−0.001217129,+0.000043232]；账号 [−0.018963666,+0.015765999] | E 对应 `comparisons[model=dual_state,reference=work_only,subset=all,evaluation=joint]`；账号区间为 `influence_dual_minus_work.fixed_prediction_account_bootstrap_95`。不能宣称控制自身记忆后普遍账号资本成立 |

**区间统一说明：** E `protocol.bootstrap_repetitions=1000`；整日配对先合并两方向，起点以整个测试窗为块；账号区间为固定预测账号重采样。都不含重拟合、选样或全部观看者/时间依赖，更非因果置信区间。正文写一次，不要每段堆防御文字。

### C．新作次级结果：没有自身过去时，共享信息确有有限价值

同一来源和四窗口；只取上传自然日实际行，在**新作子池**重新归一化。仅账号与无状态都不含目标新作自身反馈。与上表全作品主比较是不同问题，不能互相替代。

| 结果 | 数值 | 精确来源与结论 |
|---|---|---|
| 新作条件交叉熵 | 无状态4.048580914 → 仅账号3.955175969 | E `pooled_models[model=no_state/account_only,subset=new].video_conditional_log_loss` |
| 仅账号−无状态，新作条件差 | −0.093404945；日期95% [−0.142058276,−0.042077012]；起点 [−0.116424662,−0.057963665] | E `comparisons[model=account_only,reference=no_state,subset=new,evaluation=conditional]`。正文总差和区间统一用此正式比较；这是预定模型比较的次级切片，不叫独立确认 |
| 来源历史可用性，**结果后诊断** | 有历史710新作 / 49.8788%新作曝光；无历史3,605新作 | D `groups[model=no_state,group=available/unavailable].works/actual_new_exposure_mass`。来源可用不是训练见过、不是注册年龄 |
| 有源历史组内 | 2.526059261 → 2.437921703；差−0.088137558；日期95% [−0.165727425,−0.002624190] | D `groups[model=no_state/account_only,group=available].within_group_conditional_log_loss`；`date_intervals[group=available,metric=within_nll]`。共享来源信息不只是整体给这组多分概率，也改善该组内部预测；仍是关联 |
| 无源历史组内 | 3.789188647 → 3.789188647，约0 | D 同键 `group=unavailable`；h回共同先验，该组内部未新增信息 |
| 新作评分分解 | −0.093404945 = 组间−0.049442990 + 加权组内−0.043961956 | D `account_only_minus_no_state_new_pool_decomposition.joint_in_new_pool_nll/between_nll/within_nll`。这是评分恒等式，不是资本因果贡献比例 |

**避免混表：** D 的全新作条件差用另一个事后bootstrap种子，区间为 [−0.141818587,−0.041220584]，键 `date_intervals[group=all,metric=joint_in_new_pool_nll]`；与 E 略不同是抽样实现不同，不是新模型改善。正文采用 E 总差区间，D 只报告分组及分解，并标明事后诊断。不要混入 V5 的 HQF−H 分解（其指标、特征和分组均不同）。

### D．α结果：必须纠正对“实际数量压缩”的预设

| 结果 | 精确来源 | 能写 / 不能写 |
|---|---|---|
| 候选α∈{0.5,0.75,1}；4族×8个起点方向共32次选择，全部为1 | E `protocol.alpha_grid`；对 `origins_and_directions[*].models[*].selected_alpha_gamma_beta[0]` 逐项计数；上下文8格是4起点×2方向 | 能写：在当前有限网格/候选池里，校准未选择数量压缩。不能写：平台真实α已估为1，凹规则不可能有作用，或所有数量效应为零 |
| 理论线性补充：另24情景，前瞻行动24个均区别于myopic；所有所选两期路径未新增号 | L `summary.conditions/foresight_changes_first_action`；`results[*].first_actions[*].opened/second_actions[*].opened` | 若正文保留理论数值，只用一句：前瞻生产值得区分，但本线性算例未呈现拆号；不能把这24情景和上述32次校准当同一实验 |
| 其中22纯/2混合首期解 | L `summary.conditions/first_mixed_conditions`；LA `pass/maximum_independent_deviation_gain` | 不许写24个全为纯均衡；细节移数值附录。有限参数情景不是全域定理，未由曝光数据校准 |

### E．YouTube：只复制作品记忆，范围必须小

| 结果 | 数字与不确定性 | 源键及池 |
|---|---|---|
| 入选与测试 | 7公开入口中4合格；960视频日、7,824 impressions；测试2026-10-27—11-02 | 来源入口数：`docs/V7_AUDIT/openchannelstats_source_pool.json.public_channels_in_manifest`；Y `eligible_channels/channels[eligible=true].test_window/test_rows`；YA `test_video_days_per_model/test_impressions_weight`。自愿公开第三方CSV，非YouTube官方研究集或随机平台样本 |
| 年龄→自身记忆 | 2.328489450 → 1.818669024；差−0.509820426；日期95% [−0.569385111,−0.440325442] | Y `pooled_within_channel_scores[model=age_only/age_times_memory].test_log_loss`、`age_memory_minus_age`。各频道自身日池归一，合并只合并评分 |
| 更简单前日基准 | 1.718326421，优于平滑记忆 | Y `pooled_within_channel_scores[model=previous_day_with_prior].test_log_loss`。必须同时给出，不能暗示状态族优于所有简单预测器 |
| 集中度与敏感性 | koura占73.6324%权重；固定评分去掉该频道仍改善−0.326295029 | YA `largest_channel_weight_share`、`leave_one_channel_scoring_only[excluded_channel=koura_yt_mfun].fixed_prediction_delta`；Y每频道 `models` 可核4频道点方向均改善。没有重拟合，不证明代表性 |

只用一句结论：“有限 opt-in impressions 小样本也表明目标作品自己的过去信息有预测价值。”**不能写复制了公共账号资本或α=1**：频道内日份额里，同一频道公共 `h^β` 与 `T^(α−1)` 均约掉。没有多账号共享状态识别，也没有共同完整市场预算。缺失impressions不补零，Koura严重缺失需在数据表注交代；只有少数日期，bootstrap不创造更多独立频道。

## 二、只留必要负结果，避免只筛正结果

| 正文必须怎样保留 | 证据与源键 | 最简摆法 |
|---|---|---|
| 主比较不稳健且集中 | E `influence_dual_minus_work`：152账号改善、7,822变差；去前1个曝光账号评分贡献后差+0.004643306 | 主结果旁一句；声明固定评分删除、未重拟合/归一，不是移除作者后的市场 |
| 最好的全体模型并非新作最佳模型 | E新作条件CE：双4.017787306，仍差于仅账号3.955175969；E `comparisons[model=dual_state,reference=account_only,subset=new,evaluation=conditional].delta=+0.062611337` | 新作结果表保留双状态列，不只放最有利两列 |
| 双−作品的新作差不是无条件成功 | 新作联合+0.001403471（E同筛选 `evaluation=joint`）；条件−0.030793608，但日期区间跨0（E `evaluation=conditional`），起点区间不跨0 | 如果正文谈“控制自身记忆后的新作改进”，必须一起写指标与两种时间区间；若不谈该支线，可移补充结果表，不能以新作仅账号正面结果替代主比较 |
| 新作改善非作者普遍获益 | D `influence`：1,116新作账号损失改善、2,631变差；去前1/5/10评分贡献差−0.078933628/−0.048130356/−0.039968713 | 只留一句：对大账号敏感但非单一账号驱动；这些是预测误差，不是作者福利 |
| 一步状态扩展没有胜过简单延续 | E `observed_one_step_transition_predictions[*].item/account.models[model=carry_forward/ar1/ar1_plus_observed_states_supply].mse_log1p_exposure`，4窗均carry最低 | 正文若讨论“状态转移已验证”，必须纠正为“仅一步观测曝光预测”；无需把整张ridge表放正文。没有验证结构B/H转移 |
| 不能识别的对象 | 无owner关联、真实注册、AI采用/外生成本/规则冲击；理论B/H与m/h单位不同 | 一个集中局限段，避免每段冗长免责声明，同时不得删去 |

关于区间：新作710组内结果属于事后切片和固定预测诊断，没有处理全体多重探索；可写“在该切片有改善”，不要写确认了普遍资本机制。

## 三、移附录 / 删正文（原档必须保留）

| 记录 | 为什么不承担本篇正文主证据 | 建议位置 |
|---|---|---|
| V1 Civitai描述、提示词重复、假想α回放（`AI_CREATOR_GAME_REPORT.md`、`PROJECT_SUMMARY_ZH.md`、`analysis_summary.json`） | 记录数/重复不是曝光；回放由假设函数决定，不能证明真实规则或拆号 | 数据探索附录；正文最多一句最初动机，不复制大表/原始数字 |
| V2大Civitai/DiffusionDB/model-trends与用户名对齐（`V2_DATA_AND_VALIDATION.md`、V2_AUDIT） | 增大N不补owner、成本、曝光；生成图像/请求/模型版本单位不能混合 | 数据来源筛选和测量附录；删除正文下载规模竞赛 |
| V3原数量/历史模型比较（`V3_DATA_AND_THEORY_VALIDATION_ZH.md`、V3_AUDIT） | 历史份额信息集更丰富，原固定池和窗口不同；旧KuaiRec按未来行存在筛选 | 只作“历史为什么预测得好”的动机；不与V7绝对损失排同表，不将旧α结果和新α结果写为算法变化 |
| V4历史资本连续推导（`THEORY_REFINEMENT_HISTORY_V4.md`） | 由预测胜出到不可转移资本多一步未识别假设；缺继续推荐旧作的状态 | 理论演变附录；正文改用当前清楚的状态定义，不保留旧假设为实证结论 |
| V5 KuaiRand跨seed/错配、旧KuaiRec主/次级、HQF新作分解（V5专项/审计） | 说明演变、局限和此前失败很重要，但池/预测器/冻结状态口径不同，不能与V7拼成一个实验 | 一段动机/独立稳健性附录；尤其不要把V5 HQF−H的组间解释套到V7 account−none新作上 |
| 旧连续单期公式、整数门槛、19/56/case148表（V2/V5理论） | 不直接回答当前作品/共享信息的实验问题；参数并非实证校准 | 原理论附录保留；正文不反复推KKT、枚举界和数值容差 |
| V7长分段分配、完整Bellman/离轨选择、48/24情景逐格矩阵 | 若正文中心是经验预测，长推导遮住实验做什么；计算正确不等于平台机制为真 | 技术附录。若课程要求博弈主章，正文保留玩家/策略/状态/收益、两期假设及一张条件模拟表，与真实数据分章 |
| 发布日旧作均值的弱关联（Y `channels[*].old_publication_association`） | 频道方向不同，两频道无发布日对照，覆盖和需求内生；不识别蚕食 | 描述性附录，不写成主发现 |
| SSH、下载失败、字节增长、hash/复制、纯实现修复和长AI流程披露 | 证明执行与复现，不解释实验的科学贡献 | 复现/数据声明附录；正文一句数据/代码可核查。保留全部原始审计，勿物理删除 |

已核阅上述演变主文档、V5和V7专项及最新JSON。当前仓库没有独立命名的V6证据包，不凭版本号补造一轮实验。

## 给 dpsk 的硬约束与最终写作建议

1. 标题和摘要围绕**实际完成的比较**：作品记忆与跨作品共享信息；不要标题宣布AI采用、算法强化或拆号的因果发现。
2. 实验章先解释研究问题、每个对照排除什么，再给数据、时间轴、术语、四族、指标和结果；读者不需要先阅读版本史或推导包。
3. 最少正文三张表：数据/窗口与定义；四族全体结果并列主差值CI；新作比较＋有源组内结果。外部YouTube用一张小表或一段，α选择写在主结果限制中。
4. 理论与实证分开：固定平台规则下求解创作者模型，是条件数值实验；真实数据检验预测代理是否有信息。没有执行的实验不填结果，预测参数不塞回结构模型当估计值。
5. 结论应有正有负：“旧作过去信息重要；新作有有限共享信息；控制自身记忆后的全体增量不稳健；当前网格不支持预设实际数量压缩；外部证据仅支持作品持续性。”这比“我们做了很多分析且都验证理论”准确，也更像完整实验报告。

# Prefix5/P10 参考复刻独立审计

审计者：独立子 agent `reference_replica_exact_audit`。范围：只读本地源码与既有小型证据；未 SSH、未加载大 tensor、未 forward、未启动/停止训练，也未修改实现。

审计目标目录：`/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018`。
最终实际 HEAD：`7679bb8ced2c13922fa9f5c9050ed0ce7fa54dde`。初审为 `12e43936c9d8923685b34780f18e5a9ee412b89d`，后续仅增加数据身份SHA校验和controller归档，未改算法。
参考科学 runtime：`75d536575593891876df339467da73b3c4cbdc7c`；实际基准为其 prefix-cache 故障修复 `d3a03054009fe3bf0068dfefabca941df95df101`。

## 当前结论

**准备期独立审计已验收：逐字节源码、实际目标资产、实际两份模板及其全部exact-config-delta没有发现复刻遗漏或未经授权的数学/训练协议差异。尚未启动训练或进行GPU forward，不把准备验收称为运行验收。**

最终小证明：`work/prefix5-reference-replica-independent-final-small-proof.json`。新HEAD再次比较仍为347原有文件相同、4处硬件适配。

独立程序比较了参考 Git 中所有位于 `reproduction/prefix_selection200`、`reproduction/prefix20_group_joint100`、`reproduction/prefix20_online`、`search_g1`、`verl` 目录下的原有 Python 文件：**347 个原文件逐字节完全一致，仅 4 个既有文件改变**。可复核小证据：`work/prefix5-reference-replica-independent-code-comparison.json`。

四处改动均限硬件/路径适配：

- `prefix_selection200/launch.py`：接受远端 H100 namespace/路径、每组四 rank 的共卡资源布局、每 rank 独立 grounding 服务物理设备、4 shard 检索、可配置 UUID lock 根目录。
- `prefix_selection200/a800_runtime.py`：同卡 train/aux 的 UUID 去重持锁，以及绑定到实际 H100 指标/UUID；原 A800 分支保持。
- `prefix_selection200/grounding.py`：H100 grounding refit 可见设备必须等于本组末卡 3/7，校准数据、拟合与 adopt 逻辑未动。
- `prefix20_online/retriever.py`：允许 `gpu-shards=4`。原完整 Flat index、FP16/E5/query prefix/mean pooling/top3 等算法未动。

从 75d5365 到 d3a0305 的实际运行采样变更只是在 `_build_llm` 将 `enable_prefix_caching` 固定 False，并增加对应记录；恢复控制代码新增该故障修复的显式等价校验。未改变 actor/probe 数学。

## 算法链核对

| 项目 | 源码与配置检查结果 |
|---|---|
| 初始化 | 官方 actor/ref revision `aa8e725...`；new actor0/new actor optimizer/scheduler/RNG/pool；selection probe 仅导入参考 epoch5 EMA 模型；online Adam0。prepare 移除 `resume_joint/resume_import/resume_pending_probe/source_compatibility` 等恢复入口。 |
| 探针结构 | 原 `prefix5_model.py` 完全相同；实际 actor index27 postblock 含 residual、question273 槽+5 policy 槽、原 positions/masks；完整 copied block35→masked flatten278×2048→Linear1。 |
| 前缀 | 原 sample6/keep5 capture contract；第6采样值不进入 probe/PPO/tools，补齐原5前缀。没有独立 encoder forward。 |
| 选择 | 原 `h=1-prod(p)-prod(1-p)`；stable Top512、原 ordinal tie-break、exploration0。 |
| 候选时间线 | 原 `effective_from_outer=6` 保留：1..5 为2560；从6普通5120；P10..190更新轮2560；200为普通轮且不更新probe。未把历史改成从1使用5120。 |
| P10实际选题 | 周期轮先完整续写所有2560G5，每组真实strictEM、`0<K<5`含K1；mixed优先并用probe排序，不足512按原probe fallback补齐；没有先看标签影响普通轮。 |
| Probe优化 | 原 trajectory BCE + G5 mixed BCE，权重1:1；一遍2560G5、50次同步Adam/rank、LR5e-6；完整G5同rank/step；无DIST、KL50、mixup、soft0.9、online EMA或旧消融权重。 |
| Actor目标 | 原4physical/8virtual；mini128由64个micro2 masked-token mean等权构成；globalmicro16/local4/accum8；20机会、PPO1epoch；LR1e-6/57完整outer warmup（初值0）后constant。 |
| Actor safeguards | 原DualClip3/oldpolicyKL.2/negative-tail.01/expbound30/FP64 L2clip1、FP32master/Adam/loss/reduction、BF16前向均保持。只有真实returnedAdam计数与原guard语义；不将机会冒充写入。 |
| Reward | 原Eco/soft grounding；selection correctness head不作reward。all-five-correct组仅归一化统计加virtual0，不伪造轨迹；其余组原sample mean/std+epsilon。 |
| Grounding | round0使用原实际boundary/commit heads及同300校准身份；50/100/150基于保存actor重拟合，四服务全部adopt后才能51/101/151。新设备映射没有删除该生命周期。 |
| Train/QA | 同一新BF16交互协议、context4096/action500/obs500/reserve500/earlyfinal/保留info wrapper/max4search/nativefirstactionstop；不是旧FP32 QA。 |
| 数据/诊断 | 原169028 train、seed1729/原诊断排除pool和permutation；固定512问题身份保持，周期fresh same actor G5/hidden/EM，pre/post同rollout；未用旧teacher轨迹评判currentactor。 |
| 保存/恢复 | joint actor/probe/两optimizer/scheduler/RNG/pool与grounding状态原源码保持；真实故障按latest完整joint恢复的原接口保留；此准备本身不创建旧resume入口。 |

`fixed_test_features` 变为 binding-only 路径不会使评测改成固定teacher特征：`probe.initialize` 仅以传入的 manifest SHA 检查原 warmup source binding；它不读取该路径 tensor。每个周期的实际测试由 `periodic_data.diagnostic` fresh rollout生成，并通过原 `probe.update` 的NativeFeatures加载。这一改动是删除未使用大缓存依赖，不是改变诊断轨迹。

## 硬件隔离与环境

允许差异落实为两个独立 world4：A物理0..3，B物理4..7；各自完整4shard exact检索与四个reward服务共用本组四卡。grounding服务按rank一张卡一个；重拟合前停本组四服务，refit在3/7执行。检索端口18410/18510，grounding端口各+10..+13；run/base/cache/输出独立。UUID锁对重复train/aux UUID去重，不移除租约。

现有小证据 `work/prefix5-reference-a800-environment.json` 与 `work/prefix5-reference-h100-environment.json` 中关键包逐项相同：torch2.4.0、transformers4.46.3、vLLM0.6.3、NumPy1.26.4、datasets3.2.0、flash-attn2.7.4.post1、xformers0.0.27.post2。H100/A800及FAISS分片数不同，因此不承诺浮点逐bit重现；数学协议未因此变化。

## 资产运输与实际模板最终补充核对（已闭环）

1. 原完整 warmup checkpoint SHA `b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594` 是原1,242,424,352字节容器。运输可以去掉不导入的raw/Adam/RNG，但运输容器SHA不能冒称原容器SHA。
2. 代码已在原端用同一 `warmup_model_weights(...epoch5,ema)` 提取模型，并对导出再load逐tensor比较；H100 builder会把14个EMA权重和全部nativebuffers的实际CPU字节SHA与sourceproof逐一比较。需要实际目标完成小证据确认该分支已执行。
3. 实际 `copy-complete.json` 须证实全部共享数据和头资产source-target fullSHA相同；现成H100官方模型/index/corpus/E5必须由已有身份/完整哈希证据绑定到参考，而不是只以路径相似认定。
4. 两份实际 `config.template.json/exact-config-delta.json/prepared.json` 要补查：所有未列硬件/path/新run身份/运输容器provenance/fresh状态变更字段与source相同；candidate schedule完整保持；两个slot UUID不交叉、端口不交叉。
5. 应确认仅template/prepared存在，没有launch-config、submit-intent、submit-receipt或训练进程由本准备启动。用户尚未指定两个实验，不能启动。

以上待核项已由以下实际小证据闭环；没有新读大文件或模型：

- `work/prefix5-reference-assets-copy-complete.json`：14文件、383,603,476字节，实际source fullSHA全部等于independent target fullSHA。所有数据身份与原config绑定相同。
- 原warmup完整容器SHA `b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594`；model-only运输容器310,596,618字节、SHA `06ffd8c21b6208ff5a0beb11cded62da7b9551c3e1357a1d7321e69eb06d994e`。模板和证明明确区分两种SHA。目标H100实际CPU加载后，14份EMA权重与2份nativebuffer逐tensor字节SHA全等于原端提取证据。
- `work/prefix5-reference-retrieval-assets-comparison.json`：A800当前实际完整读取10文件、79,391,554,842字节，包括完整index/corpus/E5，逐项匹配H100已有独立完整SHA。官方actor9文件也逐项匹配参考config与H100已有完整SHA。没有重复大文件hash。
- `work/prefix5-reference-prepared-small-summary.json`：A模板SHA `fac3f8b4ecbd009314d215f2fceaa9390b456ae018f3c6830eba16a760556837`；B模板SHA `508a7b24fd5ff68b683cb968e7161af2812aa290376b61e06c52a24df47a270f`。全部actual exactdelta已核，差异仅硬件/path/新run身份、运输provenance、fresh-vs-resume和准备标记。候选schedule/actor与probe超参/loss/interaction/grounding/seed/exploration完全保持。
- 两组8个实际GPU UUID互不交叉；retrieval18410/18510、grounding18420..18423/18520..18523互不冲突。均无resume字段，无launch-config/submit-intent/submit-receipt，training_submitted=false，experiment_assignment_pending=true。
- `work/prefix5-reference-code-stage.json`：7679bb8c共1395目标源码fullSHA通过，未启动训练。

**最终结论：准备已通过独立复刻审计，等待用户指定两组实验。未发现需修的数学/算法遗漏；未来授权启动后的真实GPU加载、采样与优化推进属于尚未执行的运行验收，不在此次准备验收中冒称完成。**

# 纯本地逐轮成本账本工具

调用文件：append_nested_local_costs.py。仅Python标准库，不含SSH、subprocess、模型或tensor加载、训练控制。它不会读写STATUS、REPORT，也不启动任何watcher。

实际FULL19发布后，主线可执行（本次发布没有执行此调用）：

    python3 -B /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-speed-repair-20261008/nested-local-cost-controls-source/reproduction/tree-eval-resume-20261013/local-control-backup/append_nested_local_costs.py \
      --base /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-nested-branch-handoff-20261013 \
      --receipt FULL_OUTER19_PROGRESS_RECEIPT.json

后续只替换receipt轮次。缺少receipt时返回PENDING且不写账本。输入仅为base内明确命名的小型JSON；它核对FULL绑定的immutable poll、native readback和resource SHA/bytes，再从本轮step/candidate_outer读取事实。不会用latest metric替代本轮。

已完成的1–20轮块文件保持原字节；后续追加从21开始新块。后续工具创建的active块可追加，接近100000字节则开新块，不丢标量。保留v3索引、原字段名、完整scalar、来源/config/data/model身份、真实有效ancestry和历史失败成本分栏。新块使用原字段名映射，旧vector仍按其原fieldsets解码。既有两个小型本地镜像沿用账本mirrors配置。

输出为PER_OUTER_METRICS_COMPACT.json/md、新scalar块，以及CUMULATIVE_COST_THROUGHN.json/md（实际文件名例如CUMULATIVE_COST_THROUGH19.json）。历史累计文件不覆盖。累计若超过100000字节，会把完整证据字段存入cumulative_cost_blocks的小型分片，主JSON保留每片SHA、字节、字段名和顺序，字段不被删除。每个JSON块限100000字节，单条异常超限会报错，不静默截断。

未知dev、每rank机会字段、candidate outcome分摊、padding分摊均保留null；成功写入、empty-minibatch skip、完整metric机会、独立rank机会分别记录，不互相推算。native writer累计计数已含祖先，不重新相加。历史失败2/6、eb6 bootstrap与整服务CPU范围不加入有效轮分阶段成本。

完整native200存在而200的full metric缺失时，可以记录原生完成证据，但full_metric_status保持PENDING_CURRENT_OUTER_FULL_METRIC，缺失的评分/方差/完整outer耗时仍null，累计对应total为null并保留known_sum与缺失轮次；不得借199轮指标或dev。若已有当前metric却step或原receipt不一致，明确报错，不把坏记录降成pending。

8×已知完整outer墙钟只列物理共享容量上限。GPU实际活跃/独占账单/能耗仍null；stage、正常dev、整服务与失败子阶段不重复相加。

同一轮同一receipt SHA再次调用为幂等；不同receipt覆盖已计轮次会拒绝，需另行明确修正凭据。非终轮要求下一轮已进入；目标200不要求启动201，但native driver完整next_step仍按实际metadata绑定。所有新原始证据为固定receipt/poll/native/resource哈希，动态CURRENT/STATUS不是固定证据。

本次补账接入仅AST语法检查，没有执行FULL20调用、没有既有账本重写或新增训练信用，也没有review/测试/训练节点调用。执行权限和实际训练验收仍由主线掌握。


固定补账接入：累计生成默认读取base/CANDIDATE_COST_BACKFILL_THROUGH19.json，固定SHA d219a4f1af593d6dc304154e115aab218168113974109ce032d0b8402c568699；可显式传--candidate-backfill-index及--candidate-backfill-sha256。逐份核对part SHA/bytes、outer、运行source/config、原audit稳定SHA与controller身份，仅向本次累计的内存副本加入原schema引用，不改旧块。原1/2来源与新3..19每轮只计一次。引用来源及匹配证明保存在candidate_cost_backfill_join与resolved_fixed_cost_inputs。缺少20的逐条补账时，20对应outcome/branch评分覆盖仍未知，保留known_sum/known_outers/missing_outers，不填0、不猜padding或独占wall；也不会读取远端补齐。


20–200轮候选补账：backfill_nested_candidate_costs.py接受--first-outer与--last-outer，仅接受显式20..200范围，默认3..19保持旧接口/已存历史。先从本地v3已计轮次与immutable FULL/poll/native/resource SHA构建固定pin，未计轮次跳过且不SSH；只读取指定778/config355本任务路径的audit、controller及resolved config，status均COMPLETE且稳定才返回小scalar。原始逐条账本留H10018；同一轮既有源pin不同则报错不覆盖。

    python3 -B backfill_nested_candidate_costs.py --base /绝对路径/outputs/tree-nested-branch-handoff-20261013 --ssh-helper /绝对路径/work/tree-speed-repair-20261008/ssh_retained18.py --first-outer 20 --last-outer 20

每个实际轮产生immutable CANDIDATE_COST_EXTENSION_OUTER0020.json/md及candidate_cost_backfill/CANDIDATE_OUTER0020.json；后续替换轮次。每个JSON<=100000字节。CANDIDATE_COST_EXTENSION_INPUTS.json是本地派生输入列表，记录每份immutable index SHA/bytes。append的后续累计默认读取此列表，逐份验证index/part/outer/source/config及原FULL/poll/native SHA，只在内存累计副本中引用每轮一次；列表本身仅按本次读取SHA记录，不冒充原始证据。可用--candidate-extension-inputs显式指定列表。未完成/尚无候选part的21等轮仍unknown，不借上轮成本。旧through20/块及3..19补账保持原字节。origin沿fresh母链，非从基座清零；padding结果分摊、独立结果wall和实际GPU计费不推算。

若FULL轮次的固定poll已经只显示下一轮controller，可对一个已计入完整轮显式添加`--supplemental-controller-poll NATIVE5_CONTINUE_POLL_533.json`。补充poll必须是同source/run/unit/fresh/config、同attempt/outer且COMPLETE、时间不晚于原FULL观测；它只是此前采样证据，不是更新完成凭据。FULL/native绑定保持原样，远端仍只读固定本轮audit/controller/config，并核对实际controller与补充poll字段一致。补充poll SHA/bytes单独保存在candidate source_pin；后续累计重新验证该SHA，只在累计内存副本补齐原未知controller字段，旧FULL/CUM/块不回写。46示例为`--first-outer 46 --last-outer 46 --supplemental-controller-poll NATIVE5_CONTINUE_POLL_533.json`；已存46累计仍unknown，补充供47及以后使用，不加训练信用。
# Historical107–118 readback bridge

`collect_nested_dev_costs.py --base <audit-base> --ssh-helper <ssh_retained18.py>`
with this task's explicit `TREE_MGMT_CONTROL_PATH` reads only24 fixed completed
dev5..120 control JSON summaries pinned by immutableCUM120. It preserves original
summary bytes and native counter names in a separate supplement, never rewrites
CUM120 or double-adds validation wall/processCPU. It does not read JSONL/PT.

`backfill_nested_historical_full.py --base outputs/tree-nested-branch-handoff-20261013`
publishes only immutable historical FULL107–118 from the actual gap107–119 scalar
readback and existing FULL119 anchor. Every receipt explicitly says
`RETROSPECTIVE_SOURCE_READBACK`. Its `source_poll` remains the original live119
observation, while metric/rank/controller/dev source selectors bind the actual
gap SHA and original remote file hashes. No synthetic poll or current-state
write is produced. Cumulative counters and attempt accounting come from each
nativeN persisted guard; no independent tensor read/cold reload is claimed.

The appender recognizes only this explicitly marked historical evidence and
reuses the published pure metric/rank/native/restore validators. Run the usual
appendN, credited candidateN collection, then ratesN sequence. Unknown candidate
details stay unknown until a subsequent cumulative snapshot joins the immutable
supplement; old cumulative snapshots are never rewritten.

## Completed dev boundaries120–200

`collect_nested_dev_costs.py` now accepts optional `--through-outer N` for
120,125,...200; default120 retains the old CLI. It requires the corresponding
immutable `CUMULATIVE_COST_THROUGHN.json` and `FULL_OUTERN_PROGRESS_RECEIPT.json`
with actual completed dev512 (NQ256+Hotpot256), original778/355 run identity,
and complete eight-rank native metadata. Missing completion/dev evidence returns
`PENDING_ACTUAL_COMPLETED_DEV_EVIDENCE` without SSH or an output write. Final200
does not borrow199 metrics/dev; if actual dev512 is absent it remains pending.

Only after realFULL125, CUM125 and dev125 exist:

    env TREE_MGMT_CONTROL_PATH=/tmp/tree-mgmt-3fw9a1ab/control python3 -B /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-speed-repair-20261008/nested-local-cost-controls-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py \
      --base /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-nested-branch-handoff-20261013 \
      --ssh-helper /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-speed-repair-20261008/ssh_retained18.py \
      --through-outer 125

Subsequent actual boundaries replace125 with130,...200. Each new boundary creates
independent immutable `DEV5_N_NATIVE_COST_SUPPLEMENT.json/md`; original5–120,
shared original control JSON bytes and previous supplements are never replaced.
The nearest earlier supplement is reused only after its SHA, original control
SHA/bytes, source/config/model/data/outer/dev identity and both mirrors match.
Each original row retains its original observation metadata; only new missing
fixed control JSONs are read on H10018 with nice15 and the existing task route.
Original rows exceeding the small summary budget use complete-field parts with
SHA/bytes/selectors, each below100000bytes; no required scalar is dropped.
An existing identical supplement is verified locally and returns without SSH.

Dev5 remainsba9/22be, laterdev778/355. Original official model/data identity and
actual per-dataset counts are fixed; independently uncaptured split-file SHA
staysnull. HTTP calls/queryevents remain separate, PAD==EOS nonpad counts do not
become precise engine emissions, and unknown scoring calls remainnull.
Validation wall/processCPU are already contained in outer/service scopes and
must not be added again. Collection failure preserves one error receipt, makes
no automatic retry and infers no training fault. Range publication only used
AST/CLIhelp checks and did not collect future125 or run an evaluation.

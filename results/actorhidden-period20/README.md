## Current LR1e-5 execution: actual periodic40 and continued42

At UTC2026-10-16T22:48:55.105035+00:00, same3426/h100-8 completed42/200: actor840/probe200 actual Adam per each8rank;43 continuation has started. All32 original actor ledgers39-42 have20 writes each. Periodic40 independently read all8 probe ledgers:100 new writes/rank at1e-5,640 groups/3200 trajectories per rank,80 global255 and20 global260 batches. Fullpool5120G5 was completed once, with preselected512 projected without new generation. Heldout512G5 uses the same samplingactor0039 and has zero qid/content overlap with training. Actual41 selected with the newly committed probe33c7e662857511b34d041d0b9a483896ba0358c1145b5333d04d120f18c5fe3c. Frozen runtimeDCC is unchanged; readonly observer15920 naturally exited0 after this proof.

|Same-actor paired metric at40|Train before|Train after|Heldout before|Heldout after|
|---|---:|---:|---:|---:|
|Mixed accuracy|46.88%|71.25%|43.55%|62.50%|
|Balanced mixed accuracy|55.38%|64.86%|54.59%|51.20%|
|Mixed recall|71.62%|52.67%|74.34%|30.97%|
|Nonmixed recall|39.14%|77.06%|34.84%|71.43%|
|Trajectory BCE|0.618179|0.749956|0.671492|0.881458|
|Group BCE|0.951811|0.568302|1.001965|0.678385|
|Total BCE|1.569991|1.318258|1.673458|1.559843|

Trainpool mixed1219/5120=23.81%; heldout113/512=22.07%, nonmixed-majority399/512=77.93%. Selected512 contains166/512 mixed=32.42%, fixed before outcomes. After this update, train balanced accuracy improved while heldout balanced accuracy declined54.59% to51.20%. Heldout ordinary accuracy rose43.55% to62.50% by shifting toward nonmixed, still below the majority baseline. Heldout mixed recall fell74.34% to30.97%; group BCE improved but trajectory BCE worsened. This update does not demonstrate heldout mixed-discrimination improvement, stable selection benefit or QA gains. Periodic20 and40 use fresh trajectories from different actor snapshots, so their heldout values are not a fixed-label learning curve.

Recorded periodic40 main phases total846.576s: prefix80.237,score27.116,fullpool447.792,diagnostic123.086,actor127.464,probe phase40.880. The100-Adam loop alone takes8.089s; neither it nor the phase sum is a directly measured complete-to-complete wall-time interval. Model checkpoint SHA is producer-recorded; this metadata readback did not independently hash or reload the932251350B probe tensor file. Latest restore candidate is42 or later actual complete state, not archived20/source5 for convenience. QA50/future checkpoints/independent backups/final51713/native exit0 remain pending. Small full readback: lr1e5-periodic40-readback.json. No weights or full trajectories moved to Mac/Git.

---

## Current LR1e-5 execution: first actual periodic update

Same3426 on h100-8 completed20/200, actor400 and probe100 actualAdam per each8rank. The new probe performed100 writes/rank at1e-5, visiting all5120 groups/25600 trajectories once with80 global255 and20 global260 batches. All8 original ledgers and fullpool/heldout token-index/strictEM bindings were independently read as CPU metadata. The512 heldout questions and normalized contents have zero overlap with the training pool. Frozen runtime remainsdcc8715d; only the human-authorized probeLR changed from the earlier execution.

|Same-actor paired metric|Train before|Train after|Heldout before|Heldout after|
|---|---:|---:|---:|---:|
|Mixed accuracy|61.80%|62.91%|56.25%|52.54%|
|Balanced mixed accuracy|61.26%|69.19%|54.60%|58.80%|
|Mixed recall|59.95%|84.53%|50.00%|76.22%|
|Nonmixed recall|62.57%|53.84%|59.20%|41.38%|
|Trajectory BCE|0.628780|0.521055|0.706191|0.619924|
|Group BCE|0.707169|0.710282|0.743348|0.876770|
|Total BCE|1.335949|1.231336|1.449539|1.496694|

The training pool contains1513/5120 mixed groups(29.55%); fresh heldout contains164/512(32.03%), with nonmixed-majority baseline348/512=67.97%. Top512 fixed before labels contains251/512 mixed(49.02%). The update improved balanced accuracy and mixed recall while reducing heldout ordinary accuracy and worsening group/total BCE. This is a tradeoff in one paired update, not stable quality success or isolated evidence that the LR change is better than historical1e-6. Historical3425 uses different actual rollouts and remains separately labelled below.

Actual21 selection used new probeSHA9445d314f95d51d28d74f73e28467b4780c41a5232a083c851d5bef72257ad3d with actor0020. Its generated selected-group receipt reports233/512=45.51% mixed versus mean predictedq93.07%; these are selected samples, not fullpool/QA accuracy, and do not establish a causal before/after selection benefit.

Actual19-to20 complete boundaries span836.506s(13m56.5). Recorded main phases sum825.472s. Probe update phase42.503s includes paired evaluation and persistence; the100-Adam loop alone takes8.163s. Ordinary resumed6-10 boundaries averaged319.145s. The100-step scalar curve is an online pre-update series across different batches; its first/last windows cannot replace the full same-head postupdate paired evaluation.

The requested complete20 state was retained before normal rotation:28 declared assets/39226733741B in preparation/retained-joint20-once-v1/checkpoint. A separate read verified all target device/inode/size/mtime identities, relocatedJOINT/mapping and committed pool consumed102400/next21; original retention-worker birth is gone. This is same-disk preservation, not an independent backup or large-file SHA/semantic reload. Source5 independent backup remains complete; future50/100/150/200 and final retention/QA51713/exitcleanup remain outstanding. The readonly periodic observer ended normally after20, preservation and actual21 newprobe use; the same native training continues200/no201.

Small records: lr1e5-periodic20-readback.json, lr1e5-periodic20-loss-curve.json and lr1e5-periodic20-retention-summary.json. No weights, optimizer tensors, features or full predictions were moved to Mac or Git.

## Requested complete20 retention control

A separate CPU-only worker waits for the real joint20 completion and hardlinks its declared actor HF files, all8 Adam/RNG states, driver, complete probe100/Adam/RNG state, committed pool snapshot, config and source identity into preparation/retained-joint20-once-v1/checkpoint. JOINT.json relocates resumable paths and RESTORE-MAPPING.json records source→retained names. This preserves actual inodes if normal rotation later unlinks their original names. It does not alter frozen training code, budgets, loss or optimizer; no tensors are forwarded/deserialized or large model files rehashed during pinning. Same-disk retention is explicitly not an independent backup or a full SHA/semantic validation claim.

The start controller owns exclusive INTENT/RECEIPT and never launches a second worker on replay. The worker polls only a local commit file, uses noGPU and finishes after20 or a bounded6hour wait. Missing/changed assets produce a retention error while preserving the source; training is never signalled. Four CPU tests passed: hardlinks survive source unlink with all fixture bytes intact; partial clocks/missing ranks are rejected; absent joint20 stays pending without an intent; mere complete-file creation before atomic latest-joint publication also remains pending. Deployment/arming was subsequently observed as recorded above; actual20 preservation remains pending until real joint20 and the archive result exist.

A subsequent direct alias read recovered actual3426 RUNNING complete13/actor260 per rank/probe0 and14 actor training; earlier first-hop failures did not establish a native exit. No global SSH/auth/network changes were made and no training restarted.

---

## Actual resumed6–10 plus11 acceptance

Same3426 h100-8 completed6–10 after the retained5 import: all40 original actor rank ledgers independently read,20 writes each, cumulative200/rank at10, no abandoned partial stitching. First6 actually restoredAdam100/scheduler5 on all8 ranks; all40 prefix rank receipts capture current samplingactor native layer27 with no independentencoder forward. Probe remains0 as required before20. All5 selected G5 sets used the new serialized probe0 binding with LR1e-5;14 tensor values are unchanged from source5. Actual optimizer updates at1e-5 are due at20, not claimed already.

Following11 actually generated native prefixes; root observer23657 naturallyexit0 after acceptance. The same training process continues200/no201. Runtime remainsdcc8715d; these later result commits do not hotchange training. Source5 QA512 was reused rather than rerun. Actual consecutive complete boundaries6→7→8→9→10 give319.145s mean,315.630–324.591s range (about5m19/outer), excluding initial recovery and scheduledQA.

Selected actual mixed fractions6–10:67.77%,62.30%,62.89%,59.38%,64.65%, versus mean predictedq around89%. These precede the first newprobe update20 and cannot be attributed to the LR increase. Unselected4608 on ordinary rounds have no full labels, so no fullpool accuracy is reported. The full200, laterQA/final51713/count, future savepoints and newbranch retention remain incomplete.

---

## Human-authorized retained5 restart with probe LR1e-5

The user stopped3425 and, after actual20 actor/Adam assets were found already rotated, explicitly selected retained complete5 and confirmed that only the probe learning rate changes. Source3425's7754 original process births/cgroup/ownedGPU are gone;518 original log inodes were read in full. Slurm had already purged its record, so no native terminal exit code is invented. The successful scoped scancel receipt plus absence from queue and actual resource release are recorded. Latest old complete25 and all6–25 traces remain historical and are not counted on the new execution.

The source5 independent fullSHA/CPU proof is reused without rehashing38GB. The new importer preserves actor Adam100/rank, scheduler5, all8 RNG and question cursor25600/next6. The genuine online-Adam0 probe retains all14 tensor values; only its future LR metadata is rebound1e-6→1e-5. No warmup optimizer is imported, no actor reset and no new warmup occurs. Existing exact-actor5 QA512 is reused, rather than rerun as an extra baseline. Periodic probe schedule20/40/...200 and100updates/rank, actor recipe and all other budgets remain unchanged.

Focused CPU tests:11 passed. Deployment/GPU restore/resumed6–10 plus11 actual generation remain pending until real observations; source CPU validation and submission are not substitutes for runtime acceptance. The preserved5 backup remains complete, but all future checkpoints/QA/final51713/retention and final exit remain outstanding.

---

## Historical3425 / LR1e-6: first periodic update complete

At 2026-10-16T18:30:40.664064+00:00, same job3425 on h100-8 has completed21/200: actor420 and probe100 actual Adam/rank on all8 ranks;22 is in actor training. Frozen runtime700a0b3d is unchanged. Initial3–7+8 acceptance and the periodic20 observer both ended normally; no observer is active. Later QA and final retention remain unfinished.

Outer20 continued all5120 original G5 groups once (25600 trajectories). The preselected512 were projected from that same complete pool, with0 additional generation requests. Train/heldout question IDs and normalized contents have zero overlap. All8 original probe ledgers have100 writes/640 groups/3200 trajectories, LR1e-6; global batches80×255+20×260. Outer21 actually selected using the new probe250cc5875154c6eca92220ce45b0d279700f51cdd9a03db3a92a27d589daaaee.

|Same-actor paired metric|Train before|Train after|Heldout before|Heldout after|
|---|---:|---:|---:|---:|
|Mixed accuracy|62.66%|63.28%|56.84%|57.23%|
|Balanced mixed accuracy|62.35%|64.19%|55.61%|57.45%|
|Trajectory BCE|0.623388|0.603189|0.707628|0.693162|
|Group BCE|0.699062|0.681500|0.714742|0.724417|
|Total BCE|1.322450|1.284689|1.422370|1.417579|

Full-pool mixed1599/5120=31.23%; heldout freshG5 mixed167/512=32.62%. Top512 fixed before outcomes has261/512=50.98% mixed. This single paired update increased heldout balanced accuracy while group BCE worsened; it does not establish stable QA or selection benefit. Ordinary heldout accuracy remains below its nonmixed-majority baseline345/512=67.38%. Full confusion matrices, recalls and K=0..5 counts are in first-periodic20-readback.json.

Actual19→20 complete-to-complete wall time is 829.122s (13.82min), including periodic full-pool generation, fresh diagnostic and persistence. Recorded main phases sum to 817.850s. Probe update phase43.442s includes paired evaluation/persistence; the100-Adam training loop alone is8.381s. Earlier ordinary complete-to-complete rounds averaged326.078s. These scopes must not be conflated. Checkpoint20 metadata/producer SHA were checked; independent large-file readback and semantic reload were not performed for20.

CP5 independent backup is complete:152 source/destination fullSHA matches, final files/bytes checked, worker gone and staging absent; see checkpoint5-independent-backup-complete.json. Future50/100/150/200 and final experiment retention remain pending. Source ordinary available0; physical free including reserved590308528128B is shared. No actual IO failure or reserved-policy change was observed.

Control-only periodic_readback.py was executed under a separate preparation path, SHA36f5138299b04f77078880d620977e1b01f29384146ced0591880c564f1d39fe. It reads completed metadata/ledgers, without tensor loading, model forward, a training gate or runtime changes.

---

# Current actor hidden + periodic probe, fresh200

User goal updated1792163508 supersedes the stopped3420 experiment. This new run starts the official actor from0 and imports only the best retained warmupCP100 probe weights (SHA8df6bd445945ffbbf6a94171f24934495f0ce3ff17a3d2577bbd53d4b8fefd56). Probe Adam starts0. Old run/source tensors remain unchanged.

## Native representation

Capture the actual sampling actor layer27 post-block residual sum with real prepared request IDs, token IDs and positions. Native Qwen returns (MLP,residual), so copy their sum before the following fused RMSNorm mutates residual. The copied head keeps the293x2048 question273/policy20 layout and actual masks/positions. No independent encoder model is loaded or forwarded for scoring. Head construction uses the official architecture; its learned weights come from the probe checkpoint.

The user explicitly approved sampling21 and keeping20. A paused native request processes token20 on its21st forward; discard the sampled lookahead token. Freeze an earlier closing-action boundary similarly and flush its final retained token. Original prefix IDs/oldLP stay in PPO; no sampled lookahead is used by the classifier, environment action, or resumed prefix. Capture engines use eager execution/no prefix-cache hits to obtain every question hidden; full continuation keeps normal CUDA graphs/cache. This has a real profiling cost and no speedup is claimed before actual iteration timings. All captured BF16 tensors are streamed to remote CPU-backed uint16 mmap shards and consumed directly by the head.

## Clocks and labels

Each outer samples5120 distinct diagnostic-excluded questionsG5, scores Top512 with the pre-update probe, continues the same prefixes and performs20 actor Adam writes per rank. Actor/reward/PPO budget and legacy FP32 QA remain unchanged. Probe is frozen on other rounds; only20,40,...200 fit all5120 original candidate groups once,100 Adam per rank at the current LR1e-5 (historical3425 used1e-6), ending1000 probe Adam per rank. The original owntrajectoryBCE + ownG5mixedBCE loss weights1/1 remain, with mixed0<K<5 includingK1. WholeG5 stays on one rank/step;80 steps255 trajectories and20 steps260 average256 and visit all25600 once.

At a periodic outer, fix Top512 before any full labels, complete the full5120G5 pool once, and project the exact preselected2560 rows/oldLP for actor PPO. No selected trajectory is generated twice. Heldout512 question identities stay fixed, but G5 outcomes/native hidden are fresh from the exact same actor snapshot as the training pool. Report paired pre/post train and heldout mixed accuracy/BA/recalls/confusion/loss, with class counts; these are not fixed QA EM/F1.

## Evidence and operations

Targeted CPU tests cover native residual aliasing, authentic query ordering, off-by-one20th token, future-token masking, whole-pool natural schedules and exact no-generation subset projection. Production observations now prove native GPU capture on all8ranks and20 real actor writes per completed outer, with probe0 before outer20. These are actual runtime receipts and original rank ledgers, not a CPU-only configuration claim. Normal checkpoint readback uses checkpoint_native_once.py; it handles actor20N and probe100*floor(N/20), including the zero-optimizer probe before outer20. No standalone GPU gate or extra baseline is added.

Model/Adam/hidden/trajectories/full predictions stay remote. Rotate only consumed current-run hidden buffers and unreferenced nonmilestone actor states after joint commit; retain original questions/tokens/labels and checkpoint metadata. No old data is deleted. Native exit0,200/no201, scheduled512QA and final51713/count, all normal checkpoints/final state and resource cleanup remain required; this README is not completion evidence.


## First actual native run and necessary recovery

Job3423 completed actor outer1 with20 real Adam/rank and probe0, all8 native captures/25600 prefixes and all2560 selected continuations. Selectedmixed362/512=.70703125. Prefix phase65.769s/head score21.109s/continuation93.953s; no independent encoder forward. Native BF16 shard each3840409728B/3200 rows.

The pipeline then FAILED1:0 in storage.validate_base because a legacy10-probe-writes-per-outer guard was missed. This is an implementation error after a valid joint save, notOOM, a model failure or human stop. The guard now explicitly accepts the native period20/100-write clock; regression tests and a dry metadata deletion plan on the actual stopped source passed. No source tensors were deleted.

Full checkpoint1 independently read/CPU loaded: proofSHA79c15230ac8881294ae6620fa8408448ad3218aae0cc6ce14ea5d7bbaabc2217, actor20/rank, probe0. All356 original process births/cgroup/ownedGPU gone;31 originalinode logsfullSHA retained, terminalproofSHAda37e96e9c4250e0c14831182c109241159a73307147eb60862503b80e1379e2. Native job was later purged; the original actualFAILED observation is retained separately from the new unavailablequery.

Recovery namespace resume1 imports the complete joint1/HF/Adam/scheduler/RNG/probe0 and poolcursor5120/next2, same logicalrun; no officialreset or rerunning1. Actual GPU restore and resumed2–6+7 acceptance remain pending until observed. Runtime source before the failure remains immutable1f25965f.


## Imported-history retention correction

3424 actually restored joint1 and completed joint2/actor40 per rank/probe0. It then failed in the legacy previous-snapshot cleanup rule, because authentic imported snapshots retain their source absolute paths. Full joint2 independently read/CPU loaded (proofSHAd17a55b176cda932bf75618598bb92f6c64cd08cafc894a72c3daaa4bbaf091a); all342 owned births/cgroup/GPUgone and31 originalinode logsfullSHA retained, terminalproofSHA24dd4d41974c4e8705c637e3c5448abbc9acb1b5950473ab55441add2a2a50bf. Neither failure wasOOM or a human stop, and neither valid outer is rerun.

Imported history is now excluded from local snapshot pruning. The stopped source2 dry plan passed, selecting only8 current-namespace consumed hidden shards/30723277824B and deleting nothing. After a valid scientific joint commit, housekeeping failures now record a warning and preserve assets instead of cancelling training. Core restore/actualupdates/lossgrad/save checks still run. Job3425/resume2/runtime700a0b3d imports complete2 and actually completed3 and4: all8 original rank ledgers show20 new writes each, cumulative80 actor writes per rank, probe0. Resumed3–7+8 acceptance remains pending until all five rounds and following native generation are observed.

The separate native checkpoint backup worker/control support actor20N/probe100floor(N/20), immutable snapshot tensors, sourceCPU fullSHA proof and independently read destination SHA. Consumed native-hidden mmap caches are excluded; original token/label/prefix/PPO records stay in the backup.20MiB/s/128GiB ordinary reserve/h100-10 independent ext4/no Macmodel transit. Backup is pending until an actual milestone proof and full copy result exist. This does not alter the stopped old5cba backup worker or the frozen training runtime.

The stopped old5cba experiment's previously authorized CP5 copy finished independently at2026-10-16T15:19UTC:176files/40352590835B, all source and destination fullSHA values matched. A later read confirmed all176 final files at expected sizes, final directory present, originalPID3675664 gone. Small observed resultSHA7ba061c159135ba7446e2cb70c6163d3c0efb407cf4cf04ce2fff2c3d7122d97. This is old-run retention, not the newa008 CP5 backup, semantic reload or completion of all experiment backups.

## New-run checkpoint5 and QA

The newa008 joint5 is fully saved in resume2. Normal CPU inspection independently read all24 checkpoint files/38604951592B and loaded the actual8rank model/Adam/scheduler/RNG, driver, head and pool states. Actor100/rank, scheduler5, probe0/no optimizer before20; pool consumed25600/next6. ProofSHA2231cbf9ecca7e934ba34e924fb8705962fa1b0dcce2a2bd848b1812ccf160da. This is CPU/full-file readback, not GPU reloading or an independent copy.

QA5 fixed512 completed with8 actual worker exit0: EM.236328125, F1.292782738095238. NQ256/Hotpot256, original questionSHA3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb; legacy pre-d00 async FP32/raw500/noanswerreserve/noearlyfinal contract unchanged. These early QA results do not establish a selection benefit.

The distinct newa008 independent CP5 copy was admitted once on h100-10 using that CPU proof.152files/38806477095B; PID3908795/startticks1923025692; namespace a008d7ce9833-outer-0005-once-v1. Two subsequent observations show the same birth, source-copy2323644416B then a complete3929249504B first shard undergoing independent target SHA read. Only7 small files had verified at that observation. Status remainsPENDING, semantic_reload=false/all_experiment_backup_complete=false. Original runtime700a and training3425 remain unchanged; no model bytes transitMac.


## Resume2 continuous five-round acceptance

The same3425 completed3–7 with all40 original actor rank ledgers independently read: each has20 writes, cumulative140/rank at7, no abandoned partial updates stitched. Probe remains0, as required before20. All40 native prefix rank receipts report layer27/current actor/no independent encoder,3200 trajectories per rank. Following8 actually generated native prefixes and entered scoring, then continuation. Root observer88879 naturally exit0; the same production job continues200/no201. No new initial five-round gate is repeated.

Across completed3–7, the sum of all main phases averages320.163s. Actual consecutive complete-to-complete boundaries4–7, subtracting the separately measured QA5 phase118.715s from the affected interval, average326.078s (5m26), range316.954–330.776s. Startup/recovery and scheduled QA are excluded from that ordinary-round figure. This is a wall-time measurement, not only the short GPU update loop.

Selected512 own-label mixed fractions at3–7 were70.117%,71.680%,68.750%,68.555%,67.578%. There are no full labels for unselected4608 on ordinary rounds; these values are not full-pool classifier accuracy or a QA improvement claim. Paired full-pool and fresh-heldout classifier statistics remain due at the first probe update20.

All five current-run storage rotations completed without warnings. Source ordinary available bytes observed0; this does not by itself establish a write failure or this job's exclusive physical capacity. No reserved-block policy change or old-source deletion was made. Actual source capacity and writes continue to be observed while healthy training proceeds.


## Selected-group label distribution through10

The CPU-only selection_metrics.py reporter joins the original pre-update Top512 IDs with each trajectory's own strictEM and records K=0..5, selected mean predictedq and actual mixed fraction. It follows original immutable paths for imported history1–2. Actual collection completed all10 rounds with no missing records;2560 unique trajectories/512 G5 per round and mixed/correct counts match the producer receipts. It reads only small metadata, is staged under a separate preparation/selection-metrics-control directory, and never loads models or modifies the frozen runtime.

|Outer|All wrong|Mixed|All correct|Actual mixed|Mean predicted q|
|---:|---:|---:|---:|---:|---:|
|1|74|362|76|70.70%|88.41%|
|2|63|359|90|70.12%|88.45%|
|3|57|359|96|70.12%|88.22%|
|4|71|367|74|71.68%|88.04%|
|5|76|352|84|68.75%|88.42%|
|6|46|351|115|68.55%|89.01%|
|7|64|346|102|67.58%|88.75%|
|8|63|313|136|61.13%|88.49%|
|9|43|309|160|60.35%|88.82%|
|10|48|321|143|62.70%|89.46%|

The score overpredicts the realized selected mixing rate in these early rounds. This is observed selection/calibration behavior, not evidence establishing its cause or a reason to stop the authorized healthy run. Probe weights are intentionally unchanged until20. Full-pool and fresh-heldout paired diagnostics after the first periodic update remain pending. These selected-group statistics do not measure candidate-pool classifier accuracy or QA quality.


## Independent checkpoint5 backup complete

Newa008 CP5 finished at2026-10-16T17:56:45.270947UTC on h100-10 localext4. All152 source files/38806477095B were actually copied and independently read from the destination; every source/destination fullSHA matches. A separate final read confirmed all152 files at expected sizes, final directory present, staging absent and originalPID3908795 gone. The generated restore mapping's small targetSHA also matches. Source RESULT.json120285B was independently read in full, actualSHA47ba491466974e8cd4323d1d60de48594d06f33c53f65d0763eeb45bde44a99a. Earlier progress snapshots are historical.

Destination: /mnt/data/shaoshitong/search-g1-prefix-selection200-independent-backup-20261016/a008d7ce9833-outer-0005-once-v1/checkpoint. SourceCPU proof2231cbf9ecca7e934ba34e924fb8705962fa1b0dcce2a2bd848b1812ccf160da and frozen copier700a/72870be9c54337dea86a45d4440e6693bce05afed82dee21a90d6595ade9b008 remain pinned. No model bytes transitedMac and no training changes were made. Semantic reload=false/all_experiment_backup_complete=false; future50/100/150/200 and final retention are outstanding. Do not replay the consumed backup start or rehash these completed large files.

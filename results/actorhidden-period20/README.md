# Current actor hidden + periodic probe, fresh200

User goal updated1792163508 supersedes the stopped3420 experiment. This new run starts the official actor from0 and imports only the best retained warmupCP100 probe weights (SHA8df6bd445945ffbbf6a94171f24934495f0ce3ff17a3d2577bbd53d4b8fefd56). Probe Adam starts0. Old run/source tensors remain unchanged.

## Native representation

Capture the actual sampling actor layer27 post-block residual sum with real prepared request IDs, token IDs and positions. Native Qwen returns (MLP,residual), so copy their sum before the following fused RMSNorm mutates residual. The copied head keeps the293x2048 question273/policy20 layout and actual masks/positions. No independent encoder model is loaded or forwarded for scoring. Head construction uses the official architecture; its learned weights come from the probe checkpoint.

The user explicitly approved sampling21 and keeping20. A paused native request processes token20 on its21st forward; discard the sampled lookahead token. Freeze an earlier closing-action boundary similarly and flush its final retained token. Original prefix IDs/oldLP stay in PPO; no sampled lookahead is used by the classifier, environment action, or resumed prefix. Capture engines use eager execution/no prefix-cache hits to obtain every question hidden; full continuation keeps normal CUDA graphs/cache. This has a real profiling cost and no speedup is claimed before actual iteration timings. All captured BF16 tensors are streamed to remote CPU-backed uint16 mmap shards and consumed directly by the head.

## Clocks and labels

Each outer samples5120 distinct diagnostic-excluded questionsG5, scores Top512 with the pre-update probe, continues the same prefixes and performs20 actor Adam writes per rank. Actor/reward/PPO budget and legacy FP32 QA remain unchanged. Probe is frozen on other rounds; only20,40,...200 fit all5120 original candidate groups once,100 Adam per rank at1e-6, ending1000 probe Adam per rank. The original owntrajectoryBCE + ownG5mixedBCE loss weights1/1 remain, with mixed0<K<5 includingK1. WholeG5 stays on one rank/step;80 steps255 trajectories and20 steps260 average256 and visit all25600 once.

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

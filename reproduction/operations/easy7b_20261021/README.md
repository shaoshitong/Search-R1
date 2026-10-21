# Historical operations / control backup

These scripts preserve the exact code used from /tmp during preparation,
publication, resource inspection and the unique job745022 submission. Submission
has already happened: **never replay easy7b-submit.py**. Use job.id and current
squeue/sacct/logs instead. A pending job or observation timeout is not a failure.

The two bridge scripts read existing private credentials in RAM and pass them
only over stdin. They contain no credential values. The model stays on HPC3.
The initial broad preflight and bounded du attempts timed out and are retained
as failed observations. The successful storage scans use detached, explicitly
claimed PIDs; do not duplicate those scans. MOU cannot write the JXU-owned control
directory; the separate owned-/tmp MOU scan fixes only its audit destination.
No training code or source is changed by those audit tools.

The active training job is `745371`, using immutable runtime `e082a23b5a65b302e2016aa0e5ab43b10b543d4f`.
Its original HK CPU archive controller `3420413` exited after an SSH observation
authentication failure; its claim, failure and logs remain unchanged. The unique
replacement is under `job-745371/recovery-transport-r1-20261021`, using controller
code `0830581f16eee86d5a89100cda91b175de63635c` while all source checkpoint commands
still bind the original training runtime. Observe it with
`observe_hk_archive_transport_recovery.py`, not the historical root observer.
The recovery launcher has already been executed once. Never replay a launcher
because an observation timed out; inspect its existing claim and process birth.
No training or retrieval job was restarted for this CPU-only recovery.

The human's later 2026-10-21 cleanup instruction supersedes latest-two retention
only for the twelve explicit legacy roots in `apply_authorized_old_checkpoint_cleanup.py`.
It authorizes all old experiment checkpoint payloads there, while retaining logs,
all text, and evaluation trajectories. The 7B feature/probe root, QProbe roots,
current 7B source/recovery roots, shared model/data/index assets, and HK archives
are outside this cleanup. Use the frozen exact manifest and inspect existing
per-UID claims before any apply; never replay an uncertain deletion attempt.
The initial foreground read-only inventory timed out, so the completed inventories
were collected under separate pinned `/tmp/authorized-old-checkpoints-20261021-<uid>`
claims. No training was restarted for this inventory or cleanup.

The updated human goal also requires notification and a complete-state recovery
on clear EM/reward collapse. The read-only training observer exposes completed
outer metrics and actual fixed-validation summaries; it does not add a training
threshold, change the loss, or restart a healthy job for a single fluctuation.

The CP5 post-callback supplement uses `archive_post_callback5_local.py` and its
independent `archive_post_callback5_hk.py` CPU runner. The core archive must first
have actual fresh-source/fresh-destination SHA acceptance. Only then can the
one-shot runner preserve the seven post-QA5 boundary files plus the rolling
manifest in a separate child of that accepted archive. Module bytes are frozen
at `b87a0c3d7a7ca2d9f6751a99179897d8d8aca56d`; runner/observer bytes come from the
separately published operator commit. No model/Adam copy, model reload, training
edit, Slurm submission, or healthy-controller restart is performed. The optional
observer `--verify-files` reads all eight small destination files and their SHA.
Existing launch or operation claims must be inspected, never replayed. A successful
supplement preserves bytes; it does not certify an actual CUDA restore. The final
outer200 native-four-rank recovery remains a separate required acceptance.

Normal fixed512 evaluation precedes its checkpoint save; the extra QA5 callback
and the grounding refits at50/100/150 follow the save. A core snapshot taken
before a refit can therefore contain the previous grounding heads. The eight
post-callback files preserve the new manifest and RNG, but that alone does not
preserve the new head/report/dump bytes referenced by that manifest.

`post_callback_dependencies.py` independently closes those references in a new
`post-callback-dependencies` child: both sklearn heads, calibration report,
version manifest, all `dump_sha256` members and exact encoder mappings. Files
already covered by the accepted core reuse its SHA ACK and current physical
identity; default inspection does not read model tensor bodies again. New
transferred dependencies receive fresh source and destination SHA checks. A
restored round without local dumps can follow one authenticated source receipt
to its original manifest/dumps; no filesystem-wide search is performed.

`archive_post_callback_future_local.py` launches an independent CPU companion.
It qualifies this new dependency closure on CP5 without replaying the existing
eight-file writer, then handles50/100/150/200 after each actual core ACK and
post-callback boundary. Every writer has an exclusive claim. The existing core
controller and training remain unchanged. A specific dead core-controller birth
is a dependency failure; an observation timeout remains unknown. At200 the
reward round is150 and the RNG boundary precedes final51713. None of these byte
archives is an actual CUDA restore or a completion claim for the200-outer goal.

The first future companion, PID3844281, actually exited after transferring
50,580,970 dependency bytes: HK Python3.8 lacks `Path.is_relative_to`. Its
source snapshot, received stage, claims, failure and logs remain preserved.
`recover_post_callback_future_local.py --mode recover` is the explicit one-shot
compatibility recovery: authenticate that original terminal birth and exact
failure SHA, freshly verify and accept the existing stage, then run a clean CPU
companion under `future-50-200/recovery-python38-r1`. It does not replay the
source snapshot/emit/receive or the already accepted eight-file boundary.
The new companion inspects and reuses the accepted CP5 dependency child before
waiting for50. The dedicated recovery observer follows its own actual birth.
Never replay either launcher after an uncertain response; inspect its claims.
At the time of publishing this repair, actual recovery execution is still
pending. Training is healthy: outer7 is complete with20 writes per rank, CP7 is
sealed and outer8 has a real saved512-question chunk. No training, retrieval or
healthy core-controller restart is part of this CPU repair.

That Python3.8 recovery actually accepted CP5's10 dependency references, including
the6 transferred files/50,580,970 bytes. It then exited in a reader that compared
the dependency receipt's nested `binding.core` as though it were the boundary
receipt's flat `binding`. The data is already accepted; its original partial no
longer exists. `resume_post_callback_future_local.py --mode resume` continues
only from that accepted archive under `future-50-200/recovery-accepted-r2`.
It authenticates the exact r1 failure and terminal birth, and the normal
companion freshly inspects the accepted CP5 child using the corrected nested
binding. No accept, receive, emission or snapshot is replayed. Both failed CPU
roots remain immutable evidence. A fixture now exercises the real complete
dependency archive lifecycle and its actual receipt schema through the reader.

Actual r2 execution is now qualified: PID3881747/start1038573622 onxielab3,
operator c3ce9a5f9f148a87308e5ef0c27ed2457003c19b, completed the CP5 fresh
transferred-dependency SHA readback and accepted-core identity/ACK checks, reused
the10-reference/6-transfer child without writer replay and entered
WAITING_CORE_ACTUAL_ACK for50. Encoder payload bodies were not rehashed.
Core controller3603385 and both GPU jobs remain unchanged. The first readback
is in `evidence/actual-cp5-dependency-closure-and-future-wait50-20261021.json`;
the50/100/150/200 transfers and final CUDA recovery are still pending.

The next actual completed boundary is CP8: all four ranks wrote20 updates
(cumulative160), advanced the scheduler once to8, saved the23 checkpoint and7
boundary files, and published manifest52f85a226a3e2e9acddd61738dfa237e2f056466944f32e5bcb7b11053d9f816.
Outer9 subsequently saved512 candidate questions. Outer8 candidate EM was
37.3125%, selected-trajectory EM50.1171875%, reward0.61327267; all512 selected
groups were mixed, with no fallback or candidate-pool retry. These changing
pre-update training pools are not fixed512 validation measurements. No clear
collapse or training restart was observed; the200-outer goal remains active.

`watch_training_progress_local.py` is a bounded read-only local observer. A
candidate directory can exist before its progress.json: unknown step values
are retained as unknown until producer progress arrives. An earlier ad-hoc
observer stop predicate exited on that transient None value; its evidence is
preserved, and no training process exited or restarted for that reader error.
The published observer actually observed CP8 plus saved outer9 and exited0.

The user's 2026-10-21 23:23:22 CST instruction conditionally authorizes an FA3
transition after a complete checkpoint9. CP9 was sealed, and existing FA3
3.0.0b1 binaries were found in the old TOPD Python3.10 environment. They require
a process-private CUDA12.6 runtime preload; the shared Python3.12 environment
named flash_attn3 does not itself contain the package. No production process or
shared environment was changed. An independent H100 kernel diagnostic745988
completed forward/backward and isolation tests. Its random28Q/4KV inputs do not
establish full7B policy equivalence.

The independent four-H100 full-model diagnostic746013 subsequently completed
its numerical comparisons and exited1 because acceptance failed. On the same
CP9 model and32 original trajectories, FA2 repeats had exact forward log-prob
and loss equality, with1.08–1.11% backward gradient relativeL2 noise. FA3 had
log-probability max error0.528578/RMS0.041631, aggregate gradient relativeL2
25.02198% and cosine0.968835. These are not accepted small-error equivalence;
thresholds were not relaxed. All parameter bytes and caller RNG were unchanged,
and no optimizer or scheduler update occurred. This is an isolated diagnostic
failure, not a production-training failure. Evidence is in
`evidence/actual-fa3-full-model-numerical-rejection-20261021.json`.

The real HF path expands KV to28 heads before attention, unlike the preceding
random GQA kernel test. `submit_fa3_model_probe_hpc.py` therefore supports a
separate `layer-shadow` mode with its own fixed base and exclusive submit claim.
It runs one original FA2 forward, shadows the exact packed QKV with FA3 at each
layer, and returns the FA2 result to the next layer. This diagnostic isolates
local kernel error from propagation and records sampled FP32 references. It
does not approve FA3, override the full-model rejection, run backward, or switch
production. The original diagnostic root and its failure remain immutable.

Meanwhile production745371 completed outer10 on all four ranks, wrote20 updates
per rank (cumulative200), advanced the scheduler to10 and sealed CP10 manifest
1070a87785277207d86ae09f3470938829e30335ee0f2ccfb5d3964abc2dd751.
Reward was0.63938820; outer11 began afterwards. Production remains on its
immutable e082 runtime using FA2. The200-outer goal and final acceptance remain
pending. See `evidence/actual10-complete-cp10-outer11-started-20261021.json`.

The isolated layer-shadow diagnostic746054 completed0:0 in5m02 on ACD1-12,
using immutable source2ec9272fad4a29faf9ec06989c58d054b3030aca and actual CP10.
All112 rank/layer comparisons retained the original FA2 output. Repeated FA2
outputs were exact; FA3 same-input output relativeL2 median was0.000504737 and
maximum0.000737310. All expanded28 KV heads were exact repetitions of the
original4 heads. Both kernels had comparable sampled FP32 reference errors.
No large local expanded-KV error was observed on these FA2 trajectories. Small
local differences are compatible with multilayer BF16 propagation; this does
not establish the cause of the CP9 full-model mismatch on its different batch.
All parameter bytes/RNG remained unchanged and no backward/update occurred.
Postprocess weakref deallocation warnings were preserved; they did not change
the completed0 exit or the parameter/RNG readback. The evidence is
`evidence/actual-fa3-layer-shadow-cp10-20261022.json`.
This batch's longest individual sequence was2343, and the largest local output
absolute difference was0.125. Aggregate relative norms do not bound each token;
the FP32 reference samples first/middle/last queries, not every query. No
backward, full4096 batch, or subsequent FA3-propagated inputs were tested here.

The4096 synthetic attention-only first-call forward/backward timings were
FA2-first13.1184ms/FA2-repeat13.1057ms/FA3-first7.6400ms: approximately1.715x
for that kernel case. They are not warmed repeated model benchmarks, do not
cover the actual expanded28KV performance path, and do not measure a complete
training outer including rollout and retrieval. No whole-iteration FA3 speedup
or final-accuracy equivalence is claimed. Production continues with FA2; the
full-model numerical rejection remains in force and no checkpoint transition
was performed.

Production subsequently completed outer11: all four ranks wrote20 updates
(cumulative220) and advanced the scheduler once. Its23 checkpoint and7 boundary
files were sealed under manifest
3a5abfb7ea71aeb69251041055f9c42261996340ed918447cdc7e9c77f2b95fd.
Outer12 then actually saved its first512 candidates. Outer11 reward was
0.63003677, candidate strict EM39.890625%, selected-trajectory EM51.3671875%;
all512 selected groups were mixed, with zero fallback/retry. These are varying
pre-update training pools; fixed QA50/100/150/200 and final51713 remain pending.
See `evidence/actual11-complete-cp11-and-next-outer-20261022.json`.

The actual outer10 sealed-boundary wall time was1663.60s (27.73min), including
candidate fill875.713s and actor update257.928s. Logged reward/advantage took
154.514s and checkpoint save162.277s; its logged step716.812s excludes candidate
fill. Actor update accounts for approximately15.5% of that complete wall time,
and an attention-kernel speed ratio cannot be applied to the entire outer.
The synchronous frozen grounding service uses logical GPU3 and SDPA, with
micro4 for2560 commit plus2560 boundary texts before actor BEGIN. Micro8/16 is
only a reviewed future hypothesis: grouping changes left padding and default
positions as well as BF16/SDPA shapes. No batch change or new benchmark was
started. The measured timing and review scope are preserved in
`evidence/actual-outer10-timing-and-grounding-phase-20261022.json`.

Both existing archive-controller births were freshly verified live, waiting
for50 without failures or restart. The final native4 entrypoint coverage audit
is in `evidence/native4-final-acceptance-coverage-audit-20261022.json`: five
core modules remain byte-identical to productione082. Final overall acceptance
must additionally inspect all post-callback boundary/dependency archive
children and independently prove20 writes per outer/4000 per rank; a legacy
CPU consumer PASS alone does not prove those broader requirements. These are
final evidence checks, not new training-startup gates. No update201 or final
recovery is being executed at outer11; the full200-outer goal remains active.

`reproduction.easy2560_joint4.strict_final_write_audit` is an independent
final-only CPU reader for the strict retained20 budget. The existing
authenticated no201 reader already proves exactly200 successful outers, each
with1..20 retained writes, zero retained writes for failed attempts, and each
rank's complete actual write-event sequence. Requiring cumulative4000 therefore
forces20 retained writes in every completed outer on each of the four ranks,
including the original CP1 history. No scheduler or production consumer was
changed, and this reader is not connected to startup or training.

Only after actual200, run from a published immutable operator source containing
the new module, with the original training commit explicitly bound:

```bash
python -B -m reproduction.easy2560_joint4.strict_final_write_audit \
  --base /data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021 \
  --training-commit e082a23b5a65b302e2016aa0e5ab43b10b543d4f
```

It prints one JSON result and writes no files; it does not load Torch/models,
allocate GPUs, submit jobs, or claim an actual CUDA restore or whole-goal
completion. It also does not prove zero extra historical Adam calls or rollback.
Ten CPU fixtures passed, including actual200-row/per-rank ledgers,3999 rejection,
missing rank-write rejection and zero-write failed-attempt handling. Validation
is recorded in `evidence/strict-final-write-reader-validation-20261022.json`.
The reader has not been run against the still-incomplete production chain.

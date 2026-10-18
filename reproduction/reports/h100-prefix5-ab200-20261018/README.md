# Prefix5 A/B on h100-11: actual progress

The period20 snapshot records both independent four-card native runs at
**20/200 complete outers**, preparing21. Later progress is recorded in the
chronological sections below and the task's current state. The per-outer tables
are historical measurements, not a live counter; see `period10-paired-results.md`
for the first fresh paired diagnostic. The user authorized moving from h100-8 to the node with
most actual VRAM headroom, and reducing local micro-batch with accumulation.
Current deployed runtime is `2e3976230ac9425d66d8553afa9962821e964932`.
Later report/operation commits do not modify that runtime.

| Actual outer2 | A: frozen warmup | B: balanced online + EMA |
|---|---:|---:|
| Actual actor Adam per each of4 ranks, cumulative |40|40|
| Actual selection-probe Adam per rank, cumulative |0|20|
| Actual online EMA updates, cumulative |0|20|
| Selected actual mixed groups |392/512|392/512|
| Selected actual mixed fraction |76.5625%|76.5625%|
| Original post-update selected-train EMA/frozen balanced accuracy |50%|50%|
| Total wall time, including storage |974.194s|976.123s|
| Actor process: loading/rewards/PPO/checkpoint export |661.946s|661.563s|
| Retriever loading |130.084s|97.809s|
| Prefix phase |76.119s|69.340s|
| Prefix scoring |22.124s|32.667s|
| Same-prefix continuation |73.010s|81.882s|
| Probe metrics/update process |3.535s|28.336s|

These are one real fresh iteration's measured times on shared H100s, not a
stable average or an isolated PPO-kernel timing. Outer1 reused authentic saved
sampling after a user-authorized node move, so it is excluded from normal
iteration timing. No OOM occurred in the new node execution through outer2.
Local actor micro1/globalmicro4/accumulation32 preserves globalmini128 and the
original64 micro2 token-mean objective. All11 recorded actor/reward/rollout
scientific core source files remain byte-identical to the prepared7679 baseline.

B outer2 resampled only its own original512 groups:120nonmixed/392mixed became
256/256,297unique questions,41.9922% repeated draws. All four original probe
journals contain10 new Adam writes and the EMA clock advances10 to20. Learning
rate5e-6, EMA rate.99, trajectory BCE +5*group BCE remain unchanged.

On the **same original, unresampled selected512 groups**, B raw balanced
accuracy increased51.8707% to58.4269%, original-distribution total loss decreased
5.05501 to3.64579. EMA balanced accuracy stayed50%, total loss3.98274 to3.93105.
The raw ordinary accuracy74.4141% is below the76.5625% majority-class baseline.
A and B EMA/frozen predicted every selected group mixed at threshold.5. These
are selected training-data measurements, not heldout or QA results, and do not
establish a treatment benefit. Per-step balanced minibatch losses are retained
in the small curve JSON and are not a fixed-dataset loss curve.

Native actor and probe journals were independently read as small files and
checked for sequential actual writes by the observer. Model/checkpoint hashes
in producer receipts are not claimed as an independent large-tensor readback.
Normal milestone5 checkpoints, QA5, all later milestones, both200 outers, final
51713 QA and final state/resource acceptance remain pending. Both processes
continue; no outer201 is authorized.

Small actual evidence: `prefix5-ab-h10011-first-fresh-outer2-actual-facts.json`
and `prefix5-ab-h10011-small-curves-latest.json`. All tensors, optimizer states,
rollouts, full predictions and source inputs remain remote.

## Completed outer3

Both same native processes completed3/200 and continue outer4. Actor60 actual
Adam/rank in both arms; frozenA remains0, B raw/EMA30. Selected actual mixed:
A383/512=74.8047%, B392/512=76.5625%. No QA5 result yet. Full new outer3 walls:
A: 784.1525464951992 seconds; B: 814.8239887040108 seconds; 

These variable shared-node times do not isolate algorithm speed. Original
sequential rank-journal facts and per-step probe curves are retained in the
small JSON artifacts. Runtime remains2e397623 unchanged. Sole readonly observer
follows the normal milestone5/QA5; full goal remains incomplete.

## Actual milestone5 and QA5

Both same native h100-11 processes completed5/200 and continue6. All4 actor
ranks each100 actual Adam; A selection probe0/frozen; B actual50 Adam and50 EMA
updates. No training restart or OOM occurred in this h100-11 execution.

| Fixed512, identical new BF16 greedy protocol | A frozen | B balanced+EMA |
|---|---:|---:|
| Strict EM |141/512 =27.5391%|146/512 =28.5156%|
| F1 |36.1807%|37.3150%|
| NQ EM |68/256|68/256|
| Hotpot EM |73/256|78/256|
| Actual worker exits |4 workers, all0|4 workers, all0|

Both use the exact3d631836… fixed question source. These are early same-protocol
QA results, not evidence of a statistically established algorithm benefit.
The normal milestone5 independent full-file SHA plus actual CPU deserialization
passed for A25files/38605953737B and B25files/39537796626B. These proofs verify
FP32 actor/Adam/scheduler/RNG/driver/cursor and the correct frozen or raw+EMA
head state; they are not an independent backup or extra GPU-forward test. Both
CPU workers are actually gone; do not repeat completed hashes or loads.

Selected outer5 actual mixed: A408/512=79.6875%; B380/512=74.21875%. B balanced
only that original current512G5 to256mixed+256nonmixed,298unique questions.
On the original unresampled same data, B raw paired BA45.18%→53.70%, total
loss5.07047→4.54421. EMA BA remains50%; its total loss4.08221→4.09179 worsens
slightly. Do not claim all losses declined or call these heldout metrics.
The first fresh paired probe diagnostic is periodic10, not QA5.

Next natural milestone10 will use original2560candidate G5/full-pool selection
and fitting rules; B fits50 updates once, with balanced1280/1280 groups. The
sole readonly observer follows6..10 plus actual scoring11 with head10. Both
full200 budgets, later QA50/100/150/200, final51713, retention and final native
exit0/owned cleanup remain incomplete.

### Retaining fixed inputs without modifying either live run

The current private namespace still reads the reference asset bundle and corpus
cache from h100-8. `operations/h100_reference_replica/retain_fixed_assets.py`
retains those read-only files on NAS18, at 20MiB/s with 128GiB ordinary free-space
reserve. New payloads receive source and independent destination full SHA reads.
The existing immutable, independently verified FAISS/corpus backup is reused by
hardlink, with its original complete proof explicitly recorded; its large hashes
are not repeated. The explicit controller reconciles an existing intent/receipt
before launching a worker. Copy completion does not itself change a live mount,
UUID lock lease, launcher, scientific configuration, or deployed runtime. A future
source-independent bootstrap and any necessary complete-boundary handoff remain
separate work; do not claim the live jobs are already independent of h100-8.

The first retention worker finished all 19 reference bundle files and linked the
2 already-sealed retrieval payloads, then exited before copying the cache because
a mutable datasets `.lock` mtime changed. No training was stopped or changed.
The cache retry excludes process lock files and adopts the completed small
manifest without re-copying/re-hashing those 21 entries. It uses a distinct
`cache2-` intent/receipt/progress namespace and requires the original worker gone.
Only immutable cache payloads are in this retry.

### First six complete outers: selected original groups

| Outer | A mixed /512 | B mixed /512 | A wall seconds | B wall seconds |
|---|---:|---:|---:|---:|
| 1 | 396 (77.34%) | 402 (78.52%) | 589.55 | 612.04 |
| 2 | 392 (76.56%) | 392 (76.56%) | 974.19 | 976.12 |
| 3 | 383 (74.80%) | 392 (76.56%) | 784.15 | 814.82 |
| 4 | 394 (76.95%) | 409 (79.88%) | 748.52 | 738.39 |
| 5 | 408 (79.69%) | 380 (74.22%) | 899.09 | 898.21 |
| 6 | 404 (78.91%) | 361 (70.51%) | 720.92 | 718.56 |

Outer1 reuses prior exact sampling and is excluded from fresh sampling timing.
Outer5 includes QA; outer6 uses the larger ordinary candidate pool. Shared GPU
load varies, so these are actual whole-outer observations, not a controlled kernel
benchmark. Original selected groups are not a heldout diagnostic.

In B outer6, raw mixed balanced accuracy rises 50.02%→57.94%, but raw
trajectory BCE worsens 1.1310→1.3638 and weighted total worsens 4.4812→4.6690.
EMA mixed BA stays 50% (all selected groups predicted mixed at threshold0.5);
EMA total falls 4.7733→4.7567. These mixed signals are recorded without changing
the requested loss, LR, resampling, or EMA protocol. Periodic10 fresh heldout
paired diagnostics remain pending.

### Both seven complete: same native runs continue

Both arms complete7/200; all four actor ranks each140 Adam writes and scheduler7.
A remains frozen with0 online probe updates. B has70 actual probe/EMA updates.
Outer7 wall seconds including persistence: A787.6631, B785.6887.
Original selected mixed groups: A405/512 (79.1016%), B332/512 (64.8438%).
Actual outer8 pipeline has started. Fresh paired probe heldout diagnosis is at10;
there is no scheduled QA10. Same runtime2e397623, no deployment or restart.
The complete small original-group paired metrics and all four rank journals for
each finished outer are in `through7-small-actual-comparison.json`.

### Recovery path preparation independent of h100-8 storage

`h10011_retained_paths.sh` creates a **new private** mount namespace and mounts
`h100-10:/mnt/data` directly, without traversing h100-8's NFS ancestors. It then
maps the completed retained input mirror and the retained common8 runtime to the
same canonical paths. It requires the actual `cache2-RESULT.json` completion,
not a launch receipt. `inspect_retained_bootstrap.py stage` only stores this tiny
wrapper on h100-11; `inspect` performs a CPU metadata-only path/mount inspection
once, after copying completes. No model forward or large rehash is involved.

Staging or inspecting this wrapper does **not** activate it in either live run,
change their scientific configuration, or transfer their UUID leases. Any later
training handoff must preserve the actual latest complete joint state and release
old owned processes/locks first. Current live jobs still use the original private
namespace; do not claim they are already independent of h100-8. Inspection is
recovery readiness work, never a gate that stops healthy training.

The retained-path CPU inspection also reads the complete small copy result and
checks every final payload size and source/target digest equality recorded by the
copy worker. Existing sealed hardlinks must still share an inode with their
retained backup payloads. Paths are rebased onto the new direct NAS18 mount, so
inspection does not need h100-8's ancestor directories. This is metadata
acceptance of completed full-SHA copying, not a repeated large payload hash.

Static recovery routing inspection is saved in `restore-entry-static-support.json`:
the deployed `i1_i6_joint_v1` full-joint adapter delegates A/B receipts and probe
state validation to the dedicated A/B implementation. Frozen A has no online
optimizer/EMA; B preserves raw/EMA/Adam/4-rank RNG and balancing state. The legacy
`resume.py` and `native_a800_joint_v1` fixed-clock adapters are not the A/B entry.
This identifies the existing supported code path; no recovery was executed.

### Eight complete and fixed input retention complete

Both native arms complete8/200, actor160 per each of4ranks, A probe0,
B probe/EMA80. Original selected mixed fractions: A368/512=71.875%,
B336/512=65.625%. Whole outer8 including persistence: A806.226s,
B806.789s (about13min26s). Outer9 has really reserved new candidates.
No QA8 or heldout-period10 result is claimed. Actual four-rank write journals
and original-group pre/post raw/EMA metrics are in the through8 small comparison.

Fixed input retention completes with53 payloads:51 freshly copied files
28,545,953,373B with source and independent target fullSHA, plus2 hardlinks
to the existing immutable fullSHA index/corpus backup. Total retained payload
bytes107,498,608,266B. Cache retry RESULT SHA
`ac175bd5800380dde729603bb1032b6bef390c6832cdac3bdc5221db5e1bf354`.
The worker209722/start705287691 is actually gone. Do not repeat completed copying
or large hashes. The final CPU path inspection validated all53 payload rows and
36 canonical A/B path roles using direct NAS18 mounts; proofSHA
`119f5518d7df8578ed97ed48df07a2999ee1823430e72837739f1b538ab019eb`.
The inspection performs neither model forward nor new large hashes.

This is fixed-input retention and recovery-path readiness. Current live training
namespaces/UUID leases are unchanged and still retain their source8 dependency;
no training handoff, restart, or scientific change was performed. Full200 per arm,
QA50/100/150/200, final51713, final state and resource closure remain incomplete.

Periodic rollout progress has a distinct output directory: `full-pool`, plus the
fresh fixed-question diagnostic under `diagnostic/outers/outer-N`. A null ordinary
`full` progress field during that phase is not evidence of a stalled rollout.
`read_periodic_phases_once.py --outer N` reads only those producer progress JSONs
and small completion/metric headers, with a fresh local evidence tag. It does not
load tensors, start a second observer, mutate training, or repeat generation.
The original single observer still verifies complete joints and actual score11.

### Both ten complete: first fresh paired periodic result

See [period10 paired results](period10-paired-results.md) and the small JSONs for all4-rank actual clocks, original-group metrics,50 globally reduced loss updates, sampling multiplicities summary, and full wall times. Both actor200/rank, Aprobe0, Bprobe/EMA140. Same runtime2e397623; no restart or scientific change. Actual outer11 score adoption is checked separately by the sole existing observer. Full200 and final QA obligations remain incomplete.

## Actual outer11 selection and continued normal follow-up

`score11-ema-adoption.json` records the actual selection receipts: frozen A
uses its unchanged warmup head with zero online updates; B uses EMA140 from
its completed outer10 checkpoint, not the raw trainable head. The head hashes
match the actual preceding joint receipts. These metadata fields are in
`selection.json`, not `manifest.json`; the previous monitor queried the latter
and printed null. This was an observation coverage issue, not a training change.
The frozen scoring code explicitly loads `ema_model` for balanced B. No tensor
readback or new forward pass was performed for this metadata reconciliation.

The old through10 observer exited naturally. The replacement read-only
`follow_h10011_ab_normal_boundary` follows the same native leaders at five-minute
intervals, validates only newly completed original rank journals, and stops for
root attention when both normal50 savepoints exist or an actual native handle
is missing/terminal. QA50, grounding50 adoption, normal checkpoint acceptance
and all later requirements remain separate obligations; observing50 is not
goal completion. No duplicate trainer, new GPU gate, or runtime deployment is
introduced. The current run still uses its existing source8-dependent mount;
the separately verified retained-input bootstrap is prepared for a necessary
future full-boundary handoff, not hot-swapped into a live process.

## Through13 small results

See [through13-results.md](through13-results.md) and
[through13-small-metrics-and-loss-curves.json](through13-small-metrics-and-loss-curves.json).
The new parameterized `read_ab_small_report_once --through N` replaces ad-hoc
loop edits in the old two-outer reporting snippet. It reads only completed
metadata and global loss rows, reports missing future rows as pending, and
never changes the runtime or creates a second watcher. Both200-outer budgets
and all scheduled QA/final-state obligations remain incomplete.

## Completed period20

See [period20-paired-results.md](period20-paired-results.md),
[period20-paired-metrics.json](period20-paired-metrics.json), and
[through20-small-metrics-and-loss-curves.json](through20-small-metrics-and-loss-curves.json).
Both complete20, actor400/rank; frozen A0, B280 actual Adam/EMA/rank.
B EMA full-pool BA66.96%→68.15%, fresh diagnostic BA53.18%→53.69%;
diagnostic mixed BCE worsens1.04549→1.05375. No QA20 or benefit claim.
The same native processes continue toward200; the normal50 observer remains
active and the frozen runtime is unchanged.

## Actual25 and a scoped observation interruption

Both same h100-11 native leaders completed25/200, actor500/rank; A frozen0,
B raw/EMA330. Original21–25 actor and probe journals are retained in the local
small evidence bundle. The next26 was already updating when the original
observer read back again at02:24:31UTC. No trainer or observer was restarted.

During02:14/02:19 reads, the private management-to-retained18 SSH hop timed
out. Ordinary `ssh mgmt-host hostname` succeeded. Known5 could reach18 SSH
and NFS and source8 NFS, while5/8 TCP22 were reachable from management.
An offline Tailscale peer record for8 did not prove compute8 was down.
Slurm reported11 ALLOCATED; that alone was not used as training proof.
The original18 route subsequently read the same native leader births,
complete25 and actual26 writes. The network root cause was not isolated;
no global network/auth/SSH or training configuration changed. See
[private-hop-observation-recovery-20261018.json](private-hop-observation-recovery-20261018.json).
The route remains the original retained18 path and all200/QA obligations continue.

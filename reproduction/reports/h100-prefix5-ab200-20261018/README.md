# Prefix5 A/B on h100-11: actual progress

Both independent four-card native runs have completed **2/200 outers** and
continued to outer3. The user authorized moving from h100-8 to the node with
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

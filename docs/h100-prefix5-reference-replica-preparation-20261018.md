# Reference Prefix5/P10 replica preparation — 2026-10-18

Human source: `codex://threads/01a1488e-807d-7ca3-b024-96b8fe97486c`.
This is preparation only. The human will assign the two experiments later.
No actor/probe training, warmup, QA, baseline or GPU acceptance is launched.

## Pinned implementation and initialization

- Reference scientific runtime: `75d536575593891876df339467da73b3c4cbdc7c`.
- Preparation starts from `d3a03054009fe3bf0068dfefabca941df95df101`, the
  reference session's published continuation prefix-cache repair. Its sampling
  constructor disables prefix caching; actor/probe math is unchanged.
- Official actor and reference: Qwen/Qwen2.5-3B-Instruct revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Fresh actor/Adam/scheduler/RNG/pool.
- Selection probe: the **same retained warmup epoch5 EMA checkpoint**, SHA256
  `b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594`,
  1,242,424,352 bytes, source run `8af926a49630428097b94027864da68b`.
  Model weights only are imported. Do not import its optimizer/RNG/EMA clock.
  Online probe Adam starts at zero. No new warmup and no DIST-ablation weights.
  Transport may strip the unused raw model/Adam/RNG from the original container;
  the source extracts the EMA with the reference loader and proves every learned
  tensor and native buffer byte-identical. H100 independently loads that export
  and compares all tensor SHAs. Original checkpoint SHA and transported-container
  SHA are separate; a different serialized file is not described as byte-equal.
- Two independent four-rank slots: A uses physical0–3, B uses4–7. Their dataset
  and warmup source are shared read-only; state, caches, ports, processes and
  output namespaces are distinct. Neither slot has a submit receipt.

## Exact data identities

| Input | Count | SHA256 |
|---|---:|---|
| Common manifest | original split | `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24` |
| Train parquet |169028| `5d40180ba54d35513bb780408caa068d7d60123214a49b0739f835a2ae47c8a1` |
| QA dev parquet |512, NQ/Hotpot256 each| `3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb` |
| Final parquet |51713, original seven sources| `55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1` |
| Grounding calibration |same ordered300| `fa7089e21bfd56326a4fd10874362b5fda1a1b70f969a14e4bcbb581daad4be1` |

Copy the reference's original diagnostic-excluded source question pool and
fixed512 identities byte-for-byte. Do not reconstruct from the previous
20480/5120/15360 ablation split. Preserve seed1729, candidate permutation and
cross-cycle no-duplicate policy. Both replicas start with fresh pool cursors.
Retrieval uses the same Wiki2018, full E5 Flat index, e5-base-v2, FP16 GPU
cloning/query settings, mean pooling, `query: ` prefix and top3.

## Selection and online probe

1. Generate actual G5 Prefix5 with the current actor. Capture native layer27
   postblock including residual, actual question+5 positions/masks. Sample6
   solely to process token5; discard sample6 from PPO/tools/continuation.
2. Same original Qwen block35 + masked flatten over278×2048 + Linear1.
   `p=sigmoid(z)`, `h=1-prod(p)-prod(1-p)`. Same stable32-pattern implementation.
   Exploration0; descending h, original candidate ordinal breaks ties.
3. Preserve the reference **segmented candidate schedule**: outer1–5 use the
   retained original2560 setting; from6 ordinary rounds use5120 candidates.
   Do not silently change `effective_from_outer=6` to1. Updates10,20,…190 use
   2560 candidates, complete all12800 trajectories, select actual strict-EM
   mixed groups first (`0<K<5`, includingK1); probe scores rank within classes
   and fill nonmixed rows if fewer than512 mixed groups. Outer200 is ordinary.
4. Continue the **same** five scored prefixes. Actor always uses512×5=2560
   complete trajectories. No regeneration after selection.
5. Probe updates only10…190: all2560G5, one natural epoch, LR5e-6, average
   globalbatch256, wholeG5 same rank/step. Same40 batches255 +10 batches260,
   50 synchronized Adam updates/rank/period, final950. No repeated sampling.
6. Loss is the reference **trajectory BCE + G5 mixed BCE, weights1:1**:
   `mean_ij BCEWithLogits(z_ij,c_ij) + mean_i BCE(h_i,1[0<K_i<5])`.
   No DIST, mixup, soft0.9 target, intensityKL50, EMA online update, or extra
   objective from the superseded probe experiments is carried over. EMA is
   used only to initialize from the retained warmup checkpoint.
7. Diagnostics use the original fixed512 questions with **fresh same-actor G5**
   at each probe period; compare pre/post on the identical current rollout.
   They do not reuse historical teacher trajectories. Ordinary uncontinued
   candidates have no labels and cannot contribute a claimed pool accuracy.

## Actor, reward and interaction

- Same four physical ranks emulate eight virtual ranks. Each mini128 is the
  mean of64 original micro2 masked-token means, not global sequence/token mean.
  Globalmicro16/local4/accum8, one PPO epoch,20 opportunities per outer.
- LR1e-6,57 **complete outer** linear warmup, first LR0, then constant.
  AdamW betas(.9,.999), weight_decay.01; FP32 master/Adam/loss/reduction,
  BF16 model forward, clip.2, DualClip3, gradientL2 clip1.
- Original Search-G1 Eco/soft-grounding reward and GRPO advantage computation.
  Selection correctness probe is **not** the reward probe. Add a virtual zero
  only to all-five-correct groups' reward normalization statistics, then
  discard it; no fabricated actor trajectory or changed raw reward. Other
  groups use the original sample mean/std and epsilon1e-6.
- Same old-policy KL.2, negative-advantage tail.01, exp finite bound30 and
  FP64 L2 clipping. Twenty processed opportunities plus at least one returned
  Adam write makes a complete outer; skips/partial/Adam are separately counted.
  Do not force every opportunity to write or invent rollback of written Adam.
- **Grounding round0** imports the same actual boundary/commit heads and
  calibration identities. **50/100/150** refit both reward heads on the saved
  current actor; all four services adopt before51/101/151. This lifecycle is
  separate from the Prefix5 selection probe's P10 updates.
- Shared new train/QA protocol: BF16, context4096, action500, observation500,
  answer reserve500, early-final, protected information wrapper, max4 search
  turns, actual native first-complete-action stop. Keep original token IDs,
  positions, masks, old-policy logprobs and strict EM/F1 scorer.
- vLLM: same maxseq1024/maxbatchedtokens32768/free-reserve2GiB/utilcap.95,
  eager execution and reference prefix-cache repair. No batch enlargement.
- QA at5/50/100/150/200 and final51713, exact canonical IDs/seven source counts.
  Preserve complete joint saves and only-owned cleanup. Do not inherit a
  current reference actor checkpoint when the human assigns a fresh replica.

## Allowed hardware/path differences only

The original A800 run has four training GPUs plus two auxiliary GPUs. Here each
four-card H100 slot colocates retrieval/reward services with training. Exact
FAISS shards change2→4 for memory placement; the full index, search precision,
topk, encoder and retrieval algorithm stay unchanged. Grounding service/refit
physical devices become3/7. Role placement, UUID locks, ports, absolute paths
and new execution identities necessarily differ. Hardware-dependent floating
point results are not promised bitwise equal.

Scientific files (actor, safety, probe, prefix model, rollout/native hidden,
candidate/periodic selection, pooling, interaction and evaluation) retain the
reference source bytes. Changes to existing files are limited to placement
admission, remote path admission, lock deduplication, physical grounding device
validation, and permitting4 FAISS shards. The complete config delta is written
remotely for each slot; no scientific override is hidden in a wrapper.

All weights, full inputs, original trajectories, question identities, index,
corpus and outputs remain remote. Remote-to-remote transfer may stage on the
management host; model/data bytes never transit the Mac. Source and destination
full SHA must match before asset readiness. Preserve source experiment/jobs.

## Handoff boundary

`prepare_h100_reference_replica` creates `config.template.json` only, never
`preparation/launch-config.json` or submit/launch intent. The human will specify
which two experiments to run. After preparation, an independent sub-agent
audits the complete reference diff, assets and configuration for omissions.
Any residual gap must be reported and fixed before claiming readiness.

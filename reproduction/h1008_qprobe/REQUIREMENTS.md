# Fresh H1008 question probe experiment

Source branch: `codex/qprobe-a800-async-top512-20261014`, independently resolved upstream HEAD `1f75a186eea7b769c37a919096ea0b54372e5160`.

Start only after the current CP200 final evaluation (job3395) finishes and releases its resources. Do not cancel it for preparation. Preserve all old runs, results, assets, checkpoints and source.

## Scientific contract

- Fresh actor and fixed reference: official Qwen/Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. No model/optimizer/scheduler/RNG/probe state from the prior trained experiment.
- Eight physical H100 GPUs. Preserve upstream global rollout budget512 unique questions x5 responses, global mini128 and one PPO epoch. Explicitly derive per-rank micro and accumulation from eight ranks; no silent global batch increase. Actual memory determines a feasible micro, not an assumed H100 capacity.
- Selection:64 randomly sampled unique candidates, then448 highest probe diversity scores from the remainder; preserve upstream candidate pool/identity/tie-breaking/scoring protocol. Change upstream exploration0 to64 explicitly in manifest and admission.
- Frozen official encoder, full final decoder-layer token sequence including original positions and valid-token lengths. The existing single pooled vector cache cannot supply contextual self-attention and must not be accepted for the new probe.
- Three independent trainable Qwen decoder blocks. Every block starts as a deep full copy of the official model's final decoder block: every parameter and persistent buffer equal, no shared storage with the source or between copies. Architecture, attention heads, GQA, RoPE, normalization, gated MLP and epsilon follow the official Qwen config. Source encoder parameters remain frozen and absent from the probe optimizer.
- Preserve upstream six-class correct-count target K=0..5, exact count loss and diversity moments. The actual upstream head is a small count MLP, not a vocabulary LM head. New Qwen block stack feeds a trainable six-class output projection. Do not introduce a token-language-model objective or use gold labels at selection time.
- Probe training and prediction must process sequences with causal/padding masks and original position IDs, then select last valid token. Packed question boundaries must never attend across questions. All feature caches, tensors, model/optimizer state and full outputs stay remote.
- Probe checkpoint must include architecture/source initialization provenance, block/head parameters, optimizer, private RNG, replay identities/features and state paired to the actor checkpoint. Incompatible old pooled-head checkpoints must be rejected.
- Continuous200 complete outer updates after initial acceptance, same process; save5/50/100/150/200. Count real Adam writes per rank; report quality and selection/precompute costs. No201.
- Final fixed51713 across seven datasets, actual qa_count, all saved point integrity/reload/persistent backup and exit0/resource release required before goal completion.

## Completion evidence still required

This document and isolated source checkout are preparation, not a running or accepted experiment. Pending: sequence cache, Qwen block integration with online fitting/teacher warmup/checkpoint, eight-GPU launch and allocation bindings, actual official tensor full-copy readback, targeted tests, source publication/remote fullSHA, fresh initialization, five complete outer acceptance, all200 and final evaluation/closure.

## Prepared online protocol

Sequence features now flow through selection and completed-outer fitting. Three blocks and the count projection use FP32 trainable parameters and Adam state; physical probe train micro4 accumulates to the upstream logical head batch256, prediction micro32 bounds candidate memory. Checkpoint schema2 binds Qwen configuration, actual attention implementation, initialization source, optimizer, private RNG, ragged replay and actor pairing. Legacy pooled heads remain schema1. Tiny actual Qwen CPU runtime/rollback/save/reload/next-update tests pass (27 tests, no GPU or official full-size claim). Actual full-size precompute, H1008 launch/deploy and all training acceptance remain pending.

## GPU worker transport (implemented; runtime validation pending)

The three-block head may use sequence_device=allocated_rank7. Its compute and optimizer live in the original ActorRolloutRefWorker at rank7 of world8, called directly through its existing Ray handle. No new actor, GPU reservation, broadcast of replay/state to all ranks, subclass WorkerDict, or driver cuda7 access. The driver keeps only metadata and CPU outputs/checkpoint staging. Initialization records actual CUDA device/UUID and full-copy provenance; wrong owner/world/device is rejected. Two local interface/registration checks pass. The complete new remote CPU RPC regression suite and actual GPU loading are pending because both the known direct management ingress and the existing SOCKS route currently time out. This is not evidence that evaluation3395 stopped; do not restart it.

The management ingress recovered UTC2026-10-14T17:16. Same evaluation3395 was authoritatively RUNNING with23040/51713 complete journal rows. The full CPU suite now passes37 tests, including remote-proxy/local-head update, state rollback, actual checkpoint readback and actor pairing. These use tiny real Qwen blocks with a synthetic GPU-owner transport; they prove protocol/serialization, NOT physical GPU loading or full-size memory admission. New training remains unsubmitted.

## Standard-deviation selection loss (2026-10-15)

Fresh probe loss is three-class CE over `(p0+p5,p1+p4,p2+p3)` plus
normalized standard-deviation MSE with weight 1 and six-count auxiliary CE
with weight 0.1. Mean MSE is disabled and rejected. The checkpoint config
records `loss_version=std_class_ce_v1`; prior mean-loss heads are incompatible.
Both CPU and physical-micro sequence training use the same per-question loss,
retaining the original logical-batch denominator and replay age weights.

Acceptance uses predictions captured before generation and before current fit:
Top448 versus random64 mixed hit-rate difference/ratio, actual mean sample std,
std MAE/MSE, and pairwise std concordance on selected questions. This does not
claim ranking quality over ungenerated candidates. RankNet is deferred.

A new isolated fresh actor/ref/probe run may explicitly bind `frozen_teacher_cache`
to the earlier official frozen teacher namespace. Its existing identity and
per-chunk raw SHA/frozen-worker checks remain mandatory. No fitted head, optimizer
or actor state is reused, and no concurrent teacher-cache writers are permitted.
All models, checkpoints, raw logs, trajectories and computation remain remote.

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

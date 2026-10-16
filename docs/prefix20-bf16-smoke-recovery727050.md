# Restore the bound BF16 smoke contract

Job727050 failed before actor/probe training, after all four actual CUDA/JIT/FSDP
preflight ranks passed. The actual same-actor internal/separate recomputation
gate passed0.05, but an omitted profile branch applied the legacy FP32
cross-backend gate to the explicitly bound BF16 sampling-journal execution.
Observed actor/reference maximum was0.7373237609863281; actor/vLLM maximum
was1.2512952089309692. These are measured differences, not numerical parity.

The successful sigma3 implementation already distinguishes this profile:
`model_smoke.compare/aggregate` records finite cross-backend drift with
`tolerance=null` under `sampling_journal_no_extra_hf_forward`. The original
same-actor0.05 gate, real token/mask/finite checks, DTensor weight synchronization,
actual precision observations and zero-optimizer-write requirement remain.
Legacy execution without the bound policy still requires cross-backend0.05.
Restore that branch and pass the same run binding into parent admission, using
the actual four ranks. Restore its monitor-scope text from topology as well.

This does not change actor/reference computation, PPO/rewards/advantages, native
sampling log-probability denominator, probe loss/selection, or any batch/LR.
In particular the existing BF16 actor/reference FP32-residual distinction remains
as specified in `precision_evidence.verify_precision`; no new precision adaptation
or increased numerical tolerance is introduced. Preserve per-process JIT caches.

`recover727050` checks fresh own terminal accounting, original run/source and
failure-file SHA, both stopped pretraining roles, empty training artifacts and
free own lock. It preserves the failed namespace and creates a new one once.
Before staging it measures only the three known own failed/fresh trees with
`du -s -B1`, adds the next real input/source copies and64MiB metadata margin,
and requires the cumulative amount to fit the existing extra2GiB allowance.
The global storage observation keeps its original timestamp and conservative
upper bound. Personal HPC3 budget remains2TB.

CPU fixtures verify receipt semantics/rejections and cumulative reserve math.
They are not actual CUDA or training acceptance. A fresh four-card job must
produce actual model-smoke evidence and complete actor/probe outer receipts.

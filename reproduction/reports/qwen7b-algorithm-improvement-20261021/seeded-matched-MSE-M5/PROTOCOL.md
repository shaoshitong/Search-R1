# Matched MSE-M5 genuine second training seed

Stage only a single leading L15/Prefix80 own-G5 population-variance MSE replica; no grid or Slurm submission. Parent source0436c2 remains immutable. Fresh official block/head/Adam, init seed19137 and explicit local epoch-plan seed2026102302; accepted real8192question set and fixedtrain/dev ownG5 remain the same. Question order and each question own-G5 change independently by epoch.

All original matched MSE math stays unchanged: sigmoid output regresses4*population EM variance, global128/local64/micro8/accum8, LR5e-6/EMA.9995, 10epochs640actual updates. Raw/EMA/Adam/two-rank RNG/counters/plan SHA are checkpointed with explicit seeds and initial head SHA. Entry fresh mode cannot silently resume.

Root freezes seed1 E7/raw before seed2 validation metrics. The two replications use that same frozen epoch/state; no best-seed selection or final data use. Small CPU checks exercise actual seed plumbing, real8192identity/order/view differences, repeatable changed head seed and fresh/no-resume. Staging reuses source token refs read-only and generates no rollout/hidden data.

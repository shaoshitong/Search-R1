# CURRENT64 genuine second training seed

Stage only; root owns Slurm submission. Fresh official block15 and 2048-position packed group head; input is question plus ownG5 Prefix80. Explicit initialization seed19137 and local epoch-plan seed2026102302 change head initialization, question order and each question ownG5 view; fixedtrain/dev ownG5 identities stay unchanged. No imported-global seed patch is used.

The immutable source237968e8 training math is retained: global64/local32/micro8/accum4, current-only capacity0, K16, exact oracle precision-at-K ramp margin1, LR5e-6, EMA.9995, 8192questions once per epoch, 10epochs1280 batch attempts. Zero ramp/current gradient skips Adam and EMA; actual counters stay separate. Raw/EMA/Adam, two-rank RNG, queue/version/counters and plan SHA remain checkpointed; explicit init/train/diagnostic seeds and initial head SHA are added to controller and checkpoint. Entry requests fresh mode; it cannot silently resume.

Small CPU tests verify actual epoch-plan function seed reaches 8192 identities, order and ownG5 changes, repeatable changed head seed and fresh/no-resume guard. Staging runs the same tests against accepted real8192 metadata, without loading actor or checkpoint tensors, generating rollouts, caching hidden states or reading sealed final data. All existing jobs, sources, fixtures and plans stay read-only.

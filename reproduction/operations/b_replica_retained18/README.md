# Independent B replica on retained H10018

Human 2026-10-18 explicitly stops old native3433 and requests experiment B from thread01a0fcbb-863f-7450-926d-db537f648265, with an independent code directory. Original A/B jobs remain untouched.

Scientific runtime copied from published2e3976230ac9425d66d8553afa9962821e964932. B is current-outer whole-G5 class-balanced raw probe Adam5e-6, mixed-event loss weight5, EMA0.99 for selection. Actor starts official revisionaa8e72537993ba99e69dfaafa59ed015b17504d1; selection heads use the same qualified epoch5 EMA model-only initialization. No old actor/online optimizer or B training checkpoint is imported.200 outer, original candidate schedule, four ranks4–7, micro1/accum32 and original20 actor opportunities are preserved. No repeated baseline/warmup/short-run restart.

Only identity, ports and execution node change. A task-private mount namespace exposes already retained assets on H10018 under their qualified paths. No weights/data/index/environment travel via Mac. Run output/code, cache, ports, UUID leases and optimizer/RNG are independent. All paths remain remote; source copies are ordinary independent files, not mutable hardlinks to the original B checkout.

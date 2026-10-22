# Terminal-only packed checkpoint CPU acceptance

Independent verifier for online-packed-M3to10-P20-variance-8192-20261022-v1. Stage does not start validation or open a checkpoint. The verifier requires actual complete.json plus matching Slurm COMPLETED0:0 before reading final epoch3 checkpoint. It never alters Slurm, training or final4096.

Fresh full SHA is compared with immutable published latest.json. mmap loads one checkpoint; real PackedVarianceProbe with official block21 and PackedVarianceHead (positions1024x256, output1x256) is built on meta and strict-loaded raw/EMA by assignment. Finite tensors, separate raw/EMA hashes, full named Adam moments and192 clocks, bothrank Python/NumPy/CPU RNG load and serialized CUDA RNG are checked. This is CPU restore only, not actual GPU forward or next-update equivalence.

It recomputes each epoch8192 unique question/ownM3..10 label-blind deterministic plan from actual accepted G10 groups and verifies coverage, M counts, populationvariance targets and CP plan hash. It checks actual controller singlequestion/oneprefillrow/MSEonly/predvarselection fields, completed0..3epoch evaluation objective and score fields. No trajectory BCE, Mixed BCE, ranking loss or actual strictEM ranking is accepted.

Sources reuse the established bounded CPU mmap/finite/meta/Adam/RNG logic; only model geometry and packed plan/objective gates differ. Local static compilation passes. No actual packed checkpoint has been opened yet; runtime proof remains pending until terminal acceptance.

Root or delegated read-only worker may start after full training completes using deploy_packed_variance_CPU_acceptance.py --action start --commit PUBLISHED_COMMIT. Until then stage only.

## Actual terminal acceptance

Job749515 was independently observed COMPLETED0:0 and complete192/3epochs before opening finalCP. Final3773205524B SHA fbc7a7e4ea0dc739dd6168b04da55242ad78f566271f15a53fc91728ae3643a2 passed. Real raw and EMA each45tensors/235819009parameters strict-load; Adam45states with192steps, finite471638018moment elements and exact CPU optimizer restore; bothrank RNGs accepted and CUDA16B states recorded. Three8192unique ownM epoch plans, selectedEM populationvariance, actual coverage and prediction sorting fields passed. Actual checkpoint verification14.838sec, peakRSS4450304000B. Validator exited normally. GPU next-update equivalence remains untested. See accepted.json for complete source proofs.

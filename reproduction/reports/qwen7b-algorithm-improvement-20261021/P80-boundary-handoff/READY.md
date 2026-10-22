# P80 published-epoch recovery handoff

Source commit `74178e17320d2a3d44eca00766c3061330958bee` was pushed and its remote branch SHA verified. The independent code was staged on HPC3; CPU imports and shell syntax passed. No checkpoint was opened during staging and this deployment did not pause, cancel or submit a job.

The CPU validator accepts only layer15/layer18 Prefix80 epoch1/epoch2 immutable published checkpoints, with actual cursor64×epoch, fresh complete SHA, exact native raw/EMA geometry, saved Adam moments and clocks, both ranks RNG and source plan SHA. It is intentionally usable before training terminal state only at Root's verified complete-epoch handoff. It claims CPU restoration, not a GPU next-update test. A global acceptance lock serializes the mmap-backed checkpoint validation.

The independent resume source is copied from c366ad4bf11c32f78b2622775c2a7bf83e1a5299 and changes only the missing completed-epoch evaluation recovery branch. It evaluates that epoch before proceeding to the next epoch. Sampling, objective, optimizer, EMA, features, checkpoint format and source data remain the original matched pilot. Active runtime and original ready/entry files were not modified.

Root owns all job pause/cancel/replacement actions and writes a separate 90minute entry using this code root, the original output base and separate job-ID logs. CLI proof and source paths are recorded in actual-stage.json.

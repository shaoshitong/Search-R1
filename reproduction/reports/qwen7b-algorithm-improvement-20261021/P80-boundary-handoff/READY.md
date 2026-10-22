# P80 published-epoch recovery handoff

Source commit `74178e17320d2a3d44eca00766c3061330958bee` was pushed and its remote branch SHA verified. The independent code was staged on HPC3; CPU imports and shell syntax passed. No checkpoint was opened during staging and this deployment did not pause, cancel or submit a job.

The CPU validator accepts only layer15/layer18 Prefix80 epoch1/epoch2 immutable published checkpoints, with actual cursor64×epoch, fresh complete SHA, exact native raw/EMA geometry, saved Adam moments and clocks, both ranks RNG and source plan SHA. It is intentionally usable before training terminal state only at Root's verified complete-epoch handoff. It claims CPU restoration, not a GPU next-update test. A global acceptance lock serializes the mmap-backed checkpoint validation.

The independent resume source is copied from c366ad4bf11c32f78b2622775c2a7bf83e1a5299 and changes only the missing completed-epoch evaluation recovery branch. It evaluates that epoch before proceeding to the next epoch. Sampling, objective, optimizer, EMA, features, checkpoint format and source data remain the original matched pilot. Active runtime and original ready/entry files were not modified.

Root owns all job pause/cancel/replacement actions and writes a separate 90minute entry using this code root, the original output base and separate job-ID logs. CLI proof and source paths are recorded in actual-stage.json.

## Actual CPU file acceptance

At1792688967.534 the independent read-only metadata check observed both layer18 and layer15 epoch1 accepted.json receipts. Each source checkpoint is3770454598bytes; layer18 SHA09dd3c0e6f72901111f231bb8da0bf293968640ea3be3b14fc74ce1aa584ed8c and layer15 SHAad171bc1ed40ab3d08cc2d5be1689189e226feebddde11f666f0deec15a0088f. Both passed exact raw/EMA state loading,46parameter Adam states at64updates, source plan SHA and both ranks serialized RNG checks. Actual CPU verification took15.37/15.31seconds. See actual-both-epoch1-CPU-acceptances.json.

These receipts validate published checkpoint files only. They do not prove GPU step processes were frozen or that no later epoch updates occurred. Root subsequently found stopping only the batch was insufficient to freeze the GPU step; recovery handoff remains Root-controlled at a verified GPU-step boundary, potentially epoch2. Neither original job cancellation nor successful GPU continuation is claimed by this CPU report. The validator and independent missing-evaluation resume branch both support published epoch2/cursor128 without source changes.

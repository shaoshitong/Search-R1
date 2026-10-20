New job742276 is running four GPUs on ACD1-32. User-authorized source is sealed continuation epoch1, Adam10464; previous742046 terminal CANCELLED and outputs retained. New immutable runtime ea845a85e1ad78bc96359bd88f5aae60466ff5f6 is remotely published with per-file SHA checks.

Actual four-rank raw/EMA/Adam/RNG restoration passed. Gradient-only reducer warmup performed without advancing Adam. The next production update from the saved epoch1 state was repeated from restored state and all four ranks matched raw/EMA/Adam hashes bit for bit. First accepted update10465 was saved, SHAa093f32c687f639eac471eacdfedbdb9e92fe0ae8e652633fe76f2900523cd4c, complete optimizer and RNG verified by the trainer CPU reload.

Fresh observations show accepted updates advancing from batch1→12→44 of epoch2, Adam10465→10476→10508. No actor updates. Ranking weight1.0 is applied as a mean over Mixed groups using the global batch Mixed fraction, alongside unchanged trajectory BCE and group BCE weights1:1. Prefix20/LR5e-6/EMA0.99 preserved; question sampling1:1 by whole G5, unique67198 questions per epoch,1318 updates. No diagnostic or2048 held-out question is used for fitting.

Epoch2..5, sealed checkpoint verification, fitting assessment and independent2048→512 results remain incomplete. No claim that five epochs or independent gain have completed.

Independent CPU audit of the immutable first accepted checkpoint also PASSED: full file SHA, raw/EMA tensor geometry/finiteness, all14 Adam parameter states/counters, four-rank model/optimizer hashes and Python/NumPy/Torch/CUDA RNG, exact balanced epoch sampling-plan hash. Auditor ran separately without modifying active training source.

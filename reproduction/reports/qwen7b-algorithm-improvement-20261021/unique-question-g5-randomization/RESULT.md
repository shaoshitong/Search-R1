# Actual169028 metadata: two-epoch sampling experiment

This is a CPU-only experiment with the currently deployed sampler and accepted
169,028-question metadata. It uses a separate reproducible audit RNG (9137),
not the private production E1 sampler RNG. It therefore proves the deployed
algorithm's coverage and randomization; it is not a record of job749015's two
completed training epochs.

Each simulated epoch covered all169,028 questions exactly once, with84,514
Mixed and84,514 nonMixed authentic G5 groups. Both contained1,321 batches,
including the final68-question batch. No hard replay or repeated question
occurred within either epoch.

The question set is necessarily the same across epochs because both perform a
complete sweep. The order differed: no question occupied the same flattened
position in this particular experiment. The per-question selected G5 changed
for168,244 questions (99.5362%);784 questions (0.4638%) happened to draw the same
five-of-ten subset. Exact subset repetition by chance is allowed; these are not
new rollouts. The two G5 subsets shared2.5436 trajectory IDs per question on
average, which is compatible with independently resampling from the same ten.

Source sampler commit: `4ffae871859ab21c9a625b1f5b54ef6679676f07`.
Source SHA256: `0404e3a13afa605cf8614db99179dae5529d230afb969e30856c3a88842211e3`.
Remote receipt: `actual-metadata-two-epoch-randomization-audit.json` in the new
unique-G5 run directory;2,440B,
SHA256 `8d3a587f2c952a3278b42ab38b27d233656c642da3c87a53dbaef59ae2a873f6`.

Actual elapsed CPU experiment time:19.88s; no GPU, checkpoint, model, feature
tensor, or rollout payload was opened.

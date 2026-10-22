# E8 order audit staged; Root submission only

Source commit58a6cf770ed99fd6326fc45d3fcd68099cd239b0, exact packed architecture SHA91dfe479abb080ce20f0da9450c3b47bde1d04a9abec73186fe072ff790c4439 matched the actual full training source publication. Local and actual remote three-test CPU packing suite passed.

Entry: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/packed-Z-E8-trajectory-permutation-audit-20261023-v1/audit.sbatch`

Entry SHA: `f028372b9c94b43f425a246a72bb8e153529eace685f861baeb32d9c62813e78`
Config SHA: `41fbd8e36070501a257fd068f41615f05d09f0ee50d3ebfc8736b64daabced16`

Root alone checks current same-name jobs/locks and submits this entry. No submission or model/CP read occurred in staging. TwoGPU/12CPU/128GiB,normal/acd_u,25minutes. CP8 immutable SHA2d81f15cd4af00fe192963cba6f6db86aba22bf47523bd3e10982b86f7178483 is fresh-verified inside the owned audit allocation. Original/reverse/cyclic1 comparisons share the exact 2048 dev qids, selected ownG5 IDs, strictEM labels and policy tokens. The currently running Z sweep and final4096 remain untouched.

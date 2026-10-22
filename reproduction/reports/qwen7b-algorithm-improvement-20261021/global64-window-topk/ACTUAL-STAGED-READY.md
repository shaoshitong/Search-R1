# Matched current64/window2018 pair staged and ready for root submission

Runtime pin: `237968e8acb8ffd8b5ebcc1c8197247878926ce0`. Remote publication SHA: `db6a7c7cc8b7376ac42dc65fba21086a378079611cc906260c77d7ff6aff8868`. Source files were freshly SHA-verified and the accepted authentic Prefix80 fixture was freshly rehashed/reused. The root submit controller owns both future Slurm submissions; this deployer submitted none.

Suite: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-global64-window2018-L15P80-8192-20261023-v1`. Entries are `current64/training.sbatch` and `window2018/training.sbatch`, two GPUs each,12CPUs,128GiB,2h,acd_u/normal. Both are L15/Prefix80, global64, local32, micro8, accumulation4, ten epochs/1280batch attempts, LR5e-6, EMA.9995.

Actual tiny CPU integration passed: four microbatch DDP forwards per rank followed by one global autograd-all-gather objective matched one-rank global64 gradients to maximum absolute error2.498e-16. The1954 historical scores had no gradient, current64 had nonzero gradient, queue JSON roundtrip was exact, and both zero-margin and positive cached-only/zero-current-gradient cases preserved Adam/EMA by skipping both updates. This is a targeted CPU gradient/queue check; H100 execution and actual checkpoint loading remain untested until the jobs run.

The original four global128 MSE/BCE entries and source remain untouched. Their old128-batch TopK fifth entry is left unrun. Final4096 remains untouched.

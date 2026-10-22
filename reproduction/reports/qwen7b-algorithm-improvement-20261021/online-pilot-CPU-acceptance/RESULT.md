# Actual completed pilot CPU checkpoint acceptance

749077 L27/P20 and749078 L21/P20 both completed three epochs/cursor192, and independently passed entire-file fresh SHA, CPU mmap load, finite raw/EMA, exact real Qwen2 native-block plus direct Mixed-head geometry and strict state loading. Both actual46-state Adam optimizers load against the exact parameter order, all clocks192, both moments finite and second moments nonnegative. Both rank Python/Numpy/CPU torch RNG states are loadable; CUDA RNG byte-states are present (16 bytes each), but CUDA restoration/next-update equivalence was not tested on this CPU process.

One3.770GB checkpoint at a time; observed peakRSS4.420GB below8GB cgroup. Actual checkpoint work took9.89s and9.27s after import. Process92578 exited normally and published run-complete.json. L27/P80 had no completed receipt at this pass and was not opened. No final4096 input, GPU allocation or Slurm mutation occurred. Full trajectory-sampler resume is not claimed.

Source commit42f532ea8039c52de0d6711d3dca4a15ccecdf9c, independently pushed and remote SHA verified. Source validator SHA f42c6f1a6de34660b927f7533bdde9139c157dbc10821855808e3d8dbb58de2f.

Remote receipt root: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-pilot-CPU-acceptance-20261022-v1 . Each completed variant has an accepted.json. The local accepted JSONs preserve exact checkpoint and component hashes.

# Four-cell CPU acceptance support

The original accepted L27/P20 and L21/P20 receipts remain unchanged. The validator now also recognizes L27/P80 and L21/P80, while retaining the same sequential3.77GB mmap/SHA/finite raw+EMA/actual strict native-block-head load/Adam192/two-rank RNG acceptance. Code commit6f7cf7bc2a194a01ed4a0c10724a9b3e2e34cb8a was pushed and remote SHA verified before staging.

New idle entry: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-pilot-CPU-acceptance-20261022-v1/validator-6f7cf7bc2a19/acceptance.sh . Entry849B SHA6c01fe7b5a690eae3a5a9b3646d4129e519235cfd851dcbb723a98cb946cf21c; validator11480B SHA2d20a14360c8fe0dfaf412bca62d5626d7f860b823041037cc782b5f90d80d82. Old validator source was not altered.

Only stage was run, not start. At actual epoch1792681158, L27/P80 job749079 was RUNNING with no complete.json, and L21/P80 job749369 was PENDING with no complete.json; neither checkpoint was opened. Once a pilot is truly complete, use:

`python3 reproduction/operations/qwen7b_warmup/deploy_online_pilot_CPU_acceptance.py --action start --commit 6f7cf7bc2a194a01ed4a0c10724a9b3e2e34cb8a`

The launcher retains one global CPU validation lock, reuses accepted checkpoint receipts, stores new source under its own immutable commit directory, and archives the prior run summary before launching a new pass. It neither submits Slurm nor edits any training source or entry. No final4096 data is read.

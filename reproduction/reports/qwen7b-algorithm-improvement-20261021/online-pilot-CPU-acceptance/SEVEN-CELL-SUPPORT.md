# Seven-cell CPU checkpoint acceptance support

Completed-cell geometry list: L27/P20, L21/P20, L27/P80, L21/P80, L15/P20, L18/P20 andL24/P20. The actual model/Adam/RNG validation is unchanged; only the completed-cell configuration list and the depth-ready training-publication provenance are extended. All old accepted receipts and old validator source directories remain intact.

Source commit a3e13bcd4edb5afe097a0bda0930a3f9f3499eb0 was pushed and remote SHA verified before staging. New immutable CPU entry: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-pilot-CPU-acceptance-20261022-v1/validator-a3e13bcd4edb/acceptance.sh . Entry849B SHA7544368c2eb735dc03797574087831dc38843843a5d978cdbc2174701af247d3; validator12042B SHAef6d2ae0f5c6a2c7fc647c1aaa01f17aee0fc83e64dda85fd4b9d27e511b17b0. Bash syntax passed; no validator started during staging.

After an actual completed pilot, launch the existing controller with `--action start --commit a3e13bcd4edb5afe097a0bda0930a3f9f3499eb0`. It reuses old accepted receipts, accepts only actual terminal COMPLETED jobs and complete epoch3/cursor192 files, and skips unfinished variants without opening their checkpoint payloads. One global CPU acceptance lock prevents duplicate validators.

At epoch1792681400.3125, L27/P80 job749079 andL21/P80 job749369 were both RUNNING and both lacked complete.json. Neither was opened or claimed accepted. No Slurm mutation or final4096 EM/score read occurred.

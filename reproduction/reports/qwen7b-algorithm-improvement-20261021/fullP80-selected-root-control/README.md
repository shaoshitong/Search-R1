# Selected full-P80 execution handoff

Runtime source `516e3d2a6b9e8df53fc5a22fc73297a8d6617f85`; stage/readback evidence `df8dbcfe7068322b7cc7cba065b743e591a212f9`. Five CPU metadata tests and the real remote help/syntax/source readback passed. This is a required execution helper, not a new algorithm or an actual selected candidate.

Remote entry:

```sh
bash /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fullP80-selected-root-control-20261023-v1/control.sh --action bind-audit
```

Its SHA256 is `1d76063c65e7fdb82462fc627b4b098d21798860cd3f21d70aa9207b02e96a90`. The earlier live-training bind returned `WAITING_ACTUAL_PREREQUISITES_NO_BIND_FREEZE_OR_SUBMISSION`; that historical WAIT is retained in [its original receipt](actual-first-root-bind-wait.json) and is no longer the current execution stage.

Actual completed execution (2026-10-23):

- Both jobs **750189/750190 completed 10 epochs and 13210 Adam/EMA updates with `COMPLETED0:0`**. The original followers completed E10 CPU acceptance and exited naturally: [seed1 E10](../full169028-L15P80-MSE-CPU-acceptance/seed1-actual-E10-terminal-accepted.json), [seed2 E10](../full169028-L15P80-MSE-CPU-acceptance/seed2-actual-E10-terminal-accepted.json).
- The [family selector chose E9/raw](../full169028-P80-MSE-family-selection/actual-run.json) using the predeclared seed1-only rule, applied unchanged to seed2. [Both selected E9 checkpoints passed sequential immutable693 CPU acceptance](../full169028-L15P80-MSE-CPU-acceptance/BOTH-SELECTED-E09-CPU-ACCEPTANCE-SUMMARY.json): 11889 Adam/EMA updates, actual CP SHA, raw/EMA, two-rank RNG, saved plans and physical/P20 proofs. This CPU proof does not claim a tested CUDA next Adam update.
- Original audit **750474 failed during CUDA initialization**; its evidence remains intact. The same-binding successor **750476 completed `0:0` in 2m28**. [Actual audit results](actual-audit-live.json) report original/reverse/cyclic1 and successful original-score replay for both E9/raw seeds. The binding SHA remains `1bca7df0a5ba108758f7082469b2352e260fbc711047382a99ea1f8925a1acdc`. Order sensitivity remains: Top512 Mixed counts are seed1 **329/325/312** and seed2 **321/318/314** in original/reverse/cyclic1 order. No advantageous order was selected.
- Root wrote the [actual candidate freeze](actual-candidate-freeze-raw.json), SHA `2606a04636e282df2e011c962fe47c29c00be4460f3581f3fc03b58fd5d428ca`, before final access. It binds the same E9/raw state, both accepted seeds and the unchanged prospective Top512 rule.
- Final job **750481 completed `0:0` in 4m54**, scoring the original two fixed2048 pools (4096 questions/20480 ownG5 trajectories): [raw complete receipt](actual-final-complete-raw.json), [actual terminal read](actual-final-live.json). Confirmation Top512 Mixed is E1 **209**, seed1 **341**, seed2 **331**; stability is E1 **221**, seed1 **339**, seed2 **345**. Both two-seed mean paired-question95% lower bounds against E1 are positive: **+0.201171875** and **+0.1952880859375** in fraction units. Every individual seed/pool point improves; the prospectively fixed primary criterion passes. These are paired-question intervals, not training-seed confidence intervals, and the mean is not an ensemble.
- [Owned resource closeout](actual-owned-resource-closeout.json) confirms750189/750190/750334/750476/750481 completed,750474 remains its recorded historical failure, the owned queue is empty and all listed task locks are free. No outside process was stopped.

The execution and acceptance chain is complete. **Both final pools are now viewed** and cannot be used for further candidate/epoch/state/order/threshold selection or tuning; new exploration needs fresh held-out questions. GPU forward scoring and numerical replay do not establish next-Adam-step GPU restoration. The measured trajectory-order sensitivity remains a limitation.

Only root-owned explicit submission actions were used; prior unknown outcomes were reconciled. Existing objective, fixtures, trajectories and final pool identities were preserved. This local evidence-book update performs no remote read, CP load, process mutation, additional evaluation or push; root handles goal/STATE/memory and the unified publication.


Final acceptance records: [primary confirmation](FINAL-CONFIRMATION.md), [machine primary check](FINAL-PRIMARY-ACCEPTANCE.json), [10-point execution/scientific/resource acceptance](GOAL-COMPLETION-ACCEPTANCE.json). The separate task metadata observer is also confirmed naturally exited with its terminal receipt, no training failure; no other task was stopped. The final archive publication SHA and goal-tool completion are recorded by root outside this immutable evidence snapshot after push verification.

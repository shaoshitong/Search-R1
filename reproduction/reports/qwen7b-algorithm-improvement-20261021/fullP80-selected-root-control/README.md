# Selected full-P80 execution handoff

Runtime source `516e3d2a6b9e8df53fc5a22fc73297a8d6617f85`; stage/readback evidence `df8dbcfe7068322b7cc7cba065b743e591a212f9`. Five CPU metadata tests and the real remote help/syntax/source readback passed. This is a required execution helper, not a new algorithm or an actual selected candidate.

Remote entry:

```sh
bash /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fullP80-selected-root-control-20261023-v1/control.sh --action bind-audit
```

Its SHA256 is `1d76063c65e7fdb82462fc627b4b098d21798860cd3f21d70aa9207b02e96a90`. The earlier live-training bind returned `WAITING_ACTUAL_PREREQUISITES_NO_BIND_FREEZE_OR_SUBMISSION`; that historical WAIT is retained in [its original receipt](actual-first-root-bind-wait.json) and is no longer the current execution stage.

Current actual progress (2026-10-23 08:33 CST):

- Both jobs **750189/750190 completed 10 epochs and 13210 Adam/EMA updates with `COMPLETED0:0`**. Their original terminal followers completed E10 CPU acceptance and exited naturally. The raw receipts are [seed1 E10](../full169028-L15P80-MSE-CPU-acceptance/seed1-actual-E10-terminal-accepted.json) and [seed2 E10](../full169028-L15P80-MSE-CPU-acceptance/seed2-actual-E10-terminal-accepted.json).
- The existing [family selector actually selected E9/raw](../full169028-P80-MSE-family-selection/actual-run.json) under the predeclared seed1-only rule; seed2 uses the same epoch/state without reselection.
- Both selected E9 checkpoints passed the existing immutable693 CPU acceptors sequentially: [actual acceptance summary](../full169028-L15P80-MSE-CPU-acceptance/BOTH-SELECTED-E09-CPU-ACCEPTANCE-SUMMARY.json). Each has 11889 Adam/EMA updates, actual CP SHA, raw/EMA, two-rank RNG, saved plans and physical/P20 input proofs. CPU acceptance does not claim a tested CUDA next update.
- Root uniquely submitted existing order audit **750474**, `acd_u`/`normal`, two GPUs, no dependencies. Its binding SHA is `1bca7df0a5ba108758f7082469b2352e260fbc711047382a99ea1f8925a1acdc`: [actual submission](actual-audit-submission.json). The [first live read](actual-audit-live.json) shows RUNNING but no measured scoring progress yet. **Audit success remains pending.**
- **Actual candidate freeze and final scoring have not occurred. Final4096 remains blind.**

Remaining root-controlled execution:

1. Follow the existing audit handle750474 through real scoring progress and successful original-score replay, reporting original/reverse/cyclic1 for both same-E9/raw seeds. Do not submit another audit or choose an advantageous order.
2. Root explicitly calls `--action freeze-final` only after that actual order audit and the already completed750334 fixed32 E1 numerical replay pass. This writes the actual candidate freeze with full CP/source/CPU/plan/G5 proofs and the unchanged prospective Top512 confirmation rule. The existing family selection is not this final candidate freeze.
3. Root explicitly calls `--action submit-final`, then follows real inference through both original fixed2048 pools and final paired analysis. This is the first authorized final access; do not read final labels/predictions or select another candidate beforehand.

All actions return WAITING before missing actual prerequisites. Only explicit submit actions call Slurm; unknown prior submission outcomes are reconciled rather than blindly retried. The existing GPU entries, objective, training, fixtures, trajectories and final pools are unchanged. CP SHA is inherited from independent CPU acceptance and only CP size is statted by this login controller.

The goal remains active until final confirmation and the remaining acceptance/resource requirements pass. This update uses existing local receipts only; it starts no process, loader, Slurm job or final access.

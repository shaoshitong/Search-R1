# Selected full-P80 execution handoff

Runtime source `516e3d2a6b9e8df53fc5a22fc73297a8d6617f85`; stage/readback evidence `df8dbcfe7068322b7cc7cba065b743e591a212f9`. Five CPU metadata tests and the real remote help/syntax/source readback passed. This is a required execution helper, not a new algorithm or an actual selected candidate.

Remote entry:

```sh
bash /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fullP80-selected-root-control-20261023-v1/control.sh --action bind-audit
```

Its SHA256 is `1d76063c65e7fdb82462fc627b4b098d21798860cd3f21d70aa9207b02e96a90`. Root actually invoked this default metadata action while training was live: it returned `WAITING_ACTUAL_PREREQUISITES_NO_BIND_FREEZE_OR_SUBMISSION` because completed family selection does not yet exist. No checkpoint payload, final data, binding, freeze or Slurm submission occurred.

After both existing jobs750189/750190 actually finish10E/13210 successfully:

1. Run the existing family selector with source `427597a2e6031944e229e5f6751bd724e51a3848`, using `deploy_full169028_P80_MSE_family_analysis.py --commit ... --action run`. Its seed1-only rule chooses epoch/state; seed2 never reselects.
2. Reuse actual accepted E10 receipts if selected epoch is10. Otherwise invoke the already staged immutable693 CPU entry for **only** the selected epoch of each seed, sequentially. Those children own the common CPU mutex; do not wrap them in that same outer lock. Require actual selected-epoch acceptance, not just an E10 receipt.
3. Root explicitly calls the control entry with `--action submit-audit`. It binds the same actual epoch/state and accepted CPs into the existing two-seed original/reverse/cyclic1 GPU audit entry and persists a submit-once intent. Follow the resulting owned job through real progress and successful original-score replay.
4. Root explicitly calls `--action freeze-final` only after the actual order audit and already completed750334 fixed32 E1 numerical replay pass. This writes the actual candidate freeze with full CP/source/CPU/plan/G5 proofs and the unchanged prospective Top512 confirmation rule.
5. Root explicitly calls `--action submit-final`, then follows real inference through both original fixed2048 pools and final paired analysis. This is the first authorized final access; do not read final labels/predictions or select another candidate beforehand.

All actions return WAITING before missing actual prerequisites. Only explicit submit actions call Slurm; unknown prior submission outcomes are reconciled rather than blindly retried. The existing GPU entries, objective, training, fixtures, trajectories and final pools are unchanged. CP SHA is inherited from independent CPU acceptance and only CP size is statted by this login controller.

Neither stage nor this early WAIT means that selected CP acceptance, order audit, actual candidate freeze or final confirmation has completed. The goal remains active until genuine independent improvement and all remaining acceptance/resource requirements pass.

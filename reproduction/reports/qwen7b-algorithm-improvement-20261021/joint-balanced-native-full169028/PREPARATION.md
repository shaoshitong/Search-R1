New independent controller: deploy_joint_balanced_native_full169028.py. Current124194 source, PID, configuration and bank are untouched. No GPU was submitted.

The runtime is read from published Git objects, not shared working-tree files: three new worker/protocol/evaluation files from root ec7347 (byte-equal to e0559d), nineteen unchanged helpers from qualified3b56e5, and the c98ddc sampler. The exact23-file import closure and archive per-file origins passed CPU tests. Real prepare additionally checks unchanged helper hashes against the existing124194 runtime publication.

The only accepted data input is A/joint169028-native-training-bank-v1 with actual accepted status,169028/1690280 counts, canonical qid SHA, source-before-feature-union flag and true6656 identity exclusions. A missing/incompatible bank cannot create configuration or an entry. E4 full state/Adam14418, LR5e-6, EMA.99, native block27 plus readout, trajectory1+binaryMixed1 and batch32 are preserved. The budget is4x5283=21132 accepted updates; old-budget comparisons are3882/7764/11646/15528. Each integer epoch and matched node pins a full checkpoint before fixed512/2048 raw+EMA evaluation. Final4096 is identity-only.

The prepared sbatch requests normal/acd_u,1H100,8CPU,128GiB,48h and no dependency. Submit requires a future explicit --submit-authorized plus the existing same-name/intent/lock/held-identity reconciliation guards. Unknown submission outcomes retain intent and forbid blind retries.

Code-only commands, after publication:

    python3 -m reproduction.operations.qwen7b_warmup.deploy_joint_balanced_native_full169028 --action stage
    python3 -m reproduction.operations.qwen7b_warmup.deploy_joint_balanced_native_full169028 --action prepare
    python3 -m reproduction.operations.qwen7b_warmup.deploy_joint_balanced_native_full169028 --action status

Five focused tests passed: fail-closed missing/wrong bank; actual production config against worker protocol; real mixed-origin source archive/manifest; resource entry syntax and bounded control; explicit submit authorization before network. No tensor/model loads, real bank prepare or GPU launch occurred during this source task.

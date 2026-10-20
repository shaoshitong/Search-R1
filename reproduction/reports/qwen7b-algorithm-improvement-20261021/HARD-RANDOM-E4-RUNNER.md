# E4 hard-versus-random continuation, implementation ready

New independent entry points are `hard_enrichment_train.py`, `hard_enrichment_train_prepare.py`, and `deploy_hard_enrichment_suite.py`. No active runner or source checkpoint was modified. Deployment is gated on the actual E4 candidate-scoring receipt and actual CPU `selection-complete.json`; absent inputs fail before staging or job submission. The 32,768 candidates already exist, but the scored/selected two training branches are not assumed to exist.

Both branches resume the same sealed **E4 raw model, EMA, Adam at step 14,418**, SHA `e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8`. They preserve Adam hyperparameters, moments, and canonical source rank0 RNG and archive the original four ranks. This is a new single-GPU continuation, with no claim of four-rank bit-exact equivalence; no E1 reset occurs. Initial model construction is followed by strict restoration of all trained raw/EMA tensors and native RoPE buffers, never treated as reinitialization.

The common loss is trajectory BCE ×1 + three-event NLL ×1 + inter-question ranking ×1. The existing complete-global-batch loss/update implementation is reused without monkeypatching its E1 source guard. Each actual B48 forward retains its microforward graphs until the joint loss backward, so inter-question ranking sees the complete global batch. Each branch takes exactly one pass over 22,896 distinct questions, **477 real optimizer updates**, ending at Adam step 14,895. Two configs differ only in branch identity, training groups, and their frozen plan; source, loss, seed, microbatch, update budget, diagnostic and fixed validation are shared. Macro classes are matched; Mixed K=1..4 distributions are explicitly not claimed matched.

LR stays 5e-6, EMA 0.99, weight decay zero. Features are native capture L26, question+Prefix20, Q273/D3584. The original fixed diagnostic512 and accepted new validation's precommitted label-blind G5 are evaluated; the old viewed G5 or final confirmation pools are not substituted. Evaluation reports group balanced accuracy, AUC, separate all-wrong/all-correct false-positive rates, Mixed recall, and actual Top512 Mixed, in addition to losses/timings.

The first available complete B48 update is executed twice from the same full restored state and compared by raw/EMA/Adam/native-buffer/RNG hashes before acceptance. The accepted first update, every 32 updates, clean interruption and the final cursor are saved with full raw/EMA/Adam/buffers/RNG, data cursor, source four-rank archive and immutable config/source/plan bindings. Each save is independently CPU-loaded and component-hash checked. Unsaved later progress or an unresolved later optimizer transaction causes an explicit error, never silent rollback. The final cursor477 full checkpoint must exist before evaluation. Evaluation restores RNG and native buffers and must leave all accepted state hashes unchanged. A failure during evaluation can resume evaluation from the final full checkpoint.

Normal `acd_u`, QoS normal, one GPU / 8 CPUs / 64 GB / two hours per branch; no dependency. Code is staged from the published Git revision into a fresh immutable code root using SHA-bound 4,096-byte chunks. Real control commands are below16KB. Preparation launches a guarded asynchronous CPU-only metadata/test worker, retaining durable intent and result; submission is a separate explicit action with hold/identity/release and unknown-outcome protection. Repeated prepare/status or submit does not create a second writer.

```sh
python -m reproduction.operations.qwen7b_warmup.deploy_hard_enrichment_suite --self-test
python -m reproduction.operations.qwen7b_warmup.deploy_hard_enrichment_suite \
  --control-root /absolute/control/root --action prepare
python -m reproduction.operations.qwen7b_warmup.deploy_hard_enrichment_suite \
  --control-root /absolute/control/root --action status
python -m reproduction.operations.qwen7b_warmup.deploy_hard_enrichment_suite \
  --control-root /absolute/control/root --action submit --variant targeted --submit-authorized
```

Local CPU qualification covers real tiny Adam continuation/replay, full checkpoint save/load/next-update equality, B48 global pair counts, plan uniqueness/isolation, strict E4 source/budget/loss guards, unsaved-progress rejection, matched branch configs, actual shell syntax and import-closure/control-command bounds. This does **not** claim real E4 Adam/GPU qualification: that remains an allocated-node acceptance step. No remote preparation, submission or GPU training has been performed by these new entry points yet.

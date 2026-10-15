# Fresh twenty-token correctness head

The frozen official Qwen2.5-3B actor and encoder stay unchanged. Each sample's
strict EM supervises only that same trajectory. The encoder sees the actual
question/context and the retained policy prefix; layer index27 post-block is
detached. The copied official block35 and a width20 flattened linear output see
only the actual generated positions, including real position gaps between tool
turns. No question or retrieval hidden position enters the head.

Each epoch traverses all 25,600 trajectory samples exactly once. This latest
protocol deliberately replaces the old balanced replacement sampler. The loss
is unweighted mean BCEWithLogits at the natural class prior. The default is a
single global batch25,600, local3,200 on each of eight DDP ranks, no gradient
accumulation: one real Adam update per epoch, 1,000 planned updates per rank.
Learning rate remains3e-4. A smaller configurable batch remains possible after
an actual memory failure; its actual update count must be reported as changed.

The first epoch reuses the immutable historical20-token training feature cache.
All later epochs first finish their own fresh rollout, then encode and fit that
epoch before sampling the next one. The fixed historical512-question/2,560-
trajectory diagnostic feature cache is reused, without re-generating test
trajectories. Evaluation occurs after every epoch with raw accuracy, balanced
accuracy, BCE, Brier, AUROC and class counts. Predictions and tensors stay remote.

The standalone phases release GPU models between them:

1. Rollout writes `complete.json` and eight gzip compact prefix row shards.
2. `python -m reproduction.prefix20_online.features --rollout-complete ... --output EPOCH/inputs`
   materializes exact short contexts on remote CPU.
3. `torchrun --nproc-per-node8 -m reproduction.prefix20_online.features --inputs EPOCH/inputs --model MODEL --output EPOCH/features --micro-batch-size32`
   extracts only the generated twenty-token representations on all eight GPUs.
4. `torchrun --nproc-per-node8 -m reproduction.prefix20_online.train_epoch --train-cache EPOCH/features --test-cache FIXED20 --test-manifest-sha SHA --model-dir MODEL --output TRAINROOT --epoch N`
   restores the immediately previous complete head/Adam/eight-rank RNG, fits
   once, evaluates, and atomically publishes `latest.pt` with its SHA receipt.
   Immutable hard-linked head/Adam/RNG checkpoints are kept every ten epochs and
   at the final boundary. Atomic replacement of latest preserves old hard links.

Epoch1 uses the historical cache for `--train-cache` directly and starts a fresh
official block35 copy, with a newly initialized linear head. There is no actor
update, PPO restart, additional baseline rollout or numerical preflight gate.
The trainer rejects an actual nonfinite loss/gradient before committing Adam.

Rollout compact row fields are `trajectory_id`, `qid`, `question_hash`,
`strict_em`, `prefix_segments`, `policy_token_ids`, and `policy_position_ids`.
Each segment holds its actual `context_tokens`, `request_length`, `turn_index`,
`policy_start`, `policy_length`, and `position_start`. The row may also retain
`normalized_question_hash`. Complete fields are `epoch`, `questions5120`,
`trajectories25600`, `model_revision`, `model_files_sha256`, `model_dir`,
`pad_token_id`, `selection_sha256`, and `shards[{path,sha256,bytes,count}]`.

The rollout/pool owner guarantees fresh question sampling without repeats until
the finite pool is exhausted, excludes the fixed diagnostic, and regenerates
each recycled question's own new G5 trajectories. The head verifies every epoch
is complete and disjoint from the fixed diagnostic. It never infers future
prefix features from final answers or labels.

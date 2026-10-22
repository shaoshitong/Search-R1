# Local packed variance implementation — not deployed

User-requested new architecture: a single genuine question prompt followed by M trajectory Prefix20 segments, M uniformly random from 3 through 10 for each training question and epoch. The text separator is `\n### trajectory\n`, tokenized with the existing official tokenizer; there are no newly introduced vocabulary tokens/embeddings. Every selected trajectory contributes at most its first20 retained policy tokens, with explicit masked padding when shorter. Trajectory order is random.

The actor sees **one packed row per question group**, not M independently prefetched rows. The official frozen actor retains blocks0..20, whose output is fed into a trainable copy of official block21 and a group regression head. This is a deliberately new packed group context: genuine token sources are repacked, and this input is **not claimed to preserve each rollout's original causal tool context**. No full answer, retrieved documents, strictEM, target or label is placed in the model input. The detached actor hidden states are transient GPU tensors, never persisted.

M and the selected subset are sampled before looking at EM. Labels use the selected M actual completed trajectory strictEMs: `v = mean(EM)*(1-mean(EM))`, population variance (no Bessel correction). The head predicts `sigmoid(z) = 4*vhat` in [0,1]. **The sole training loss is ordinary MSE(sigmoid(z),4v)**. No trajectory BCE, Mixed BCE, ranking loss, class resampling or class weights. Raw parameters receive gradients; EMA tracks raw at decay0.99.

Selection is descending `predicted_population_EM_variance`, equivalent to descending4v, with qid tie-break. All2048 candidates use the same fixed originalG5 for comparable Top512 realMixed reporting. strictEM is not used to rank. Predictions are not Mixed probabilities. The sole predeclared G5 classification diagnostic is `vhat >= 0.08` (midpoint between homogeneous0 and the smallest positive trueG5 variance0.16), with no fitted threshold. FixedM3,5,10 diagnostics use an independently predeclared seed2026102228 and report regression error/rank correlation. Final4096 is never read.

Initial pilot: same8192 training question IDs as layer pilots, all own10 genuine trajectories; each epoch visits each q once in a fresh random order, but M/subset/order are resampled independently of EM. 3epochs/192updates, LR5e-6, global128/local64/micro8/accum8, 2GPU12CPU128GiB/30min/normal acd_u. Fresh official block21, new head and Adam; no existing experiment state is overwritten. The 8192 pilot is not the full169028 goal. The runtime supports `--training-questions 169028 --full-training-descriptor PATH`: descriptor provides accepted groups ref and accepted genuine Prefix20 source refs. Group metadata is canonical full169028G10; no trajectory regeneration is assumed. Full entry/time/budget will require an explicitly bound full-run stage.

Changes include packed context, variableM, new group head and variance objective together, so outcomes cannot be attributed to MSE alone. Existing full training and randomG5 controls remain unchanged.

## Local verification

`python3 -m reproduction.operations.qwen7b_warmup.test_packed_variance_probe` passed all5 CPU tests in0.342sec. Tests cover ownid sampling, M3..10 range and label-blindness, exact population targets, seed/epoch shuffle, one question occurrence, shortpad masks and target exclusion, actual toyblock/head backward with detached actor features, and label-blind predicted-variance ranking. No7B weights/assets, remote state, CUDA, or real dataset payload was loaded locally. Static compilation also passes.

Remote prepare has not run: it reuses accepted8192 fullG10 Prefix20 token rows, then supplements the independent2048 pool's missing secondG5 from already accepted G10 full trajectory journals. It extracts at most20 policy tokens per genuine source, strips all labels from token rows, writes only remote compact prefix fixtures, and emits source SHA plus count receipts. GPUs must not be submitted until102400 own trajectory rows and train/validation identities are accepted.

## Execution after connectivity resumes

1. Push the local source commit to `codex/q7b-packed-variable-M-variance-20261022`, then verify ls-remote.
2. Run `deploy_online_packed_variance.py --action stage --commit SOURCE_COMMIT`. It stages immutable code, standalone entry and ready binding; it never calls sbatch.
3. Run `--action prepare` once and `--action status` to confirm actual CPU fixture progress/completion. Do not infer prepare completion from PID alone.
4. Root alone executes `packed-variance/submit-once.py` via the private control wrapper. It checks ready/fixture/source/token SHA, reconciles existing receipt/intent/queue, submits held2GPU30minnormal entry once, validates actual Slurm identity, then releases.
5. Follow two real updates and one full epoch, then full3epoch/checkpoint/evaluation acceptance. GPU/native7B forward, DDP, remote fixture schema and source metadata are not yet runtime-verified.

Network unavailable at handoff: **not pushed, not deployed, not submitted**. Existing remote job status is unknown after the final earlier observation; no exit is inferred from connectivity loss.

# Matched packed-input objective pilot — source only

No remote stage, Slurm submission, model loading or final4096 read has been performed for this entry.

The four factorial cells are `objective={mse,mixed_bce}` × `fixed-m={0,5}`. Here `0` draws M uniformly from 3…10; `5` always selects five own trajectories. All cells use the same 8,192 distinct training questions once per epoch, independently shuffled question order and own trajectory subsets each epoch, the same fixed train2048 G5 and viewed independent2048 G5, identical fresh official copied block and identical packed head. Sampling never consults EM. Objective changes cannot change the plan; fixed-M changes the trajectory budget by design. Each cell is 2 GPUs, global128/local64/micro8/accum8, 64 updates per epoch, ten epochs/640 global batch attempts (640 Adam/EMA updates for BCE/MSE; ramp may skip satisfied-margin batches), LR5e-6 and EMA.9995. There is no class weighting, replay, trajectory classifier, or extra loss. The optional fifth arm is `objective=topk_ramp,fixed-m=5`, using the same layer15/Prefix80 and ten-epoch budget.

`mse` predicts four times population EM variance with sigmoid and MSE; `mixed_bce` predicts the true Mixed label directly with binary cross entropy on the same single group logit. Fixed evaluation uses the predeclared variance threshold0.08 (normalized score0.32) or direct Mixed threshold0.5 respectively. Reports include raw/EMA Mixed BA, ROC-AUC, Top512, classwise errors and score/variance rank correlation. TopK-ramp uses the published group-level `precision_at_k_ramp` on global logits gathered with autograd, with K=ceil(actual global batch/4), margin1, and no BCE/variance anchor. At batch128 this is K32: it is a minibatch proxy, not a theorem about the future2048-question pool. A satisfied global margin performs backward for DDP synchronization but skips both Adam and EMA updates, preserving optimizer moments unchanged. Checkpoints distinguish batch attempts, actual Adam updates, EMA updates and margin-satisfied skipped batches. Its sigmoid(logit) report score is explicitly uncalibrated; threshold0.5 is a secondary predeclared diagnostic. Variance is never reported as a Mixed probability, and its MSE is never called binary BCE.

Supported geometry is `--layer 15|21 --prefix 20|80`. Both use the same 2,048-position head capacity, which also covers real Prefix80 packed sequences. Preparation reads the accepted original causal Prefix80 token fixture, not the derived Prefix20 bank. It extracts only authentic retained policy tokens; genuinely short trajectories are masked/padded. Prefix80 is never manufactured by repeating or extending Prefix20. The original fixed-G5 identity SHA is bound in the fixture receipt. Hidden features remain online in GPU memory.

The current declared geometry is layer15/Prefix80. Root selects the isolated deployment path before running any of these commands. In the published code root:

```bash
bash reproduction/operations/qwen7b_warmup/runtime.sh -m reproduction.operations.qwen7b_warmup.online_packed_matched_objective \
  --action prepare --base "$NEW_SUITE/fixture"

# One isolated output directory per cell; same L/P for all four cells.
bash reproduction/operations/qwen7b_warmup/runtime.sh -m torch.distributed.run --nproc_per_node=2 \
  -m reproduction.operations.qwen7b_warmup.online_packed_matched_objective \
  --action train --base "$NEW_SUITE/mse-Mrandom" --fixture-base "$NEW_SUITE/fixture" \
  --objective mse --fixed-m 0 --layer 15 --prefix 80 --epochs 10 --lr 5e-6 --ema-decay .9995
```

The fifth cell uses only `--objective topk_ramp --fixed-m 5 --margin 1.0`, in its own directory. The remaining factorial cells replace only the own output directory and `--objective/--fixed-m`: `mse/5`, `mixed_bce/0`, `mixed_bce/5`. Each epoch saves raw, EMA, Adam, both rank RNG, cursor, exact plan SHA and immutable geometry/objective/fixture binding before evaluation. Resume requires that binding to match. Existing jobs and their source files are untouched.

Validation performed at preparation of this source: Python compilation and exact byte equality of the reused helper/runtime against published7ac274ac. GPU forward/backward and actual accepted fixture preparation remain unrun until root authorizes staging.

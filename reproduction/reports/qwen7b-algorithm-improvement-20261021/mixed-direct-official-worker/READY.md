# Official-start direct Mixed-only worker ready

Worker source: `e2fd6e7fbe4fbd6d62fc86923565c01765d02794`, pushed and remotely verified on `codex/mixed-direct-official-worker-20261022`. Only the three named operation files are new; the retired full dual-BCE resume drafts are excluded. Model uses `8c0ccaf4`, direct evaluation `7380f3bf`, stateful real-G10 sampler `c98ddcd7`, and remaining qualified support `3b56e583`.

```bash
python -m reproduction.operations.qwen7b_warmup.joint_mixed_direct_pretrain \
  --base /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-v1
```

Use the exact `typed_config()` contract. Entry must export `FLASH_ATTENTION_DETERMINISTIC=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch. Official inventory is an explicit absolute SHA reference and can be outside the model asset directory. Every declared actual model file is freshly SHA-verified against official LFS or Git-blob metadata; the index must cover all shards. Only official native block27 is loaded into the probe, with raw=EMA and both Adam0. Old trained probe checkpoints are never opened.

The native block encodes each question's ten real trajectories once. Its four chosen own G5 views go to the direct binary group head, each question's mean four-view BCE is divided by32 and immediately backpropagated. Both native block and group head get gradients; only Mixed BCE is optimized. Sampler observations retain the original shuffled view order. There is no p_j, product formula, trajectory loss or trajectory accuracy.

Full checkpoints bind raw, EMA, both Adam states, both native buffer sets, all training RNG, sampler/hardness/data position, official source, stage and cursor. Complete CPU reload precedes latest publication. CP0/1/2 and completed epochs are immutable; remaining accepted checkpoints roll every128 updates. Evaluation happens only after saving at0/5283/10566/15849/21132. Existing incomplete epoch evaluation resumes using the same accepted checkpoint, without advancing training.

Four bounded synthetic CPU fixtures actually passed in1.902s (5.978s including imports, 2CPU/0GPU, peakRSS541245440B). See `actual-cpu-qualification.json` for exact remote receipt and hashes. This qualifies the mathematical accumulation and checkpoint machinery, not a real H100 update. No GPU was submitted by this implementation task; the deployment owner must still qualify actual CP1, next update, both changed parameter groups, and five complete updates plus the next before reporting healthy training.

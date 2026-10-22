# Reproducible second training-seed boundary

This is a static compatibility note only. No seed change, new source deployment, rollout generation, Slurm action or final4096 read occurred.

| Recipe | Initial head/RNG | Question order | OwnG5 / randomM |
|---|---|---|---|
| Full169028 variance/Z P20 | Torch/NumPy/Python9137 | packed-question-order,2026102227,epoch | packed-population-variance,seed,epoch,qid; M drawn before trajectories |
| Original nativeP80/depth pilots |9137, including newP80 slot weights | frozen plans generated with20261022 | frozen class-feasible conditional ownG5 |
| Native naturalG5 control | unchanged core9137 | unchanged original three epoch orders | metadata plans from2026102207 |
| Packed P80 MSE/BCE |9137 |2026102227 packed order | same per-qid seed; fixed5 bypasses randomM draw |
| Packed P80 window pair |9137 | imported matched-module plan | fixedG5 via matched epoch_plan; default2026102227 |

All current entry parsers lack init_seed/train_plan_seed CLI arguments. Changing sbatch environment, PYTHONHASHSEED or the global random seed does not override the explicit private Random instances and SHA-derived seeds. The window module imports the SEED value and the epoch_plan function; changing only window.SEED changes its record but not the function’s owning matched-module globals.

The official block comes from fixed Qwen revision a09a35458c702b33eeacc393d103063234e8bc28, so a training-seed replicate changes the new readout/head and sampled training order/views, not pretrained block parameters.

A second seed requires a new immutable source/config binding and fresh output root. Existing latest.json causes an exact state resume, which restores raw/EMA/Adam/RNG and is not a fresh second-seed initialization. Full runtime currently omits explicit init/plan seed fields from its checkpoint metadata; a new seed-capable fork should bind both fields rather than trusting an environment setting.

Possible second pair, not applied: init_seed9138 and train_plan_seed2026102237. Keep fixedtrain2048, fixeddev2048 G5, data identities, official weights and diagnostic_seed2026102228 unchanged. Share each second-seed plan across the compared objective/cache arms and record initial head/plan hashes.

Original native and packed samplers are different recipes: native original plans were class-feasible/conditional, while packed plans are EM-blind. A changed seed alone does not turn them into an architecture-only matched comparison. NaturalG5 control preserves the original epoch orders and changes only own5 draws.

For native variants, a new small fixture/plan directory can reference the same existing token shards. For packed/window variants, explicitly pass the seed into the owner of epoch_plan. No new rollouts or hidden caches are needed.

Detailed file SHA snapshots and per-recipe facts are in NOTES.json.

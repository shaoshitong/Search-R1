FullP80 MSE terminal CPU coordinators

One independent durable worker per actual job:750189=seed9137/plan2026102227;750190=seed19137/plan2026102302. Root alone starts them with explicit --seed. Source-stage does not start workers or CPU readers.

Running training: only accounting, complete/controller/progress metadata; no checkpoint or model packages, no shared CPU lock. A real COMPLETED0:0 and exact10epoch/13210update complete receipt are both required. Then the worker calls the already immutable693a81df CPU entry with --epoch10. The existing validator owns COMMON_LOCK around every full checkpoint SHA/load/restore, so do not add a second outer flock on the same file. Both seed workers may wait concurrently, but their CPU loads serialize through that existing lock.

Existing exact CPU acceptance is reused only after matching actualjob/epoch/13210steps/binding/seed/physical fixture+P20/source/complete refs. Terminal training or validation failure leaves failure.json and is not blindly restarted. Unknown existing same-job worker/validator identities stop duplicate launching.

No Slurm mutation, hot changes, training cancellation or final4096 read.

Actual stage source9772c3a72e4b5d5190b2b39673cce8e893d781b0; worker SHA03f147a6405ee3ea44e62e1a3097d8202120956af444be4fb378583efa7afd38.

Root start commands from repository cwd:

```sh
python3 reproduction/operations/qwen7b_warmup/deploy_fullP80_MSE_terminal_CPU_coordinators.py --commit 9772c3a72e4b5d5190b2b39673cce8e893d781b0 --action start --seed 9137
python3 reproduction/operations/qwen7b_warmup/deploy_fullP80_MSE_terminal_CPU_coordinators.py --commit 9772c3a72e4b5d5190b2b39673cce8e893d781b0 --action start --seed 19137
```

Status uses the same helper with --action status (optional --seed). Do not directly execute an entry just to check --help: it is a real coordinator launcher.

Job750189 entry /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fullP80-MSE-terminal-CPU-coordinators-20261023-v1/code-9772c3a72e4b/job-750189.sh; SHA 28d6176720789285019e3e0a1435323a4a0f756429f76889a0e0ddb0be43089c; configSHA 85297248f2d861773dfc56eca861f9245f11ee8ccfcd2e6e3cbd2e388ea53cb3

Job750190 entry /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fullP80-MSE-terminal-CPU-coordinators-20261023-v1/code-9772c3a72e4b/job-750190.sh; SHA 0b17900e8c95429832b2dd2f4f736331ddd0adab381382a997aa09522a9e0063; configSHA 256ceb057420b707b88aa594e10bc8232388b1c772663f25baf5d7d721f26b9e

Both entries/configs/source and original693 entries/validator were independently rehashed; only bash-n ran, no worker or CP reader started by this preparation.

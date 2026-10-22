# E1 low-LR joint then head-only continuation

Actual stage and metadata preparation passed; this controller did not submit a GPU job. Root alone runs the guarded submit command below. The prior completed B128 experiment is unchanged.

```sh
python3 reproduction/operations/qwen7b_warmup/deploy_mixed_direct_e1_lr_headonly.py --action submit --submit-authorized
```

The new namespace is `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-20261022-v1`. Source is the immutable E1 CP2577 (3,781,073,882 bytes, SHA `c0a1c7dc3663d4216966a3440c2103348bcc52db3cd04c2f5c59a784a2792917`), with the accepted raw/EMA/native Adam/head Adam/RNG/sampler state. CPU preparation read only checkpoint metadata/stat; the worker must freshly hash and restore the full payload in its GPU allocation.

E2 updates 2578–3898 train native and head at 5e-7. E3 updates 3899–5219 freeze native raw/native Adam and train head at 5e-5. EMA continues at .99 over the entire model. Global batch128/micro64, source sampler/order and inherited bank seal remain bound; no repeated 277GB seal scan. Native Adam ends3898, head Adam ends5219.

Runtime is the accepted original27 files plus worker/updater from `cdd7942f5727875a9fdc59f8d0f4af9419d689eb` and phase checkpoint helper from `d2acaf92f5d34e20eb05593cd1647e039a34ea73`. Controller published `8cf67d463421c889e8c89b01ee335c08bb1ce4de` with remote SHA verified. No extra review, fixture or GPU experiment was added.

Actual configuration: 22937 bytes, SHA `662f0931dabdcaace9d645d7c0fded786a9fe34cbe1421fdfd557b450803caf7`.
Actual source publication: 8762 bytes, SHA `c1826a6eabb497178feeffd4701c586b30ae61b5ff4450d46db87f86a300ab52`.
Actual Slurm entry: 1319 bytes, SHA `377a74269296ee0c95d64e86d55d90bcf810a851e23bfba52d9054060deb42b8` (normal/acd_u, 1H100,8CPU,128GiB,24h, no dependency).

`actual-stage.json` and `actual-prepare.json` record the two successful remote operations; `actual-config.json`, `actual-source-publication.json` and `actual-training.sbatch` are small exact metadata/code backups, not model state. No final4096 outcome was read.

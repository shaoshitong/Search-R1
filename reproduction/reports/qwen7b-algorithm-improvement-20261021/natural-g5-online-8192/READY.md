# Natural own-G5 paired online control ready

Source commit: `1fdb1755c699994ec9d8f7f0a7527957c3f58796`. New isolated suite: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-natural-g5-8192-20261022-v1`.

L21/P20 and L27/P20 each request2 GPUs and30 minutes. Global128, local64, micro8, accumulation8;3 epochs /192 updates; official fresh block/head, LR5e-6, EMA0.99. Original online prefill/native/head/runtime files are byte-identical to the deployed `5a3cfa1588c883d1c816d77ff95191acaa71eb35` control. The independent wrapper only redirects suite/fixture paths.

All8192 qids and all three original epoch orders are preserved. Each new epoch uses an independent seed derived from2026102207, selects five indices uniformly without consulting EM, then constructs the genuine G5 Mixed target and inverse-frequency class weights. Both layer variants share these exact plans. Original four token-shard references and exact fixed training2048 / independent selection2048 views are reused; only small metadata plans were created. No new rollouts, token shards or hidden caches; no final4096 data accessed.

|Epoch|Mixed|nonMixed|G10Mixed→G5 homogeneous|all-wrong|all-correct|
|---|---:|---:|---:|---:|---:|
|1|2638|5554|851|391|460|
|2|2709|5483|780|355|425|
|3|2664|5528|825|374|451|

Original conditional plans made all3489 G10-Mixed questions G5-Mixed each epoch. The natural control retains851 /780 /825 genuine homogeneous groups from those same questions. This supplies the previously removed hard negatives without duplicating questions. It is an intervention to test causality, not a demonstrated accuracy gain.

Root-only submit-once commands (cwd repository):

```sh
python3 -m reproduction.operations.qwen7b_warmup.deploy_natural_g5_online_control --action submit-once --commit 1fdb1755c699994ec9d8f7f0a7527957c3f58796 --layer 21
python3 -m reproduction.operations.qwen7b_warmup.deploy_natural_g5_online_control --action submit-once --commit 1fdb1755c699994ec9d8f7f0a7527957c3f58796 --layer 27
```

Default is normal/acd_u; root may explicitly set `--partition`, `--qos`, `--time-limit`. No dependency is added. Submission intent, same-name live jobs and existing job IDs prevent a blind duplicate. The deploying agent has not invoked submit-once.

Read-only observer:

```sh
python3 -m reproduction.operations.qwen7b_warmup.deploy_natural_g5_online_control --action status --commit 1fdb1755c699994ec9d8f7f0a7527957c3f58796
```

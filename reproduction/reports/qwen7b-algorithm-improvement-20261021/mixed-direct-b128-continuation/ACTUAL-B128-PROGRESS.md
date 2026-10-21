# Actual global128 continuation

Job746266 is genuinely training on ACD1-3. Source1675 became1677 then1687:12 complete updates; both native/head Adam1687. The full model/EMA/Adam/RNG and sampling position came from highest1675, not a rollback. Initial micro128 exhausted CUDA memory before either Adam; the same global128 batch retried micro64×2. Subsequent updates had no new OOM/failure. Full CP1676 was saved and CPUreloaded.

Eight owned-GPU samples:73467/81559MiB (71.745/79.647GiB,90.08%); mean sampled utilization52.125%, range0–100. Normal actual1677→1687 window10 updates/35.40297s:3.54030s/update,2169.31question visits/minute and21693.10 genuine native-row visits/minute. Excludes startup, first actual OOM and first checkpoint saving. Larger globalbatch does not imply a proportional throughput increase.

Latest worker40acb405, updater23000f23, controller659848a0; all pushed and remote verified. Original failed746263 had zero updates and is preserved in attempts/746263. Old746220 completed0:0 and must not restart. Global question batch128,512 balanced genuine G5 views,1280 native rows, two optimizer steps (one each) and one EMA update per accepted global batch. LR5e-6/EMA.99 unchanged. Four original epoch question visits retained with full128 rounding, scheduled global cursors2577/3898/5219/6540. Scientific Mixed goal remains active and final4096 untouched.

# Official direct Mixed CP128: early independent result

Job746154 completed0:0 in4m16s. All9 saved state hashes stayed unchanged; fresh full checkpoint SHA and actual two-Adam128/sampler128 restore passed on its owned GPU. The CPU audit independently rehashed all eight small prediction files, checked identical ownG5 IDs/EM across E0/raw/EMA, and recomputed every classification/ranking metric. No additional forward, threshold fit or final4096 access.

Fixed2048 has663 Mixed/1385 nonMixed questions. Primary threshold remains0.5.

|State|MixedBCE|BA|Mixed recall|nonMixed recall|AW false-Mixed|AC false-Mixed|AUC|Top512 Mixed|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E0 raw|0.772123|49.4109%|89.2911%|9.5307%|91.2796%|87.8788%|0.501164|164/512 (32.0312%)|
|CP128 raw|0.686178|50.9092%|42.6848%|59.1336%|39.2417%|46.0606%|0.529808|176/512 (34.3750%)|
|CP128 EMA|0.681879|49.3244%|27.6018%|71.0469%|29.8578%|26.0606%|0.513963|162/512 (31.6406%)|

Loss reduction mainly comes from nonMixed. AW BCE0.885493 becomes0.672129/raw or0.658812/EMA; AC0.846244 becomes0.683749/0.653443. Mixed BCE rises from0.554829 to0.709742/0.732736. The raw model has a small early ranking/BA gain; EMA has no balanced-accuracy or Top512 gain over initialization and misses480/663 Mixed questions. This128-update snapshot does not establish a successful trained discriminator. Main training remains unchanged.

|Fixed512 diagnostic|MixedBCE|BA|Mixed recall|nonMixed recall|AW false-Mixed|AC false-Mixed|AUC|
|---|---:|---:|---:|---:|---:|---:|---:|
|E0 raw|0.731487|48.3944%|71.7073%|25.0814%|73.6842%|78.4810%|0.491952|
|CP128 raw|0.692914|53.2915%|58.0488%|48.5342%|48.6842%|59.4937%|0.539342|
|CP128 EMA|0.685792|54.8526%|40.9756%|68.7296%|29.3860%|36.7089%|0.537833|

Snapshot complete34404B SHA dd717c8ed785aa74dfc03abd20fda898b8fe7d69b12053445882c849c5e95efe. Original E0 complete28272B SHA 32353ce8db3d8a514ca94b70201a1fe2e73f54593dbf785a8f9399853dbe29f5. Full references, confusion counts and selected K0..5 composition are in actual-readonly-audit.json. The512 diagnostic includes all512 questions, so its Top512 composition is the full diagnostic class distribution rather than a useful selection comparison.

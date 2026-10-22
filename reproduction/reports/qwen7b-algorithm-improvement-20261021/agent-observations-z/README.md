# Z749577 read-only observations

The unique replacement job **749577** is monitored independently from Root reports. The original 749570 failed CUDA initialization before any optimizer update; it is not treated as the active writer.

Two actual observations at 1792685063.466 and 1792685295.823 confirmed both ranks advancing from epoch 1/cursor 95 to cursor 407. Cursors 400–407 were consecutive finite updates with nonzero block and head gradients and identical global alignment statistics across ranks. The measured interval is 312 updates/232.36 seconds ≈0.745 sec/update (171.9 question groups/sec at global batch 128), excluding evaluation and checkpoint time.

At cursor 407, normalized MSE was 1.64180, anchor MSE on 4v was 0.16195 and total was 1.80375; prediction/target standard deviations on that optimizer batch were 0.08667/0.39797. These are training-batch measurements, not independent evaluation. They do not establish final quality or an absence of later collapse.

Two intervening control-read timeouts were recorded as unknown current state, not as a training failure. Subsequent reads recovered through the same adapter.

The temporary local watcher reads only controller/progress/curve/coverage/evaluation metadata and Slurm accounting for this job, with no remote mutation. It never opens checkpoint contents or the final4096 pool. Once training publishes complete.json and Slurm reports COMPLETED 0:0, the separately staged terminal CPU acceptor can verify immutable checkpoints. CPU loading does not establish GPU next-step consistency.

Local observations.jsonl is compact metadata; latest-actual.json preserves the most recent response. Full epoch evaluation snapshots are stored once per completed epoch.

# Z749577 read-only observations

The unique replacement job **749577** is monitored independently from Root reports. The original 749570 failed CUDA initialization before any optimizer update; it is not treated as the active writer.

Two actual observations at 1792685063.466 and 1792685295.823 confirmed both ranks advancing from epoch 1/cursor 95 to cursor 407. Cursors 400–407 were consecutive finite updates with nonzero block and head gradients and identical global alignment statistics across ranks. The measured interval is 312 updates/232.36 seconds ≈0.745 sec/update (171.9 question groups/sec at global batch 128), excluding evaluation and checkpoint time.

At cursor 407, normalized MSE was 1.64180, anchor MSE on 4v was 0.16195 and total was 1.80375; prediction/target standard deviations on that optimizer batch were 0.08667/0.39797. These are training-batch measurements, not independent evaluation. They do not establish final quality or an absence of later collapse.

Two intervening control-read timeouts were recorded as unknown current state, not as a training failure. Subsequent reads recovered through the same adapter.

The temporary local watcher reads only controller/progress/curve/coverage/evaluation metadata and Slurm accounting for this job, with no remote mutation. It never opens checkpoint contents or the final4096 pool. Once training publishes complete.json and Slurm reports COMPLETED 0:0, the separately staged terminal CPU acceptor can verify immutable checkpoints. CPU loading does not establish GPU next-step consistency.

Local observations.jsonl is compact metadata; latest-actual.json preserves the most recent response. Full epoch evaluation snapshots are stored once per completed epoch.

## Full epoch 1 and next-epoch progress

At 1792686105 both ranks had entered epoch2/cursor1420. Epoch1 covered169028 unique questions exactly once,1098580 genuine trajectory visits,1321optimizer updates including the68-question tail. The published3773206036-byte checkpoint SHA is `1be20daf9a2c7bde2a1ac5b8c264e33de07daf9e53cf7e8cbe696ea17a431f18`. It was not opened by this observer; terminal CPU loading remains pending. Full E1 evaluation took42.98seconds and all5 fixed-pool roles finished.

| E1 state | Fixed-train BA | Independent G5 BA | Independent v MSE | Independent Spearman | Independent Top512 Mixed |
|---|---:|---:|---:|---:|---:|
| raw |56.9870%|53.7839%|.00828889|.191058|218/512=42.5781%|
| EMA.9995 |57.3757%|57.2725%|.00883213|.156836|215/512=41.9922%|

The only predeclared G5 classification diagnostic uses predicted variance≥.08. The sigmoid output predicts4v, not Mixed probability. Selection ranks predicted variance only; the independent2048/ownfixedG5 pool contains663trueMixed. The train diagnostic is a fixed2048subset of already trained data.

Independent raw score std=.018014 versus true variance std=.092725; EMA score std=.008034. Scores are not literally constant, but spread remains markedly compressed. Raw MSE is about3.60% below the unattainable validation-oracle constant-mean baseline.008598; EMA MSE is still worse than that oracle baseline. This is an early evaluation, not an accepted final improvement or completed10epoch experiment.

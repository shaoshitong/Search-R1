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

## Epochs2–3 and continuing progress

Published immutable evaluation metadata records the following primary independent2048/G5 results. This is the same663-Mixed pool at every epoch. Top512 selection uses predicted variance descending; the predeclared BA diagnostic threshold remains0.08.

| Epoch | State | Train BA | Independent BA | Independent variance MSE | Independent rho | Independent Top512 Mixed |
|---:|---|---:|---:|---:|---:|---:|
|0|raw|50.000000%|50.000000%|0.01198474|0.041191|175/512=34.1797%|
|0|EMA|50.000000%|50.000000%|0.01198474|0.041191|175/512=34.1797%|
|1|raw|56.987030%|53.783916%|0.00828889|0.191058|218/512=42.5781%|
|1|EMA|57.375738%|57.272490%|0.00883213|0.156836|215/512=41.9922%|
|2|raw|60.528503%|55.575521%|0.00823818|0.203084|224/512=43.7500%|
|2|EMA|58.568267%|54.940077%|0.00834652|0.191491|223/512=43.5547%|
|3|raw|61.700450%|54.908876%|0.00820418|0.216648|217/512=42.3828%|
|3|EMA|60.036502%|55.176721%|0.00822664|0.210059|225/512=43.9453%|

E2 and E3 each covered169028 distinct questions exactly once with new label-blind M3..10/subset plans. Trajectory visits were1100933 and1098015; measured epoch durations were994.09 and985.52seconds. These counts are not evidence of new independent question data. E3 independent raw/EMA score standard deviations were0.02355/0.01659 versus true variance0.09273: neither score vector is literally constant, but both remain compressed. Raw Top512 decreased224→217 while EMA increased223→225; this does not establish a stable independent improvement. Fixed-train Top512 was340raw/335EMA, so the larger training gain does not transfer equally to the independent pool.

The observer subsequently confirmed epoch4/cursor4714 at1792688671 with no new errors. No checkpoint contents or final4096 data were opened; final10epoch completion and CPU state acceptance remain pending.

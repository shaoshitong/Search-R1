# Official 7B direct Mixed-only training: actual startup and speed

Job746083 runs on ACD1-25, normal/acd_u1H1008CPU128GiB. It starts only from official Qwen2.5-7B-Instruct block27 (a09a), with a new direct G5 Mixed head and both Adam0. LR5e-6, EMA0.99. No old probe checkpoint or optimizer is inherited.

All169028 realG10 question inputs/1690280rows/5356manifests passed the source-bound join and fresh reader SHA. CP0raw=EMA and bothAdam0 was independently saved/reloaded. Actual update1 changed both native block and head; complete CP1 independently reloaded; next update after durable restore matched bit-for-bit. At least14updates now accepted and later iterations continue without failure. Latest complete restore point at this measurement is CP7; ordinary rolling interval is128updates after the startup pins.

Each optimizer update:32questions,320own native trajectory rows,128realG5 views with64Mixed/64nonMixed. Updates8–14 averaged 6.481seconds of training computation per update (296.3questions/minute). The accepted-update wall interval8→14 measured290.9questions/minute, approximately2909trajectory-row visits/minute. This is a short actual startup sample, not a long-run guarantee; checkpoint/evaluation can lower whole-run throughput.

Initial independent2048test before any optimization: MixedBCE0.772123, BA49.4109%, AUC0.501164, Top512 trueMixed164/512. Raw and EMA match exactly. No trained-epoch independent result exists yet; no claim of algorithmic improvement.

Four obsolete terminal checkpoint paths were removed under human instruction;15,005,276,200logical bytes. Models/data/features/code/statistics and other threads were preserved. All training state remains remote.

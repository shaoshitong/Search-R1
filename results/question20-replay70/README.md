# Question plus20-token replay70

Latest human instruction replaces fresh-rollout1000 with 70 cached batches in original E1..70 order. Each batch has5120 G5 questions,25600 own-EM trajectories,globalBS256/local32/world8/noaccum,100 actual Adam updates per rank.

Source native3409 was explicitly stopped at completed E71,7100 Adam/rank. The source complete head/Adam/eight-rank RNG was independently SHA checked and CPU loaded. The learned block and20-token Linear coefficients/moments are preserved; new question coefficients/moments start at zero. Authorized target LR is3e-6 from source3e-5. New replay coordinates1..70 are separate from source epoch71; expected final Adam14100/rank.

Frozen official encoder layer27 post-block emits actual initial request and same-trajectory retained20 policy positions. The probe uses copied block35 and masked flatten[S*2048] to1 correctness logit. Fixed Qmax is scanned over all70 saved batches plus the fixed test actual prompts; no question truncation or retokenization.

No new rollout or retrieval server. One fixed512G5 test feature cache is encoded; each train batch is encoded then trained and its derived features pruned only after durable head completion. All tensors/checkpoints/raw data and full predictions remain remote. Only fixed-test G5 mixed accuracy is reported: q=1-prod(p)-prod(1-p),true label0<sum(strictEM)<5,threshold0.5.

Implementation compiled and13 focused CPU tests passed. Native3410 is RUNNING on h100-8 and has completed replay iteration1/70; all eight ranks made100 new Adam updates to7200 at actual LR3e-6. Iteration2 has started input preparation. Fixed Qmax is273, S is293, and the flattened dimension is600064. The fixed512-question G5 mixed-classification accuracy after iteration1 is56.4453125% (289/512). The training update loop took9.8364 seconds; this excludes feature encoding, preparation, and checkpoint saving and is not the full iteration runtime. Full70 iterations remain incomplete.

Training runtime is frozen at5534d253d9d1a94a17838b6f9f9c861536d0cdfb. Later control669097fe only adds an optional delta deployment path and does not hot-change runtime. The original gzip deployment was recovered through its completed remote receipt; the delta fallback was not executed. FIRST-ITERATION-SUMMARY.json is an exact copy of the small actual observation summary; model states and full predictions remain remote.

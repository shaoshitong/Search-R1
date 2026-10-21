# Direct-hidden count runner: prepared, no GPU submission

Immutable production source:501cf80478b77f68c3e13a96d4659e3da0971b2e, dedicated branch codex/direct-hidden-count-runner-20261021. Remote target algorithm-improvement-20261021/direct-hidden-count30016-train-v2; code-only stage and actual remote18-test CPU qualification passed (3.355s), followed by real metadata preparation. No official E1 tensor/GPU update was executed in preparation.

The exact frozen30016/B32×938 G10 plan is consumed from E1 cursor0. Inherited block27/trajectory head/raw/EMA/Adam10464/RNG are retained; a new masked prefix-token attention→128D embedding→symmetric G5 mean/variance/max→count6 MLP gets its own Adam0 and EMA. Objective is trajectoryBCE1+countNLL1; q_direct=sum count probabilities K1..4. Four real randomG5 views carry exact total question weight1; no adaptive or label-aware validation selection.

Independent joint checkpoint serialization checks all nine state hashes, native buffers, both clocks, plan/cursor and full original four-rank RNG archive before publishing latest. The GPU entry additionally repeats the first actual update from exactly restored old+new source states, saves CP1, and blocks further updates until an allocation-local independent CPU full-file SHA/reload accepts it. First six actual transactions remain an execution requirement, not a CPU qualification claim.

Full joint checkpoints precede fixed same-HF validation at256/512/768/938; validation restores buffers/RNG and verifies all nine training-state hashes unchanged. The new external read-only auditor validates own IDs/strictEM, six count probabilities and q_direct (not the old product-p formula), independently recomputes BA/recalls/AUC/Top512 and reads back the full joint checkpoint. It does not fit, tune or access final4096.

Prepared config16144B SHA3e910b12c0ff52eac70e61566c6135e2ec188bb47ec64ad79524abbd7a657526; sbatch1225B SHA62c6181bd2a2159ef46465b41b014df1700bdb2d645d3a681570cda947fa4858. CPU metadata accepted1004 L26/P20 manifests and300160 real trajectories. Local raw vectors, checkpoints, probabilities and coefficients are not retained.

The old v1 attempt only created upload metadata before a local bytes/str control parser failed; v1 partial staging is preserved. The parser was fixed and v2 staged immutably. Existing baseline job743517 was never patched, stopped, restarted or used as a new-head source.

Actual submission remains gated on parent GO after baseline938. Requested budget is one normal/acd_u H100,8CPU,64GB,2h, no Slurm dependency. New head512518 parameters add about8.2MB for FP32 raw/EMA/two moments; it uses the same two native-forward passes per trajectory. H100 throughput/peak memory are not yet measured; CPU toy timing is not GPU evidence.

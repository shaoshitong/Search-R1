# Exact G20 all15504 DP objective: CPU qualification and typed preparation

2026-10-21; runtime source commit `5c75a157803adb7696db2f7aec8d5c6e72917e55`, pushed and remote SHA verified. The old 32-pattern loss and immutable GPU jobs were not changed. No GPU was requested in this preparation.

The new loss is exactly

\[
\frac1Q\sum_i\left[\frac1G\sum_j\operatorname{BCE}(z_{ij},EM_{ij})
+\frac1{\binom G5}\sum_{S:|S|=5}-\log P(E_{i,S})\right].
\]

Each original question has total weight1. For each authentic five-subset, `E` is allwrong if its own EM sum is0, allcorrect if5, otherwise Mixed. The heterogeneous label of all20 is never substituted for a G5 label. The uniform mean of all G5 trajectory losses equals the source-G BCE analytically because each trajectory appears in exactly `C(G-1,4)` subsets.

The published three-state log-space DP retains allwrong, allcorrect and Mixed mass. Its probability recurrence is `W'=W(1-p)`, `C'=Cp`, `M'=M+Wp+C(1-p)`. This replaces 32-pattern expansion without changing the Bernoulli independence assumption, labels or inference score. Bounded chunks are immediately differentiated and discarded; the sole retained loss VJP has `Q*G` entries. No all15504-view graph or expanded pair matrix is retained.

Six local and six actual remote CPU tests passed: G10 all252 float64 value/gradient versus old32 and independent enumeration; G20 all15504 at logits±1000; permutation/chunk invariance; validation/unsupported-objective rejection; G20 toy Adam10464→10465→10466 with EMA, native buffers and RNG, actual first-update double replay and fullstate next-update restoration; exact source/plan preservation and a closed GPU authorization gate. These are CPU/toy tests, not native GPU qualification.

Actual login CPU benchmark used two pinned cores, two torch threads, identical synthetic float32 logits `[48,20]`, all744192 G5 views and seed9137. Each implementation ran in a fresh subprocess. `RLIMIT_DATA=3GiB`, a parent RSS watchdog and finite wall limits were active. Shared CUDA-library virtual mappings were recorded separately and were not presented as resident memory. CUDA was never initialized.

| Implementation | Chunk | Value+gradient wall | Peak RSS, max of child/watchdog | Speedup |
|---|---:|---:|---:|---:|
| Old 32 patterns |64|1.142455s|441.819MB|1×|
| Three-state DP |64|0.411415s|431.796MB|2.7769×|
| Three-state DP |256|0.299000s|427.696MB|3.8209×|

All variants enumerated exactly15504 subsets/question and had identical confusion counts. DP64 loss difference was0, max gradient difference `3.2014e-10`; DP256 loss difference `1.1921e-7`, max gradient difference `1.3970e-9`. DP256 held at most12288 G5 rows, no pattern expansion, and960 retained VJP entries. Total benchmark including interpreter imports was9.172s; owned wrapper exited0. Two real observations confirmed old32 complete/DP64 running, then all three complete. Independent final metadata/worker-gradient audit passed and the benchmark lock was free. These timings measure only the synthetic loss, not transformer IO, GPU training speed or model quality.

Typed preparation is at:

`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g20-exact-uniform15504-dp-preparation-v1`

Its config SHA is `815b6c14f1705aab5aecded0981ce5050c4a046bf4bb471c0f8087f229462da4`. The real743370 randomG20 config is pinned by SHA `428161412354177e509fe0336e1b5cf9eb7beb28cd58c0ce7c78c88335338a1c`; original B48×136 question order/plan SHA `9d2ec808aa4365d80c7c7031487e0e746ad2165ceb2bfd59214aa631ab587e7a`, authentic6528×20 IDs/ownEM bank, E1 raw/EMA/fullAdam10464/RNG, native features and exact same fixed-G5 validation refs are preserved. Bulk source tensor acceptance is explicitly inherited, not represented as freshly repeated. No checkpoint or feature tensors were opened by this preparation, and no final data was accessed.

The typed config has `explicit_launch_authorized=false`, `training_entrypoint=null` and `actual_GPU_qualification=false`. A separately published immutable GPU entry and explicit launch authorization are required for any subsequent actual experiment. Exact refs, checksums, source manifest and terminal evidence are in `actual-cpu-independent-audit.json`; raw synthetic gradients stay remote.

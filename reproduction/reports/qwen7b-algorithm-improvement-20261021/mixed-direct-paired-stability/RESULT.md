# Paired stability of official direct Mixed checkpoints

CP512 raw remains below the established Mixed balanced-accuracy frontier: **51.8199% versus E2 56.2505%**. The paired BA difference is **−4.4306 percentage points (95% interval −6.9933 to −1.7277)**. Its Top512 exceeds E4 by ten Mixed questions, but the paired interval spans −17 to +34. These results do not establish a joint improvement.

This is descriptive analysis of existing, already-viewed recipe-validation predictions, not a new model evaluation or an untouched generalization test. The original true fixed-G5 IDs and strict-EM labels match across all seven models; E0 raw and EMA are exactly identical. Pool2048 has663 Mixed. The source protocol is `1a0f5b411e9c31ff9a3b64a02aa9644156eee6c0`: seed9137, threshold0.5, four identity-only round-robin hash strata, and2000 paired question bootstrap replicates in each of five scopes. Selection is reranked independently in each resample.

| Model | Mixed BA % | Mixed recall % | nonMixed recall % | Top512 true Mixed |
|---|---:|---:|---:|---:|
| E0 | 49.4109 | 89.2911 | 9.5307 | 164/512 (32.0312%) |
| CP128_raw | 50.9092 | 42.6848 | 59.1336 | 176/512 (34.3750%) |
| CP128_EMA | 49.3244 | 27.6018 | 71.0469 | 162/512 (31.6406%) |
| CP512_raw | 51.8199 | 17.6471 | 85.9928 | 206/512 (40.2344%) |
| CP512_EMA | 51.0359 | 9.6531 | 92.4188 | 190/512 (37.1094%) |
| E2 | 56.2505 | 74.8115 | 37.6895 | 181/512 (35.3516%) |
| E4 | 55.7714 | 67.5716 | 43.9711 | 196/512 (38.2812%) |

Differences below are candidate minus baseline; BA units are percentage points. The intervals are paired percentile2.5/97.5 intervals, pointwise and without multiplicity correction. Decimal count endpoints reflect interpolation of bootstrap quantiles.

| Comparison | BA difference pp | 95% BA interval pp | Top512 Mixed difference | 95% count interval |
|---|---:|---:|---:|---:|
| CP128_raw − E0 | +1.4983 | [-1.1067, +4.1916] | +12 | [-11.000, +34.000] |
| CP128_EMA − E0 | -0.0865 | [-2.3258, +2.2235] | -2 | [-17.000, +16.000] |
| CP512_raw − E0 | +2.4090 | [+0.2314, +4.5854] | +42 | [+18.000, +66.025] |
| CP512_raw − CP128_raw | +0.9107 | [-1.6829, +3.6044] | +30 | [+7.000, +56.000] |
| CP512_EMA − E0 | +1.6250 | [-0.2370, +3.5514] | +26 | [+4.000, +52.000] |
| CP512_EMA − CP128_EMA | +1.7116 | [-0.4862, +3.9882] | +28 | [+8.000, +51.000] |
| CP512_raw − E2 | -4.4306 | [-6.9933, -1.7277] | +25 | [+0.000, +49.025] |
| CP512_raw − E4 | -3.9515 | [-6.7202, -1.1186] | +10 | [-17.000, +34.000] |
| CP512_EMA − E2 | -5.2146 | [-7.5929, -2.6844] | +9 | [-13.000, +35.000] |
| CP512_EMA − E4 | -4.7354 | [-7.2645, -2.1568] | -6 | [-31.000, +19.000] |

CP512 raw improves over E0 on BA by2.4090pp (interval0.2314 to4.5854) and selected-Mixed count by42 (18 to66.025). Relative to CP128 raw, the BA interval crosses zero, while selected-Mixed count increases by30 (7 to56). Against E2 and E4, both raw and EMA BA differences are negative throughout their pointwise intervals. The raw Top512 difference over E4 is uncertain; it is not evidence of stable superiority.

Each cell below reports **BA% / true-Mixed count in the stratum’s Top128**. Each stratum has512 questions. Stratum Top128 is a separate25% selection; its counts are not the global Top512 and need not sum to the global selection.

| Model | Hash stratum0 | Hash stratum1 | Hash stratum2 | Hash stratum3 |
|---|---:|---:|---:|---:|
| E0 | 51.1344 / 37 | 49.3229 / 47 | 50.5690 / 43 | 46.4440 / 36 |
| CP128_raw | 50.4438 / 42 | 48.8021 / 47 | 53.4536 / 47 | 50.8803 / 41 |
| CP128_EMA | 49.5451 / 36 | 48.8542 / 46 | 51.4585 / 43 | 47.4535 / 38 |
| CP512_raw | 50.6796 / 42 | 51.1458 / 56 | 52.2724 / 50 | 53.2128 / 56 |
| CP512_EMA | 51.1897 / 42 | 47.7604 / 57 | 53.1403 / 47 | 52.4435 / 50 |
| E2 | 57.3370 / 44 | 56.6667 / 52 | 58.0236 / 46 | 53.0190 / 40 |
| E4 | 58.3186 / 47 | 56.1979 / 48 | 56.7488 / 53 | 51.7029 / 46 |

CP512 raw Top128 is `[42,56,50,56]`; E4 is `[47,48,53,46]`. Raw wins in two strata and loses in two. CP512 raw BA is below E2 in three strata and slightly above in one. The full small aggregate retains per-stratum paired intervals for every fixed comparison; no model, checkpoint, threshold, recipe, or stratum was selected from these intervals.

The thin adapter reuses the original protocol’s identity partition, metric definitions, percentile interpolation and scope-specific RNG. It accepts direct group probabilities as direct probabilities; it does not invent trajectory p_j or impose the old product formula on the direct head. E2/E4 probabilities still pass the original independent-Bernoulli formula check. Shared paired indices are used for all seven models, so adding comparisons does not change any individual pair’s resamples.

Actual CPU execution:42.9665s, peak RSS35076KiB, affinity limited to two available CPUs and3GiB address-space cap. All five scopes used2000/2000 replicates, with no single-class skips. First observation saw full-pool bootstrap500→1000→1500; terminal observation saw2000 in all scopes and the complete aggregate. No torch/model/checkpoint loading, GPU submission, new forward, final4096 access, threshold fitting, bootstrap-driven training change, or mutation of running746266 occurred.

Protocol strata SHA256: `72c498c3fd8481c72ead7ee81d57b0fd6092cb7e5e0ff5b00f4e7a1eeaa5b69c`.

Remote complete: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-E0-CP128-CP512-paired-stability-v1/paired-stability.json`, 100845B, SHA256 `01a3c70621341e45a67d5be50e2200fa8ffeab7d0c808a8ca1981efb37f5df91`. Raw predictions and stratum IDs remain remote; the neighboring `actual-result.json` is aggregate-only. Source adapter/operator publication: `02f672b5016c3d2502765c7f68d9e4f0c7c7ca40`.

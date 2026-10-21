# Fixed2048 input ambiguity: existing direct-Mixed scores

CPU metadata audit: 32 sealed indexes / 10240 own trajectories / 2048 fixed questions, 1.873 s. All six whole-pool BA, recalls, AUC and Top512 counts independently match accepted E0/CP128/CP512 reports. No new forward, tensor/raw-rollout read, fitting or final-pool access.

165/2048 questions (8.06%) contain strict duplicate trajectory inputs; only 44/663 Mixed questions (6.64%) are in these groups. 21 Mixed questions contain at least one same-input pair with opposite EM, but no group has all five inputs identical. This establishes no group-level irreducible error bound.

Token-only and strict-feature unique-count strata happen to coincide for this pool; these remain separate definitions. Strict equality includes retained BF16 feature SHA, masks/lengths and causal positions. Nonmatching SHAs do not measure numerical or semantic distance.

| Strict unique inputs per own G5 | N | True Mixed | Mixed rate | K0..5 |
|---|---:|---:|---:|---|
| unique2 | 5 | 1 | 20.00% | 4,1,0,0,0,0 |
| unique3 | 20 | 9 | 45.00% | 7,1,1,3,4,4 |
| unique4 | 140 | 34 | 24.29% | 69,6,9,8,11,37 |
| unique5 | 1883 | 619 | 32.87% | 975,173,126,121,199,289 |

| Model | Whole-pool Mixed BA | Mixed recall | nonMixed recall | AUC | Global Top512 true Mixed |
|---|---:|---:|---:|---:|---:|
| E0_raw | 49.41% | 89.29% | 9.53% | 0.50116 | 164/512 |
| E0_EMA | 49.41% | 89.29% | 9.53% | 0.50116 | 164/512 |
| CP128_raw | 50.91% | 42.68% | 59.13% | 0.52981 | 176/512 |
| CP128_EMA | 49.32% | 27.60% | 71.05% | 0.51396 | 162/512 |
| CP512_raw | 51.82% | 17.65% | 85.99% | 0.57511 | 206/512 |
| CP512_EMA | 51.04% | 9.65% | 92.42% | 0.57059 | 190/512 |

| CP512 raw strict unique inputs | BA | Mixed recall | nonMixed recall | AUC | Global Top512 selected / Mixed | AW false positives | AC false positives | Mixed false negatives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unique2 | 50.00% | 0.00% | 100.00% | 0.25000 | 0 / 0 | 0 | 0 | 1 |
| unique3 | 66.67% | 33.33% | 100.00% | 0.63131 | 3 / 3 | 0 | 0 | 6 |
| unique4 | 57.63% | 29.41% | 85.85% | 0.65733 | 38 / 14 | 5 | 10 | 24 |
| unique5 | 51.32% | 16.80% | 85.84% | 0.57073 | 471 / 189 | 125 | 54 | 515 |

CP512 raw misses 515/619 Mixed questions with five distinct strict inputs. These account for 515/546 = 94.32% of all its Mixed false negatives; input equality does not explain the dominant failure. The five-distinct stratum itself remains near chance (BA 51.32%), despite AUC 0.57073.

All per-model/per-stratum BA, recalls, AUC, three-class error counts/rates, K0..5 and **membership in the original global Top512** are in FINAL.json and the remote receipt. The audit never selects a separate Top512 inside a stratum. The 5- and 20-question strata are descriptive small samples.

Remote complete: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fixed2048-input-ambiguity-v1/complete.json` (173837 B), SHA256 `10bf775f6a58e367ec3863bba9ee9ab7eb8b16e19bdd42c66ac19708eb9b0da4`.

Source: `cb2e3020099df43ce86f077351ee16d185ce070f`, `codex/fixed2048-input-ambiguity-20261022`.

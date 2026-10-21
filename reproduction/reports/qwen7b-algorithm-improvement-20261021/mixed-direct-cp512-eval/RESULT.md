# E0 → CP128 → CP512: direct Mixed evaluation

CP512 completed normally (`746247`, `COMPLETED 0:0`, 3m52s). Independent metric recomputation passed. On the fixed 2048-question G5 pool, raw Mixed balanced accuracy is **51.8199%** and EMA is **51.0359%** at the unchanged threshold 0.5. Ranking improves relative to E0/CP128, but Mixed recall remains low; this is not a satisfactory balanced classifier result.

The pool has 663 Mixed, 1055 all-wrong (AW), and 330 all-correct (AC) questions. Each snapshot uses exactly the same own five trajectory IDs and strict-EM labels. BA is `(Mixed recall + nonMixed recall)/2`. Top512 is selected by descending raw score, with qid tie-break; no threshold or model parameter was fitted by this audit.

| Snapshot | Mixed BA % | Mixed recall % | nonMixed recall % | AUC | Total binary BCE | Top512 true Mixed |
|---|---:|---:|---:|---:|---:|---:|
| E0 raw=EMA | 49.4109 | 89.2911 | 9.5307 | 0.50116416 | 0.77212275 | 164/512 (32.0312%) |
| CP128 raw | 50.9092 | 42.6848 | 59.1336 | 0.52980762 | 0.68617770 | 176/512 (34.3750%) |
| CP128 EMA | 49.3244 | 27.6018 | 71.0469 | 0.51396290 | 0.68187851 | 162/512 (31.6406%) |
| CP512 raw | 51.8199 | 17.6471 | 85.9928 | 0.57511094 | 0.66884276 | 206/512 (40.2344%) |
| CP512 EMA | 51.0359 | 9.6531 | 92.4188 | 0.57059205 | 0.66230044 | 190/512 (37.1094%) |

Total BCE and true-Mixed-class BCE are different measurements. At CP512 raw, total BCE is 0.66884276, while BCE restricted to the 663 true-Mixed questions is 0.74543749. Overall BCE falls while true-Mixed-class BCE rises from E0; the loss reduction does not establish improved balanced classification.

| Snapshot | AW → Mixed % | AC → Mixed % | True-Mixed BCE | AW BCE | AC BCE |
|---|---:|---:|---:|---:|---:|
| E0 raw=EMA | 91.2796 | 87.8788 | 0.55482914 | 0.88549303 | 0.84624397 |
| CP128 raw | 39.2417 | 46.0606 | 0.70974186 | 0.67212892 | 0.68374871 |
| CP128 EMA | 29.8578 | 26.0606 | 0.73273607 | 0.65881224 | 0.65344321 |
| CP512 raw | 12.3223 | 19.3939 | 0.74543749 | 0.62789914 | 0.64585247 |
| CP512 EMA | 7.2986 | 8.4848 | 0.77124625 | 0.60754900 | 0.61845709 |

CP512 raw confusion is TP117 / FN546 / FP194 / TN1191; EMA is TP64 / FN599 / FP105 / TN1280. Raw misses 546 of 663 Mixed questions; EMA misses 599. The selected raw Top512 contains AW217 / Mixed206 / AC89, with K0..5 counts `[217,50,50,39,67,89]`; EMA contains AW234 / Mixed190 / AC88, with counts `[234,52,43,38,57,88]`.

The separate diagnostic512 pool has 205 Mixed, 228 AW, and 79 AC. Its Top512 is the entire pool and is not a ranking selection metric.

| Snapshot | Diagnostic Mixed BA % | Mixed recall % | nonMixed recall % | AUC | Total BCE |
|---|---:|---:|---:|---:|---:|
| E0 raw=EMA | 48.3944 | 71.7073 | 25.0814 | 0.49195201 | 0.73148732 |
| CP128 raw | 53.2915 | 58.0488 | 48.5342 | 0.53934218 | 0.69291361 |
| CP128 EMA | 54.8526 | 40.9756 | 68.7296 | 0.53783268 | 0.68579226 |
| CP512 raw | 54.7621 | 52.1951 | 57.3290 | 0.56285056 | 0.68518635 |
| CP512 EMA | 53.3844 | 43.9024 | 62.8664 | 0.57004052 | 0.68129496 |

The GPU evaluation fresh-verified the full pinned CP512 SHA and official model files, restored all nine accepted components (including both Adam clocks and sampler at 512), and verified state unchanged after raw/EMA scoring. It did not construct the training feature reader. The independent stdlib CPU audit rehashed all twelve small prediction files and recomputed identities, labels, confusion, BA, recalls, FPRs, AUC, BCE, and Top512/K counts. It inherited the already accepted own-label provenance; it did not recompute strict EM from raw responses. CPU audit did not read/load checkpoint payloads, tensors, or models.

The original scientific configuration and fixed validation proof match across E0/CP128/CP512. The explicit fast-continuation transport/batching/binding migration is recorded, rather than described as an identical execution backend. No additional scoring, threshold tuning, bootstrap, actor update, or final4096 access was performed. Main training continues independently.

Fresh remote evidence:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-official-cp512-eval-v1/complete.json` | 34961 | `a621707be4be935ebda3984a38e66369ad80c3293b42754da359a1d732efae6e` |
| `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-official-cp512-eval-v1/E0-CP128-CP512-independent-audit.json` | 24262 | `445529fe00e9a81edaf4321a1ee12c0cd89f449025360519ad18191e49ef3366` |
| `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-official-cp512-eval-v1/source-config.json` | 14625 | `22098d5a4678db2f9c56d56cb5709829c3a4619c626e0b52d16ce726844402ac` |

The complete retained packet, prediction descriptors, all twelve metric summaries, observed progress/process identity, pin and source bindings are in the neighboring small JSON evidence files. Source audit origins: `1e95d531e7ecf077b665cf24ff39d9c8c262fdd2`; readonly helper identity fix `4d87835f5f0a2ee671c34fc0fbd6ea9147ab723c`. Runtime worker origin remains `3573e88b8d320baa3f5fb1f442787777f42591a5`; the main training runtime was not changed.

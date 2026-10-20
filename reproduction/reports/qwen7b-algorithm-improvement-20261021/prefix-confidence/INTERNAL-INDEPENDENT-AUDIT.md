# Prefix-confidence v2: independent internal terminal audit

Both actual CPU scalar fits completed normally (matched2:7LBFGSiterations/8closure evaluations; full8:26iterations/27evaluations). The independent audit accepted their complete small auxiliary checkpoint/state, input provenance and metrics. The internal gate is **negative**, so **external_opening_permitted=false** and no external prediction or untouched final pool was opened.

The holdout contains2048whole questions/own NEWSEED5 trajectories,655trueMixed(31.9824%). These questions were seen by the original probe warmup; this is a heldout split for the new scalar calibrator, not unseen-to-probe questions or a new external benchmark result.

| Model | Mixed BA | Mixed recall | NonMixed recall | AllWrong FPR | AllCorrect FPR | AUC | Top512 true Mixed |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Raw E4 | 64.0519% | 76.4885% | 51.6152% | 42.5000% | 68.6901% | 0.680658 | 238/512 |
| Clipped identity E4 | 64.0519% | 76.4885% | 51.6152% | 42.5000% | 68.6901% | 0.680658 | 238/512 |
| Matched2 control | 63.4370% | 87.1756% | 39.6985% | 51.2037% | 91.6933% | 0.670539 | 229/512 |
| Prefix full8 | 63.3607% | 87.0229% | 39.6985% | 51.2037% | 91.6933% | 0.670563 | 232/512 |

Full8 is **0.6912 percentage points lower in BA and six fewer Mixed Top512 questions than rawE4**. Relative to matched2, it adds three Mixed Top512 questions but lowers BA0.0763points. Thus it fails both the rawE4 joint-improvement gate and the no-BA-decrease requirement against matched2. This fixed experiment does not support promoting the six-prefix-statistic correction. It does not prove all possible prefix information is unhelpful.

The full8 Top512 K=0…5 histogram is **[158,46,35,54,97,122]**. Its high Mixed recall is accompanied by AllCorrect FPR91.69% and AllWrong FPR51.20%; NonMixed false positives remain the main classification issue.

## Independent acceptance performed

- Exact config`3ec15b790d5b0db67b88fdc22e74ea692eb216e34e0955853f4c273a30392371`, actual fit source publication`5a020457f2a0309afa6745946253b232b4aeb76d5400e51b3ae419069076d5d2`, protocola1ec8bbe/version2 and qualified math file SHA3244d6a0 were bound.
- Whole-qid SHA seed9137 split independently reproduced6144fit/2048holdout with zero overlap. Six-feature population mean/std/scale were independently recomputed from **fit rows only** using stdlibfsum; maximum difference from saved normalizer was8.881784197001252e-15.
- Fresh full-file SHA and actual CPU load/roundtrip of the28,598B matched2 and54,474B full8auxiliary checkpoints passed. Full LBFGS and all three auxiliary RNG restore interfaces passed; source/mode/phase/optimizer counters were verified. Matched2 retained zero prefix weights and no normalizer. Complete checkpoint/reload evidence precedes each internal prediction file.
- Every internal newseed5qid/owntrajectoryID/strictEM was rejoined to the accepted G10bank. Calibrated probabilities were independently recalculated from the saved8scalars and fit-onlynormalizer **in remote RAM** (maxp difference≤4.996e-16); q formula error≤2.915e-16. Metrics, classwise false positives, Top512 composition and pre-external gate were recomputed by an independent stdlib evaluator.
- Raw→canonical clipped identity has **0thresholdflips,512/512TopIDs in common, identicalTop512order,0changed rank positions across2048questions**. Both have zero tied score pairs; theTop512boundary score belongs to one question. Clipping is not counted as learned improvement.
- No original probe/model/Adam payload, GPU, external prediction or final pool was read. Original probe file metadata were unchanged. Real8scalars, normalizer values, LBFGS/RNGstate and per-question rows remain remote; this report includes only SHA/count/qualification/metric aggregates.

## Durable receipt

`fit/independent-internal-audit.json`,12,716B, SHA **f77c10b3868ad895c799574b06cd47191e4968fad8cf90f0d141860249b6cdd6**, under `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix-confidence-incremental-v1`.

[Small aggregate receipt](independent-internal-run.json); [separate immutable auditor stage](independent-internal-stage.json). Auditor/entry source commit`39d5af483d5d25871f500fd09987d2a9a5f69dbb`; three stdlib fixtures passed locally and onHPC3 before the actual audit, which exited0. No optimizer update/refit/threshold search was performed by the auditor. Existing training743517 was untouched.

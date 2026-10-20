# G20 matched pair: completed; joint performance gate failed

Independent audit epoch: `1792531833.8197708`. Both jobs completed normally: random `743370` in 45m52s, balanced `743371` in 48m49s. The read-only finite observer also exited 0.

Both branches inherited the same sealed E1 raw/EMA/block/readout/Adam/RNG state at Adam10464. They used the identical G10 question order: 6528 questions, B48 ×136 real updates, authentic20 trajectories per question, four unique G5 views each weighted1/4. The balanced branch enforced both view-count and weighted Mixed/nonMixed mass1:1. First actual update replay passed; every bounded second-forward logit check passed; final full checkpoints at cursor136/Adam10600 were saved and producer CPU-loaded/component-hash checked before evaluation. Actor updates were0.

The evaluation uses exactly the same precommitted label-blind native Prefix20/L26 G5 trajectories as the accepted E4 baseline. Their own IDs and strict EM were independently joined, every saved q was checked against `1-prod(p)-prod(1-p)` (max error <1.4e-7), and BA/AUC/ranking were independently recomputed. Natural Mixed prevalence is663/2048=32.373%. This is a viewed selection validation, not an untouched final test.

| Model | Mixed BA | Mixed recall | NonMixed recall | AUC | Top512 true Mixed | Allwrong FP | Allcorrect FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native E4 | 55.5291% | 66.3650% | 44.6931% | 0.576101 | 200/512 (39.0625%) | 49.8578% | 72.7273% |
| random | 55.4422% | 91.1011% | 19.7834% | 0.575250 | 201/512 (39.2578%) | 75.5450% | 95.1515% |
| global-weighted-balanced | 54.7227% | 92.9110% | 16.5343% | 0.580847 | 193/512 (37.6953%) | 79.2417% | 96.9697% |

Random gains one selected Mixed question but loses0.08685 BA percentage points against E4; balanced loses0.80642 BA points and seven selected Mixed questions, despite slightly higher AUC. Neither simultaneously improves BA and Top512 Mixed. The dominant error remains classifying nonMixed questions as Mixed, particularly allcorrect questions. This completed experiment does not satisfy the broader improvement goal.

Selected K=0..5 composition:
- `random`: `[184, 40, 38, 49, 74, 127]`.
- `global-weighted-balanced`: `[193, 46, 35, 41, 71, 126]`.

Both same-process CUDA peaks were6,655,523,328B allocated and8,887,730,176B reserved. A future isolated branch may qualify a larger microbatch for speed; this run used the predeclared micro5 and was not hot-edited.

Random and balanced are matched to each other in original question order, update budget and source state. Historical E4 is an identical-input performance reference, with no claim of equal historical training budget. G20 also uses twice G10’s real trajectory count; equal trajectory-compute is not claimed. No untouched final pool was opened.

Full prediction, checkpoint and source hashes are preserved in [the compact independent audit](actual-launches/g20-final-native-E4-independent-audit.json). Source runtime remains immutable `code-c1ea22d2ed4e` under `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g20-matched6528-v1`. No model or optimizer payload was returned to the Mac.

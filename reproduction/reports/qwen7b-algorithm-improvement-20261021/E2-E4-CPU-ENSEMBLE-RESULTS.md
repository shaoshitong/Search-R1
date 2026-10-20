# Fixed E2/E4 CPU validation combinations: all six failed the joint gate

The six-item grid was independently preregistered and published as `342cb4d6` before any combination result was computed. Runtime/test source was published as `75a9c4e9`. The actual CPU operation completed with exit0 at epoch1792533091; a separate bounded audit independently rejoined own EM/IDs, checked every score formula, and recomputed all group metrics/rankings. No fitting, threshold search, new GPU, Platt-grid modification or final4096 access occurred.

The pool is the previously viewed, tuning-allowed2048-question recipe-selection validation, with the identical qualified HF Prefix20/L26 features and fixed label-blind G5 for E2 and E4. It is not an untouched generalization test. Pool K=0..5 is `[1055,181,136,132,214,330]`, with663/2048 actual Mixed.

Fixed recipes: E2 weights0.25/0.50/0.75. `p-mix` interpolates each trajectory probability, then computes `q=1-prod(p)-prod(1-p)`. `q-mix` directly interpolates the original saved Mixed scores and is explicitly not a single independent-Bernoulli model. Classification stays at0.5 and Top512 orders by `(-score,str(qid))`.

The joint gate requires BA strictly above the exact E2 baseline `0.5625049686633887`, and Top512 actual Mixed strictly above E4’s196. No candidate passed; the preregistered selection rule therefore promotes none.

| Recipe / E2 weight | BA | TPR | TNR | Allwrong FP | Allcorrect FP | AUC | Top512 Mixed | Joint |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| E2 | 56.2505% | 74.811% | 37.690% | 56.398% | 81.212% | 0.569053 | 181 (35.352%) | — |
| E4 | 55.7714% | 67.572% | 43.971% | 51.185% | 71.515% | 0.576476 | 196 (38.281%) | — |
| p-mix / 0.25 | 55.9110% | 70.739% | 41.083% | 53.555% | 76.061% | 0.576234 | 193 (37.695%) | fail |
| p-mix / 0.50 | 56.1365% | 72.851% | 39.422% | 55.071% | 78.182% | 0.575376 | 187 (36.523%) | fail |
| p-mix / 0.75 | 56.3556% | 74.661% | 38.051% | 55.829% | 81.515% | 0.571874 | 189 (36.914%) | fail |
| q-mix / 0.25 | 56.0787% | 70.136% | 42.022% | 52.796% | 74.545% | 0.575739 | 195 (38.086%) | fail |
| q-mix / 0.50 | 55.9793% | 72.247% | 39.711% | 54.787% | 77.879% | 0.574816 | 187 (36.523%) | fail |
| q-mix / 0.75 | 56.3034% | 73.906% | 38.700% | 55.355% | 80.303% | 0.572536 | 183 (35.742%) | fail |

The best BA is p-mix0.75 at56.3556%, only0.1051 percentage points above E2, while Top512 drops to189. The closest selected-Mixed count is q-mix0.25 at195, still below E4 and with BA below E2. Neither is a promotion candidate.

Selected Top512 K=0..5 compositions:
- `trajectory_probability_mix-w25`: `[206, 42, 34, 45, 72, 113]`.
- `trajectory_probability_mix-w50`: `[212, 39, 34, 42, 72, 113]`.
- `trajectory_probability_mix-w75`: `[218, 37, 33, 46, 73, 105]`.
- `direct_Mixed_score_mix-w25`: `[209, 43, 32, 48, 72, 108]`.
- `direct_Mixed_score_mix-w50`: `[212, 40, 33, 42, 72, 113]`.
- `direct_Mixed_score_mix-w75`: `[216, 37, 30, 46, 70, 113]`.

At the fixed0.5 threshold E2/E4 disagree on309/2048 questions: E4 corrects174 E2 errors but creates135 errors on E2-correct questions;856 questions are wrong for both. This error complementarity did not translate into a joint gain within the six preregistered formulas. The result does not prove that every possible ensemble is unhelpful; this experiment does not extend or tune the grid.

The complete six prediction outputs, original prereg, runtime source and receipts are archived on HPC3 only: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/E2-E4-fixedG5-ensemble-validation-20261021/six-fixed-validation-results.tar` (uncompressed tar,778240B, SHA `73ed359590f1363ce5925574391f4c67acf151bff37ded4ffd36770edc5418d7`). No full probability rows were returned to the Mac.

Small evidence: [actual six-item results](actual-launches/e2-e4-fixed-cpu-actual.json), [independent audit and archive manifest](actual-launches/e2-e4-fixed-cpu-independent-archive.json), [published prereg](E2-E4-CPU-ENSEMBLE-PREREG.json). All six are validation candidates only, and all failed the declared strict joint gate.

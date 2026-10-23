# Full L15/P80 MSE: completed-epoch development Top512 composition

Read-only snapshot `1792713895.5239663`; aggregation `0.634s`. The unchanged published reader uses only atomically published completed evaluation JSON and closed dev2048 saved predictions. E0 is initial untrained evaluation, not a completed training epoch.

| seed / job | epoch | state | all wrong | Mixed | all correct | Mixed fraction | all-wrong share of remaining errors |
|---|---:|---|---:|---:|---:|---:|---:|
| seed1 / 750189 | 0 | raw | 280 | 175 | 57 | 34.18% | 83.09% |
| seed1 / 750189 | 0 | EMA | 280 | 175 | 57 | 34.18% | 83.09% |
| seed1 / 750189 | 1 | raw | 190 | 284 | 38 | 55.47% | 83.33% |
| seed1 / 750189 | 1 | EMA | 212 | 248 | 52 | 48.44% | 80.30% |
| seed1 / 750189 | 2 | raw | 187 | 310 | 15 | 60.55% | 92.57% |
| seed1 / 750189 | 2 | EMA | 201 | 274 | 37 | 53.52% | 84.45% |
| seed1 / 750189 | 3 | raw | 191 | 318 | 3 | 62.11% | 98.45% |
| seed1 / 750189 | 3 | EMA | 196 | 298 | 18 | 58.20% | 91.59% |
| seed1 / 750189 | 4 | raw | 197 | 312 | 3 | 60.94% | 98.50% |
| seed1 / 750189 | 4 | EMA | 190 | 313 | 9 | 61.13% | 95.48% |
| seed1 / 750189 | 5 | raw | 193 | 314 | 5 | 61.33% | 97.47% |
| seed1 / 750189 | 5 | EMA | 195 | 312 | 5 | 60.94% | 97.50% |
| seed1 / 750189 | 6 | raw | 185 | 320 | 7 | 62.50% | 96.35% |
| seed1 / 750189 | 6 | EMA | 193 | 315 | 4 | 61.52% | 97.97% |
| seed1 / 750189 | 7 | raw | 184 | 320 | 8 | 62.50% | 95.83% |
| seed1 / 750189 | 7 | EMA | 192 | 316 | 4 | 61.72% | 97.96% |
| seed1 / 750189 | 8 | raw | 187 | 318 | 7 | 62.11% | 96.39% |
| seed1 / 750189 | 8 | EMA | 188 | 320 | 4 | 62.50% | 97.92% |
| seed1 / 750189 | 9 | raw | 176 | 329 | 7 | 64.26% | 96.17% |
| seed1 / 750189 | 9 | EMA | 178 | 326 | 8 | 63.67% | 95.70% |
| seed2 / 750190 | 0 | raw | 232 | 155 | 125 | 30.27% | 64.99% |
| seed2 / 750190 | 0 | EMA | 232 | 155 | 125 | 30.27% | 64.99% |
| seed2 / 750190 | 1 | raw | 202 | 286 | 24 | 55.86% | 89.38% |
| seed2 / 750190 | 1 | EMA | 186 | 249 | 77 | 48.63% | 70.72% |
| seed2 / 750190 | 2 | raw | 196 | 310 | 6 | 60.55% | 97.03% |
| seed2 / 750190 | 2 | EMA | 196 | 281 | 35 | 54.88% | 84.85% |
| seed2 / 750190 | 3 | raw | 192 | 317 | 3 | 61.91% | 98.46% |
| seed2 / 750190 | 3 | EMA | 195 | 304 | 13 | 59.38% | 93.75% |
| seed2 / 750190 | 4 | raw | 189 | 318 | 5 | 62.11% | 97.42% |
| seed2 / 750190 | 4 | EMA | 191 | 317 | 4 | 61.91% | 97.95% |
| seed2 / 750190 | 5 | raw | 190 | 320 | 2 | 62.50% | 98.96% |
| seed2 / 750190 | 5 | EMA | 190 | 319 | 3 | 62.30% | 98.45% |
| seed2 / 750190 | 6 | raw | 182 | 325 | 5 | 63.48% | 97.33% |
| seed2 / 750190 | 6 | EMA | 183 | 325 | 4 | 63.48% | 97.86% |
| seed2 / 750190 | 7 | raw | 185 | 318 | 9 | 62.11% | 95.36% |
| seed2 / 750190 | 7 | EMA | 180 | 328 | 4 | 64.06% | 97.83% |
| seed2 / 750190 | 8 | raw | 178 | 322 | 12 | 62.89% | 93.68% |
| seed2 / 750190 | 8 | EMA | 179 | 328 | 5 | 64.06% | 97.28% |
| seed2 / 750190 | 9 | raw | 185 | 321 | 6 | 62.70% | 96.86% |
| seed2 / 750190 | 9 | EMA | 175 | 332 | 5 | 64.84% | 97.22% |
| seed2 / 750190 | 10 | raw | 187 | 317 | 8 | 61.91% | 95.90% |
| seed2 / 750190 | 10 | EMA | 172 | 331 | 9 | 64.65% | 95.03% |

All42 raw/EMA rows exactly reproduce their evaluation Top512 Mixed counts. Both seeds and all epochs match the same2048 NQ question IDs, ordered own five trajectory IDs and strict EM; identity SHA `0520e04dd7fc6e8eb9f66865299d6345c013ca4222141f794255bc57d9954914`. The unchanged pool contains663 Mixed/1055 all-wrong/330 all-correct. Ranking uses saved scores only, with qid tie-break, never labels or a classification threshold.

Seed1 E3→E4 raw changes from191 all-wrong /318 Mixed /3 all-correct to197 /312 /3. The net change is **+6 all-wrong, −6 Mixed, 0 all-correct**; Mixed@512 falls62.11%→60.94% (−1.171875 percentage points). This is a net class-composition change, not a claim that exactly six individual questions were swapped. The existing separate evaluation reports a higher fixed-threshold BA at E4; Top512 ranking and threshold classification measure different behavior.

At the same E4, seed1 raw/EMA are197/312/3 and190/313/9, while seed2 raw/EMA are189/318/5 and191/317/4 (all-wrong/Mixed/all-correct). Remaining raw nonMixed selections are98.50% all-wrong for seed1 and97.42% for seed2. Seed2 E5 raw/EMA are190/320/2 and190/319/3; these later seed2 figures do not select the epoch/state.

Seed2 E6 raw/EMA both select325/512 Mixed (63.4765625%); their compositions are182/325/5 and183/325/4 (all-wrong/Mixed/all-correct). All rows continue to match the original evaluation counts and the unchanged fixed ownG5 identity. This update does not select an epoch/state.

Seed1 E5 raw selects314/512 Mixed (61.328125%), composition193/314/5; EMA selects312/512 (60.9375%), composition195/312/5 (all-wrong/Mixed/all-correct). These are copied existing fixed-development outputs, not a new selection or inference.

Seed2 E7 Top512 compositions (all-wrong/Mixed/all-correct): raw 185/318/9, EMA 180/328/4. Saved evaluation counts and fixed ownG5 identities match. No epoch/state was selected.

Seed1 E6 Top512 compositions (all-wrong/Mixed/all-correct): raw 185/320/7, EMA 193/315/4. Saved evaluation counts and fixed ownG5 identities match. No epoch/state was selected.

Seed2 E8 Top512 compositions (all-wrong/Mixed/all-correct): raw 178/322/12, EMA 179/328/5. Saved evaluation counts and fixed ownG5 identities match. No epoch/state was selected.

Seed1 E7 Top512 compositions (all-wrong/Mixed/all-correct): raw 184/320/8, EMA 192/316/4. Saved evaluation counts and fixed ownG5 identities match. No epoch/state was selected.

Seed2 E9 Top512 compositions (all-wrong/Mixed/all-correct): raw 185/321/6, EMA 175/332/5. Saved evaluation counts and fixed ownG5 identities match. No epoch/state was selected.

Seed1 E8 Top512 compositions (all-wrong/Mixed/all-correct): raw 187/318/7, EMA 188/320/4. Saved evaluation counts and fixed ownG5 identities match. No epoch/state was selected.

Seed2 E10 Top512 compositions (all-wrong/Mixed/all-correct): raw 187/317/8, EMA 172/331/9. Saved evaluation counts and fixed ownG5 identities match. Actual750190 terminal COMPLETED0:0/all10E13210 is separately recorded in the latest diagnostics table; independent CP acceptance is not implied by this metadata aggregation. No epoch/state was selected.

Seed1 E9 Top512 compositions (all-wrong/Mixed/all-correct): raw 176/329/7, EMA 178/326/8. Saved evaluation counts and fixed ownG5 identities match. No epoch/state was selected.

This is repeatedly viewed NQ development, not final4096. No checkpoint/model tensor, final pool, new inference or Slurm command was used, no per-question predictions were exported to Mac, and no remote/training state was changed. The prospective full-budget seed1-only selection rule is unchanged; no epoch/state or best seed was selected. All K=0..5 histograms and source evaluation/prediction proof hashes remain in [actual-summary.json](actual-summary.json).

Unchanged read script source `62de4f53975cb3aa3fd296abb4bc12593dc3b8d5`, SHA `e2122619efc791e33c9a380a4dc8f19a18f21047aebd6f3ee424c1109a813aa1`. Completed evaluation and prediction artifacts were SHA-checked before and after aggregation.

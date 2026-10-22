# Full E1 continuation: E2 joint, E3 head-only — actual result

Job749015 completed0:0. Observed epoch1792683345.0397263. All values below use fixed question/G5 pools; online batches are not substituted.

| Pool | Epoch/state | Mixed BA | BCE | AUC | Mixed recall | NonMixed FPR | AW FPR | AC FPR | Top512 Mixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|fixed_train2048|E1 EMA|58.99%|0.671798|62.71%|62.29%|44.31%|40.00%|56.91%|344 (67.19%)|
|fixed_train2048|E1 raw|58.87%|0.671493|62.84%|55.64%|37.89%|33.75%|50.00%|349 (68.16%)|
|independent2048|E1 EMA|52.88%|0.653016|59.50%|21.87%|16.10%|13.93%|23.03%|205 (40.04%)|
|independent2048|E1 raw|52.28%|0.643807|59.28%|13.88%|9.31%|8.53%|11.82%|201 (39.26%)|
|fixed_train2048|E2 EMA|59.69%|0.662616|63.67%|64.42%|45.03%|40.69%|57.72%|350 (68.36%)|
|fixed_train2048|E2 raw|59.18%|0.661705|63.67%|66.08%|47.72%|44.17%|58.13%|356 (69.53%)|
|independent2048|E2 EMA|53.57%|0.636835|59.79%|28.81%|21.66%|20.09%|26.67%|201 (39.26%)|
|independent2048|E2 raw|53.75%|0.640283|59.77%|30.32%|22.82%|21.33%|27.58%|200 (39.06%)|
|fixed_train2048|E3 EMA|59.98%|0.662745|63.84%|64.79%|44.82%|40.28%|58.13%|352 (68.75%)|
|fixed_train2048|E3 raw|60.00%|0.662476|63.84%|65.34%|45.34%|40.83%|58.54%|352 (68.75%)|
|independent2048|E3 EMA|53.97%|0.636255|59.87%|30.32%|22.38%|20.85%|27.27%|201 (39.26%)|
|independent2048|E3 raw|54.20%|0.636744|59.90%|30.92%|22.53%|21.04%|27.27%|203 (39.65%)|

E2 and E3 each visited all169028 distinct questions once, included the68-question tail, and used84514 authenticMixed plus84514 authenticnonMixed G5 views. Both used5e-7. Native Adam stopped at3898; head ended5219. Raw native block frozen hash15fc2bfdf2e10028f94d34fd6f89c9fbbec6262df2ad82e0dd7a945566a7fcdd matched actual accepted checkpoints. Raw/EMA and bothAdam/RNG/sampler/source CPU reload were accepted; GPU next-update equivalence is not claimed.

Final checkpoint: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-e1-unique-g5-20261022-v1/checkpoints/cursor-005219.pt`, bytes3782361818, SHA`f67fe1f79cfb61908a5b6fc84f99ccd6f462ed3e5f4a4819355aae715180ad4a`.

Interpretation: E3 EMA independent BA53.9670% is above E1 EMA52.8846%, but Top512 remains201/51239.2578%, below E1 EMA205/51240.0391%. No stable ranking improvement is established. Untouched final confirmation has not been scored; the whole scientific goal remains active.

Fixed-train and independent pools have different class prevalences, so raw accuracy/Top fractions between pools alone are not evidence of overfit. Compare each model across epochs on each unchanged pool.

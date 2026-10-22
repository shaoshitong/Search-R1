# 7B Mixed-only E1 continuation and matched feature exploration

Human corrections supersede old hard-replay recipe. Main job749015 resumes accepted E1 raw and EMA independently, two Adam and RNG intact; gradients update raw and then EMA(.99). No EMA-to-raw substitution. Both new E2/E3 LR5e-7; E2 native block+head; E3 raw native frozen/head only.

Each new epoch visits169028 questions once, one authentic G5 from own10,84514 Mixed/84514 nonMixed and final68 tail. New per-epoch seed is derived from inherited sampler entropy plus seed2026102202 and epoch. Full question SET repeats across epochs intentionally; order and authentic five-trajectory combination are randomized. Finite random sampling can coincidentally choose the same subset. CPU real-metadata two-epoch audit uses an independent audit RNG, so its99.5362% changedG5 is algorithm evidence, not actual production epoch history.

Fixed training2048 is sampled from E1-seen questions with label-blind G5 seed2026102201; same inputs across E1/E2/E3 and raw/EMA. Independent512/2048 pools remain unchanged, original E1EMA reference52.8846%BA/Top205of51240.0391%. Online training batch statistics are reported separately. Final4096 pool remains untouched for frozen-candidate confirmation.

Parallel pilots compare block27/P20,block21/P20,block27/P80 on the same8192 real G10 training questions and fixed independent2048. They use fresh official matching block plus identical seed direct Mixed head,3epochs. Latest human two-GPU authorization replaces only PENDING4GPU pilots749049/50/51 with749077/78/79:2GPUeach/acd_ue, global128/local64/micro8/accum8 instead of4,12CPU128G3h. All original real inputs reused. Frozen official actor produces features online; only selected native block and Mixed head receive gradients. This pilot is a matched feature comparison, not a substitute for full169028 training nor an E1 continuation.

Prior747723 hard-replay E2/E3 results are preserved historical measurements. They cannot establish a causal learning-rate conclusion or be presented as results of the new unique-G5 recipe.

Scientific completion requires stable independently reproducible Mixed BA or Top512 gain and untouched independent confirmation. Submission/RUNNING/lower trainloss/terminal0:0 alone do not satisfy the goal.

Actual main progress:749015 real new29updates,3712 distinct questions; source E1 baseline matched exactly in independent2048, first new CP2578 fullSHA/CPUreload accepted. New E2/E3 and pilot scientific outcomes remain pending.

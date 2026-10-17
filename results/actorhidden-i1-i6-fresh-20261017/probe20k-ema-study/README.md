# Fixed actor50: larger dataset, EMA, mixed loss and hidden mixup

This is a local probe study. Formal actor training remains human-stopped.

The human explicitly restored question+20 direct readout and chose hidden+true soft-label mixup. Actor0050 stays frozen. The shared dataset comprises20,480 distinct questionsG5 (102,400 actual trajectories): reuse the existing5,120G5 and generate only15,360 additional different questionsG5 with the same actor. The original same-actor512G5 heldout remains fixed and excluded by question ID and normalized content. Features are the sampling actor's actual layer27 post-block question+20 native hidden, with original masks and positions; no independent encoder or future answer input.

All three arms load the same authentic joint50 head40/Adam200, preserve source moments/RNG, override LR to1e-5, and train ten epochs. Each epoch naturally uses all102,400 trajectories exactly once in completeG5 groups,400 synchronized Adam writes per rank, average global trajectory batch256 (255/260), no accumulation. Each arm ends at4,200 Adam writes/rank,4,000 new. EMA decay0.99 starts from the source raw weights and updates after every real optimizer write. Raw and EMA are separately evaluated on the same complete real train and heldout rows, after updates. Ordinary/balanced mixed accuracy, both recalls/confusion matrices, trajectory accuracy/BCE and groupBCE are recorded. EMA is evaluation only; gradients/Adam continue on raw parameters.

|Arm|Trajectory BCE|Group mixed BCE|Augmentation|
|---|---:|---:|---|
|data_ema|1|1|none|
|data_ema_group2|1|2|none|
|data_ema_mixup|1|1|compatible hidden G5 mixup, Beta(0.2,0.2)|

The mixed-event probability is `q=1-prod(p)-prod(1-p)`, own trajectory targets are strict EM, and true mixed is `0<K<5`. The loss arm changes only group coefficient; Pearson/DIST is not combined into this first ablation. For fair comparison, every evaluation also records the unweighted common1+1 loss; weighted totals from different arms are not directly comparable.

All arms share the same label-blind, reshuffled, complete-G5 step/rank plans. Mixup pairs are restricted to identical attention-mask/position/segment layouts and stay on one rank/step. Hidden, ownEM targets and mixed-event targets are convex-mixed; the mixed-event target is mixed independently from the two original binary group labels. It is never derived from a fractional ownEM sum. Integer token IDs are never interpolated. Compatible pairs exchange symmetrically, preserving batch label mass; unpaired questions remain natural. Actual augmented coverage is logged, rather than claiming all samples were mixed.

Three arm phases run sequentially in a single eight-rank allocation, loading the same source for each arm. The data-only job and the fit job have separate immutable code deployments. No actor/reward/PPO or formal200 resumes are performed. All checkpoints, native hidden, original/full rollouts and predictions remain remote; this directory holds only source/configuration, small aggregates and final scientific figures.

CPU fixtures verify EMA direction, separate convex targets, unchanged default BCE, differentiability and complete dataset coverage, without a GPU forward. These do not block or modify the healthy data-generation job. `cpu-fixtures.json` contains the actual CPU result.

Data3468 frozen runtime84167eb2, data base `/mnt/data/shaoshitong/search-g1-prefix-selection200-probe20k-dataset-20261017-0bdddc20840b-r48`. Fit has not yet been submitted at this publication. Deployment/submit controls use unique intent/receipts and must not be replayed.

## Data initialization recovery

Native3468 completed block52 (25,600 new continuations) and then actually failed at block53 engine initialization. Rank5 reported zeroKV cache blocks. The installed vLLM0.6.3 computes peak memory from the change in free GPU memory and explicitly assumes other processes do not change their usage while profiling. Concurrent preserved video occupied approximately50GiB/card in the terminal sample and varied in the next sample. The failure is not a probe/actor update failure. Original inode logs and103 owned process births have been retained and actual nativeFAILED1:0/owned cleanup/cgroupgone observed.

Recovery data namespace appends`-resume52`, refers to completed block52 in its original location, preserves exact source questions/IDs/actorversion/seeds and generates only the remaining two blocks. All eight inference workers use the already-working rank0 resource profile: utilization0.16,256 sequences,8192 batched tokens,eager. This changes concurrency/cache resources, not question count, G5 sampling parameters or token budgets. No video is stopped. The separate fit dataset pointer follows the recovery manifest. The original3468 intent is never replayed.

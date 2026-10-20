# Native Prefix20 confidence: preregistered matched CPU information test

Full extraction accepted all8,192 fixed original NQ training questions and81,920 authentic original5+newseed5 trajectories. Every native chosen-token logprob is aligned through actual native chunk→turn→response policy-mask positions and the exact sealed Prefix20 tokens, positions and causal contexts. No future answer, post-cutoff policy or environment placeholder enters the six observables. Extraction took258.190s and peak128MiB. The existing E4 source model/EMA/Adam/RNG were never loaded or modified by this test.

The qid-only split was frozen before this fit:6,144 fit and2,048 internal holdout questions, seed9137. Assessment uses NEWSEED5 only. These question texts were previously seen by the native probe warmup; this is an auxiliary-calibrator question split, not unseen-probe performance.

Before any realfit, protocolv2 fixed a matched2-scalar control and a full8-scalar model, each starting from identity, identical E4p_j, data split, uniform252 genuineG5 subsets, trajectoryBCE+three-eventNLL1:1, regularizer.01, optimizerbudget40iter/60closure and one common26min limit. Only full8 could advance, requiring strictBA+Top512 improvement over rawE4 plus neither metric decreasing versus matched2 and at least one strictly improving. No grid, threshold fitting or later refit was permitted.

Both realfits completed in18.590s with process exit0. The saved actual closure journals show count/loss changes, not merely a launched process. Complete raw scalar/LBFGS/RNG/source/mode/phase state and, for full8, the fit-only normalizer were saved and immediately reloaded/restored before any internal assessment. Real learned scalars, optimizer histories, RNG, normalizers and all per-question feature/p/q records remain exclusively remote.

| Internal NEWSEED5,2048q | BA | Mixed recall | nonMixed recall | AllWrong FPR | AllCorrect FPR | AUC | Mixed@512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen raw E4 | 64.05188% | 76.48855% | 51.61522% | 42.50000% | 68.69010% | 0.680658 | 238/512 |
| Matched2 control | 63.43703% | 87.17557% | 39.69849% | 51.20370% | 91.69329% | 0.670539 | 229/512 |
| Prefix full8 | 63.36070% | 87.02290% | 39.69849% | 51.20370% | 91.69329% | 0.670563 | 232/512 |

The error direction is increased false Mixed predictions: full8 raises Mixed recall from76.4885% to87.0229%, but nonMixed recall falls51.6152%→39.6985%. AllCorrect false-positive rate rises68.6901%→91.6933%, and AllWrong false-positive rate42.5%→51.2037%. Relative to rawE4, that is69 more true-Mixed positives alongside166 more nonMixed false positives on this fixed2048 pool.

Full8 did not improve either BA or Top512 over rawE4; its BA was also below matched2. The registered jointgate is false. The E4 external prediction payload was not opened for this experiment and no final pool was accessed. This is a completed negative result for this one additive eight-scalar recipe, not evidence that every possible actor-confidence method is ineffective.

`matched_platt2`: 8 closure calls /7 optimizer iterations, regularized fit loss 2.066953251→1.934598002; CP `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix-confidence-incremental-v1/fit/aux-matched_platt2-complete.pt`, 28598B, SHA256`5a529527573bee098536df213082eae1cadbc51f6f8686f534431a2f80553805`.

`prefix_full8`: 27 closure calls /26 optimizer iterations, regularized fit loss 2.066953251→1.934353805; CP `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix-confidence-incremental-v1/fit/aux-prefix_full8-complete.pt`, 54474B, SHA256`afcf5760dd60acb81763e1d9968defbba04cc9acd4720a3670a97f663fefdd3e`.

The six inputs are native chosen-logp mean, population std, minimum, first-five mean, last-five mean and actual kept-policy length/20; all reductions exclude padding. Genuine policy logprob0 is legal, so causal-native source agreement and mask1 distinguish it from environment placeholders. All29,422 actual zero policy values were retained (~1.796% of1,638,399 policy tokens); no trajectory was allzero and no question was removed. The new immutable projector also explicitly permits an allzero vector if the native causal checks prove it real. The active completed4c extraction was never hot-edited.

| Quality, per seed | Old original5 | Newseed5 |
|---|---:|---:|
| Mean chosen-logp | -0.468380244 | -0.464886318 |
| Population std of per-trajectory mean | 0.265099648 | 0.264409866 |
| Real policy zero values | 14632 | 14790 |
| Constant-logprob trajectories | 0 | 0 |

The same3,665 identical-causal-input pairs identified by the prior P20 study were checked during this extraction. Their recorded chosen-logprobs have between-run numeric differences in2,078 pairs: overall mean absolute token difference0.00502753, maximum0.37605798; across original/newseed events the mean is0.00885592. This does not create a different semantic token/context input or remove the505 opposite-EM semantic collisions. Native actor logprob is an additional numeric observable with batch/run variation, not a full-vocabulary entropy estimate; no entropy or new correctness label was fabricated.

Data complete:`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix-confidence-incremental-v1/data/complete.json`,72920B,SHA256`ccc6d0c1686aa11e52496dc2879c26f3047748f0f09803228a19749fe42f8d7a`.

Feature metadata:`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix-confidence-incremental-v1/data/features.jsonl.gz`,4026019B,SHA256`1c96d5fb2b520826f5627395e0a3aa7b1dc88e39b3336cb4bd76a06bd3b7a814`.

Internal complete:`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix-confidence-incremental-v1/fit/internal-complete.json`,22149B,SHA256`45de5e01ff1802f27acba1c25d7393865e281e8c53da673f24a2de73312086ed`.

Terminal complete:`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix-confidence-incremental-v1/fit/complete.json`,22149B,SHA256`45de5e01ff1802f27acba1c25d7393865e281e8c53da673f24a2de73312086ed`.

Actual runtime sourcee11fe2b1cc01977dcb0bde57490967e2445ce98f; qualified math9932cee9e04eebb4e617ed5ae5f938452ff36264; prereg-v2a1ec8bbe44a24deb9ada77b59f3edf57cdc653e2. Independent terminal audit PASSED: completeauxCP/LBFGS/RNG reload, fit-only normalizer (maximum independent fsummismatch8.88e-15), exactownnew5 IDs/EM, CP-before-assessment chronology, p/q and all registered metrics. Raw-vs-canonicalclip causes0thresholdflips, all512TopIDs and order remain identical, no changed ranks and no score-tie pairs. Audit `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix-confidence-incremental-v1/fit/independent-internal-audit.json`,12716B,SHA256`f77c10b3868ad895c799574b06cd47191e4968fad8cf90f0d141860249b6cdd6`; external_opening_permitted=false. This closes only this registered negative hypothesis; it does not complete the broader probe-improvement goal. No new GPU, rollout or change to the ongoing Prefix20 append-only probe training was made.

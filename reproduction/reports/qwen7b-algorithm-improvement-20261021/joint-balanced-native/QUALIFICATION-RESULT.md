# Native joint qualification: actual seven accepted updates

Job **745177** finished `COMPLETED 0:0` in 7m46s. Immutable runtime source was `919a277f94f12bb8e2d22b9bec056c0169617c7e`; the source E4 checkpoint remained read-only. This is a finite engineering qualification on the accepted58432-question bank, **not124194 main training or an external accuracy result**.

- The native block27 and trajectory readout both changed, with finite nonzero gradients. First-update raw/EMA/Adam/native-buffer/RNG plus sampler replay matched exactly.
- The actual CP1 was independently SHA-checked and six-state loaded on CPU **inside ACD1-13 allocation745177**. Its restored sampler next batch matched exactly. The durable CP1 next-update replay also matched exactly.
- All seven updates completed, Adam14418→14425. CP7 is 3,748,140,634 bytes, SHA `c93c5eafb8393efc80bc615b91adf87cd24591d9dd11c3504a6282d36a1082cc`. The final checkpoint's full SHA/component reload is producer evidence; the terminal external audit read metadata/stat only, with no login tensor load.
- Every batch had32 distinct questions,128 real G5 views, exactly64Mixed/64nonMixed, per-question view weights summing to1. Four draws can repeat a subset; the sampler reports this explicitly. Only trajectoryBCE1+binaryMixedBCE1 contributes to training; other logged diagnostic components have zero weight.
- Two earlier immutable attempts failed before any accepted update:745147 omitted exclusions in a lookahead sampler binding;745155 omitted `import time` in the new update adapter. Both failures and their original source/output directories remain preserved. The fixes were separately published; neither failure discarded an accepted model update.

`actual-terminal-qualification.json` binds the exact source/config/first-replay/CPU-reload/next-replay/final receipts and verifies the exact main Slurm ID plus a released training lock. `actual-v3-stdlib-global-check.json` records the narrow unbound-global check after the time-import mistake.

The independent main runner preserves E4 Adam14418 and the same qualified micro5 two-pass update. It adds124194 input gates, rolling full checkpoints, epoch0–4 fixed raw/EMA evaluations, and a normal24h allocation. It will return WAIT while any full-cohort input is absent; no main GPU allocation is claimed here. Former Count5120 is training data, never an unseen evaluation pool.

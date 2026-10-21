# Actual124194 native joint main: startup qualified

Job **745501** remains RUNNING on ACD1-28 under immutable source `3b56e583efe2469ac9108873c31de7144a5201e6`. Latest accepted cursor is **62/15528**, Adam **14480**, at epoch1792587547.8640. Actual progress advanced from16 to54 to62; no failure receipt was present. This records startup qualification, not completion or improved external accuracy.

- The actual source is E4 full raw/EMA/Adam14418/native buffers and historical four-rank RNG; current execution is one GPU. Block27 and trajectory readout both changed at accepted step1. Actor updates remain0.
- First accepted update double replay passed all six state hashes and sampler state. Independent allocation CPU CP1 full-file SHA and six-state/sampler reload passed. Durable CP1→step2 replay also passed, followed by actual accepted steps3–8 and continued progress to54.
- Each observed batch has32 source questions,128 real G5 views,64 Mixed/64 nonMixed, per-view weight0.25 and per-question total1. Views use their own trajectory EM; repeated subset draws are reported.
- CP1 SHA `8e5c69e8b995d86e47964f89a586f4fe92cde728683ff5f6d90d9d2dbc72db5d`; independent receipt SHA `80909ed909d535170c5a89685ab7c24a764cda9beb2148eaf3142dd508c79d75`.
- Durable-next-update receipt SHA `4f9e7378d00ac58ff73620d96bea580a565c2e107b4020f2266aa944d1310187`. Latest complete rolling CP7 SHA `1b97d3cb18754b6ce0970c805ec4308ae1bce696d40f911eb493f444b2bd5def`,3,750,884,762B; producer six-state CPU/sampler reload passed. Later regular saves are every128 updates plus epoch pins/USR1. This does not claim every accepted step is already durable.

The already-completed epoch0 selection2048 evaluation gives the following starting baseline. Its input has663 Mixed questions out of2048; decision threshold is0.5 and Top512 selection is label-blind. No new forward or threshold fit was performed for this summary.

| State | Mixed BA | Mixed recall | NonMixed recall | AUC | Top512 real Mixed | TP / FN / TN / FP | AW / AC false-positive rate |
|---|---:|---:|---:|---:|---:|---|---|
| raw | 55.2138% | 66.8175% | 43.6101% | 0.570032 | 190/512 (37.1094%) | 443 / 220 / 604 / 781 | 50.5213% / 75.1515% |
| EMA | 55.7714% | 67.5716% | 43.9711% | 0.576477 | 196/512 (38.2812%) | 448 / 215 / 609 / 776 | 51.1848% / 71.5152% |

Epoch0 complete receipt: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1/evaluations/epoch-00/complete.json`,23930B SHA `be5ff3311fa5bbfe23bf2052fd2645c695232acee2ddcd1936497f46d52ba5d7`. Training six components and RNG/native buffers were restored unchanged after evaluation. Former internal5120 is part of training and was not treated as heldout; final4096 was not scored.

All model/checkpoint tensors stayed on HPC3. Independent real checkpoint tensor reload ran in the owned allocation, and this final summary used metadata/receipts only. Full4epochs, epoch1–4 external validation, final resource closure and any eventual scientific improvement remain outstanding.

The observed steps54 and62 took6.738s and6.741s. At an unchanged6.74s/update,15528 updates alone project to about29.1h, before epoch evaluation/checkpoint overhead; the actual allocation limit is24h. This is an early estimate, not a guaranteed runtime. The current writer remains untouched; any required continuation must use the highest verified complete transaction and sampler state, with no silent rollback.

# B arm: CP50 prefix and two-stage selection continuation

Human instruction, 2026-10-18: stop B only, resume full CP50. A remains on its original frozen runtime.

| Actor outer | Native policy prefix | Ordinary candidate questions | Predicted Top continued | Actual actor groups |
|---|---:|---:|---:|---:|
| 51–100 | 20 | 5120 | 768 | 512 |
| 101–150 | 40 | 5120 | 1024 | 512 |
| 151–200 | 80 | 5120 | 1536 | 512 |

After continuing the exact original G5 prefixes, use the referenced thread's final ranking: actual mixed groups first, then original predicted mixing probability descending and original ordinal ascending; fill any mixed shortfall with nonmixed groups in the same predicted ordering. Labels are used only after all expanded-set continuations complete. Report this as outcome-assisted selection, not pure probe accuracy. All selected tokens and original sampled probabilities are projected without regeneration.

Existing B periodic10 rules remain: 2560 full G5, balanced replacement, one pass, 50 Adam writes when both classes exist. Ordinary updates use only final selected512 G5, balanced replacement and10 writes. Raw head trains; EMA0.99 scores. LR5e-6 and trajectory BCE +5×mixed BCE remain unchanged. Actor and grounding/reward/QA mathematics are unchanged. No new warmup or actor reset. Native20/40/80 post-block27 hidden capture retains question plus generated positions; a lookahead forward processes the last retained token, but its sampled token is discarded. Short terminal prefixes use mask-zero padding. Tool context retains its original positions.

Readout migration preserves the copied Transformer block, old Linear coordinates, raw/EMA optimizer moments, Adam steps, all rank RNG and balancing state; new Linear coordinates/moments are zero. Migration is not an optimizer update. Initial CP50 actor1000/rank and probe700/rank are independently verified; discarded51+ results remain at the stopped source and are not stitched into this continuation. Each migration has a separate receipt.

All model/optimizer/hidden/rollout/prediction payloads stay remote. Publication, prepare and submit have separate immutable evidence and once-intent controls. Only submitted/running is not training acceptance; follow actual continuation and real updates.

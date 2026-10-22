# B128 four-epoch training fit and sampling-target diagnosis

This is a CPU read of the completed746266 curve and four frozen evaluation receipts. It is not a new training-set evaluation. Online training metrics use the changing **raw** model and changing sampled G5 views; EMA validation metrics use the single saved checkpoint. No model, checkpoint payload, feature or prediction rows were loaded.

## Immediate E1 answer

| Scope | Mixed BCE | Overall accuracy | Mixed BA | Mixed TPR | nonMixed TNR | nonMixed FPR | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|---|
| Online raw1676–2577 (902 B128 updates) | 0.684927 | 55.09% | 55.09% | 58.18% | 52.00% | 48.00% | 134347 / 120063 / 110849 / 96565 |
| Online raw last100 (2478–2577) | 0.683794 | 55.44% | 55.44% | 58.73% | 52.14% | 47.86% | 15035 / 13349 / 12251 / 10565 |
| Frozen E1EMA fixed2048 | 0.653016 | 63.82% | 52.88% | 21.87% | 83.90% | 16.10% | 145 / 1162 / 223 / 518 |

E1 training online also misclassifies many nonMixed views as Mixed (~48% FPR), but it does not predict everything Mixed:53.09% of all online views,53.29% in the last100, were predicted Mixed, with no all-Mixed batch. Its902-update window contains461824 correlated G5 views from115456 question occurrences, not461824 independent questions. The earlier1675 B32 updates are excluded from this requested B128 window. No fixed E1EMA training-set accuracy has been measured here. The40.04% user figure is **205 true Mixed among globally selected512**, not whole-pool classification accuracy.

## Online raw fit

| Epoch (B128 interval) | BCE all / last100 | BA all / last100 | TPR all / last100 | TNR all / last100 | Predicted Mixed all |
|---|---:|---:|---:|---:|---:|
| epoch1 (1676–2577) | 0.684927 / 0.683794 | 55.09% / 55.44% | 58.18% / 58.73% | 52.00% / 52.14% | 53.09% |
| epoch2 (2578–3898) | 0.600563 / 0.592029 | 69.54% / 70.75% | 69.42% / 69.61% | 69.66% / 71.88% | 49.88% |
| epoch3 (3899–5219) | 0.634189 / 0.616312 | 65.75% / 68.03% | 67.56% / 69.79% | 63.95% / 66.27% | 51.81% |
| epoch4 (5220–6540) | 0.668585 / 0.664893 | 59.97% / 60.09% | 57.93% / 59.33% | 62.00% / 60.84% | 47.97% |

BA and accuracy coincide online because every stored update has256 positive-role and256 negative-role views. Both are recomputed from summed TP/TN/FP/FN, not from rounded per-step metrics. The complete4865 accepted updates are unique, ordered and gap-free; two Adam clocks match each cursor.

| Epoch | Native gradient L2 mean | Head gradient L2 mean | Pre-clip total norm mean / p95 / max | Clipped updates |
|---|---:|---:|---:|---:|
| epoch1 | 0.2281 | 0.3575 | 0.4247 / 0.6970 / 1.2381 | 4/902 |
| epoch2 | 1.1051 | 1.2660 | 1.6911 / 3.8194 / 7.1148 | 905/1321 |
| epoch3 | 0.4974 | 0.7155 | 0.8756 / 1.6879 / 4.8225 | 358/1321 |
| epoch4 | 0.3531 | 0.4622 | 0.5855 / 0.9588 / 2.9187 | 55/1321 |

Full gradient distributions and first100/last100 windows are retained in the aggregate JSON. E2 has much more gradient clipping and improved online fit. At the E3 sampling boundary, first100 BCE is0.7282 versus0.5920 in E2 last100; this is a changed hard-sampling distribution and changing model, so it is not a fixed-dataset forgetting measurement. All reported values are finite. E3 contains five all-Mixed training batches, all in its first100; the other epochs contain none. This is a transient event, not the persistent overall training behaviour.

## Same fixed2048 checkpoint evaluations

| Epoch/model | Mixed BCE | Mixed BA | Accuracy | TPR | TNR | AUC | GlobalTop512 Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| epoch1 raw | 0.643807 | 52.28% | 65.82% | 13.88% | 90.69% | 0.59277 | 201/512 |
| epoch1 EMA | 0.653016 | 52.88% | 63.82% | 21.87% | 83.90% | 0.59501 | 205/512 |
| epoch2 raw | 1.035228 | 48.25% | 34.52% | 87.18% | 9.31% | 0.45952 | 156/512 |
| epoch2 EMA | 0.967913 | 48.07% | 33.64% | 88.99% | 7.15% | 0.47498 | 158/512 |
| epoch3 raw | 0.624133 | 50.99% | 66.85% | 6.03% | 95.96% | 0.57706 | 199/512 |
| epoch3 EMA | 0.623642 | 51.26% | 67.63% | 4.83% | 97.69% | 0.57829 | 205/512 |
| epoch4 raw | 0.802757 | 52.01% | 36.52% | 95.93% | 8.09% | 0.57035 | 198/512 |
| epoch4 EMA | 0.768949 | 53.38% | 38.43% | 95.78% | 10.97% | 0.56652 | 191/512 |

Training is not uniformly unable to fit: E2 last100 reachesBA70.75%, yet its frozenEMA fixed2048 isBA48.07% withTPR88.99%/TNR7.15% (92.85% nonMixed FPR). E3 swings toward nonMixed (TPR4.83%/TNR97.69%); E4EMA swings back toward Mixed (TPR95.78%/TNR10.97%). Thus the problem includes unstable fixed-pool discrimination and threshold behaviour, not simply persistent all-Mixed training predictions. However these online sampled raw metrics do not prove full fitting of the natural169028 training distribution or isolate classic overfitting.

## Actual G10→G5 role target mismatch

The deployed sampler uniformly chooses positive-role qids amongK10=1..9, uniformly chooses negative-role qids among **all** G10 questions, then uniformly chooses a true-class-conditional five-subset. Four draws are with replacement and average to question-occurrence weight1. This is not natural uniformG5 sampling conditioned on its class.

For a natural uniform qid followed by uniform one-of252 G5: `pMix(K)=1-[C(K,5)+C(10-K,5)]/252`. Conditioning on a positive class requires qid probability proportional to pMix, and on a negative class proportional to 1−pMix. K10=5 has only2 nonMixed subsets: negative probability1/126.

| Scope | K5 share of current negative-role qids | K5 share under natural class-conditional negative | Current / target |
|---|---:|---:|---:|
| nq | 3.34470% | 0.03943% | 84.83x |
| hotpotqa | 5.64460% | 0.08281% | 68.16x |
| pooled | 4.57143% | 0.06020% | 75.94x |

The pooled75.94x is a normalized **K5 sampling-frequency ratio**, not an error ratio; these qids still comprise only4.57% of the current uniform negative-role pool. Each K5 negative occurrence draws four times from only two valid whole-G5 combinations, so repeated views are unavoidable. Full K0..10 role distributions and positive/negative ratios are in the aggregate.

After epoch0, half the slots in each role request the previous epoch observed top-quarter hardest qids (mean four-view training BCE); the other half remain uniform cycles. If every role qid was observed and rejections are negligible, a hard-member qid receives approximately2.5x uniform mass and a nonhard qid0.5x, a5x hard/nonhard ratio. In general a hard member has multiplier `.5+.5*P/H`. Actual hard-member lists, sizes and K/domain composition are not available as small metadata and were not extracted from GB checkpoints; finite-batch duplicate rejection/fallback also changes the simple approximation.

The earlier uniform-domain calculation (NQ Mixed-role prior45.80%, Hotpot53.18%) demonstrates a possible shortcut, but cannot alone explain the E2/E4 near-all-Mixed and E3 near-all-nonMixed validation swings. Neither the sampling mismatch nor hard mining is proven to be the sole cause.

## Minimal data-only correction candidate (not implemented or launched)

Freeze a natural-class-conditional G5 target: use the same predeclared domain share for each role (for example canonical NQ46.6615%/Hotpot53.3385%), positive qid weights proportional to pMix inside that domain, negative qid weights proportional to 1−pMix, then retain the existing uniform authentic conditionalG5 draw, four-view average and global50:50 class mass. This removes the role/domain shortcut and the K5 rare-negative over-weighting by construction, without changing model/data/labels. A clean target-distribution control should initially omit hard mining; if hard proposals are retained, exact target semantics require explicit target/proposal importance weighting, otherwise it is a different objective. This is a proposed sampler correction requiring a separate authorized comparison, not a validated winner or a change to any existing run.

Source `929d1c88f0c4c4ae5ac5468ee5345c645f91fec4`; fixed configSHA `e5f0027dfcff4b6db442b38e9b2452e86cc2c9adcabd97a194971dc4d8f8d837`. Curve8355088B SHA `d9f9c824782721108cb19ae1f104d7cd802a09fbdfd4a23ed95ea5e225b97eff`. A second freshSHA readback of the curve, config, maincomplete and four epochcomplete files is identical. All actual aggregate output is small; no predictions/model/features/final4096 were read and no newGPU/scoring/training was run.

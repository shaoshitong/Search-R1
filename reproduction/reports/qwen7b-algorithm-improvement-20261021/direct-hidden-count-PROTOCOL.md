# Preregistered direct-hidden G5 count readout — prepare only

Frozen before any real direct-count fit or new external candidate scoring. This is an independent probe-only branch; current append job743517, peer production, actor weights and foreign tasks remain unchanged. Submission is not authorized by this document. The root will first use the final938 append evidence: a genuine joint winner takes priority for frozen confirmation; otherwise a CPU-qualified direct-count implementation may proceed. No Slurm dependency on success/failure of unrelated experiments will be added.

## Matched data, start and budget

Use the exact30016-question frozen append plan SHA `8d6f5b046c8ee459c529f0750c80bac63455b2663f19cdcdce48d0fddd648cbf`, containing old256 B32 batches and682 new B32 batches. Start from the same archived E1 native block27/trajectory head raw/EMA/full Adam10464/RNG used by the peer8192 baseline, not from a post-validation chosen model. Train938 question batches in the exact frozen order and actual four unique G5 views per question, total weight1 per original G10 question. Old Adam must progress10464→11402. Existing legacy parameters and Adam moments are inherited; a new head has explicitly new parameters, separate Adam0→938 and EMA. Its initialization must preserve the original inherited RNG state. Keep the existing learning rate and per-question weighting; do not add negative ranking, statistics or actor-logprob branches from failed controls.

## Model and loss

The legacy block receives the same detached actual L26 causal question+Prefix20 features, masks, positions and backend. It returns the unchanged scalar trajectory logits and masked Prefix20 hidden vectors. A new trainable attention pool over actual policy-prefix tokens, followed by a128-dimensional projection, yields one embedding per trajectory. Five embeddings are aggregated with symmetric coordinatewise mean, population variance and max, then a small MLP produces six logits for the true strict-EM countK=0,1,2,3,4,5. Padding contributes no evidence; short and zero-prefix behavior is explicit and finite. No future tokens, answer identity, correctness labels or question ordinal enter the head's input.

For each real five-view, loss is mean-five trajectory BCE at weight1 plus six-class count NLL at weight1, then view-weighted and averaged across original questions. The direct Mixed score is `P(K=1)+P(K=2)+P(K=3)+P(K=4)`. It does not use `1-product(p)-product(1-p)` as its group decision. This is distinct from the already negative linear head over scalar probability summaries. All real features/model/Adam/RNG/probability payloads remain remote.

## Minimal execution qualification

CPU synthetic tests cover output geometry/masks, G5 permutation invariance, own-ID/EM/count and per-question weights, exact legacy output/state inheritance, independent new-head Adam, same next-update/RNG restore, and failing a replay mismatch before either Adam writes. A real owned-allocation first step must verify deterministic replay of both optimizers and full checkpoint reload before acceptance. Observe at least five consecutive complete updates and the next update. Preserve all failures and immutable attempts; no hotpatch or optimizer reset.

Use one normal-priority H100, no emergency queue, no actor training. Reuse the existing accepted feature bank; no new rollout budget is claimed. Expected memory/step costs are hypotheses until actually measured. Retain native two-forward bounded microbatch behavior; do not keep an entire native-transformer graph for the full B32/G10 batch.

## Validation and success

Use identical fixed-HF2048 actual G5 identities/backend and own strict EM as the accepted baselines (663 Mixed). Primary group threshold is0.5, selection is global Top512 with deterministic ID tie-break, fixed before fitting. Report BA, Mixed/nonMixed recalls, allwrong/allcorrect false-Mixed rates, AUC and selected K0..5 composition. Predeclare only checkpoints256/512/768/938, no adaptive external screening, new threshold fitting or extra candidate grid. Report the entire learning curve; do not call fit-loss reduction a success.

A success must jointly exceed the stronger fair BA frontier E2=0.5625049686633887 and E4 Top196/512=38.28125%, with paired original-question/subset stability. These references are same-HF; native-backend E4 Top200 is a separate comparison. A jointly improved frozen candidate may access the untouched4096 confirmation questions once; there is no confirmation access after a negative gate. The goal remains active after scientific-negative outcomes, with no claim that Prefix20 uniqueness guarantees learnability.

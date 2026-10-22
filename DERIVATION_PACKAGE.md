# Derivation Package

Updated: 2026-10-23 Asia/Shanghai. The historical2026-10-16 trajectory-BCE note is preserved in reproduction/reports/qwen7b-algorithm-improvement-20261021/group-topk-alignment/DERIVATION-20261016-ARCHIVE.md. It is not the current objective.

## Target

Explain when a group-level loss can control the number of true Mixed questions selected by Top-K. Derive a margin objective that stops contributing gradients once sufficient ordering is achieved. Extend the analysis to the human-proposed window:64 current question groups with gradients and1954 historical detached scores, total2018, selecting500. The established development/confirmation evaluation remains2048-select512.

The experiment geometry is Qwen2.5-7B, block15 and real Prefix80. The actor is frozen; only the probe block and group head train. No trajectory BCE is introduced.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.

A deterministic upper bound on empirical Top-K regret is available. This is not a guarantee of a strong measured correlation between training loss and future Top512 outcomes, nor of monotonic SGD improvement. Stale-score windows add a quantifiable score-drift term; small-batch and unseen-question generalization remain empirical requirements.

## Invariant Object

For one pool of N distinct question groups, define

\[
 y_i=\mathbf1\{0<\sum_{j=1}^{G_i}EM_{ij}<G_i\},\quad
 P@K(s)=\frac1K\sum_{i\in S_K(s)}y_i .
\]

S_K uses descending model score with a deterministic label-independent tie rule. Each label belongs to the actual sampled own trajectories, not the wholeG10. The invariant quantity is the lost true-Mixed count relative to the best possible sizeK selection:

\[
 M=\sum_i y_i,\quad b=\min(K,M),\quad
 r(s)=b-\sum_{i\in S_K(s)}y_i,\quad
 R_K(s)=r(s)/K .
\]

When M<K, optimal precision is M/K, not100%.

## Assumptions

- N is finite,1<=K<=N; the scores are finite and gamma>0.
- The pool and its actual binary own-group EM labels are fixed while evaluating a loss/bound.
- Scores and model inputs do not contain labels, full future rollouts or label-derived visible features. The prescribed prefix is retained and masked/padded consistently, even when a naturally short generation ends before80.
- Exact finite-pool statements use the complete stated score vector. A batch128/K32 is not the future2048/K512 pool.
- The current warmup actor is frozen. Historical trajectory labels therefore remain valid for their saved ownG5; this does not automatically justify reusing old-actor labels during later changing-student RL.
- Conditional expected-utility statements use true conditional probabilities. Training losses do not establish that these probabilities or a population risk are known.

## Notation

- s_i: real group-ranking logit; not a trajectory probability.
- v_i: observed population variance of the actual binaryEM group; v_i=p_i(1-p_i), p_i=mean_j EM_ij.
- C*_K(s): maximum score sum over sizeK sets containing exactlyb positives.
- L_K: normalized K-specific structured margin surrogate.
- tilde s: vector combining historical detached scores and current differentiable scores.
- delta: maximum absolute discrepancy between tilde s and all-fresh scores from the current model on that exact window.
- B: number of current groups; cache rows do not receive gradients.

## Derivation Strategy

Define the real selection utility first. Separate exact observed variance, conditional mean variance and Mixed probability. Construct the loss-augmented Top-K objective and prove its empirical bound. Then detach historical score entries and state precisely which guarantee survives.

## Derivation Map

1. Identity: G5 observed Mixed iff its variance is positive.
2. Counterexample: lower variance MSE need not improve empirical precision.
3. Population distinction: conditional mean variance and conditional Mixed probability can rank groups differently.
4. Proposition: the K-specific surrogate upper bounds empirical precision regret.
5. Proposition: all-pair hinge gives another, looser Top-K regret bound.
6. Interpretation: margin satisfaction removes that loss gradient; actual Adam/EMA skipping is separately implemented.
7. Proposition: a detached historical window has the same bound for its mixed score vector, with a drift correction for all-fresh current scores.
8. Approximation: current-batch gradient scaling and partial gradients are training heuristics under staleness, not an unbiased/full-pool gradient theorem.

## Main Derivation

### Step1. Why continuous variance is a proxy rather than the final metric

For G5, v is0,0.16 or0.24. Both positive values count equally as Mixed in P@K.

An exact empirical counterexample, all predictions in[0,0.25]:

\[
 v=(.16,.24,0),\quad y=(1,1,0),\quad K=1.
\]

Prediction a=(.16,0,0) has MSE=.0192 and selects a Mixed group. Prediction b=(.07,.08,.09) has smaller MSE=.0139333 but selects the nonMixed group. Thus MSE reduction is not a monotonic guarantee on Top-K.

There is still a conditional MSE bound: if positive target variance is at least d>0 and negative variance is0, pairr disjoint missed positives withr selected negatives. A reversed pair requires squared errors totaling at least d^2/2. Therefore

\[
 R_K(s)\le \frac{2N\,MSE_v}{K d^2}.
\]

For G5,d=.16. AtN2048,K512 and MSE around.008, this bound exceeds1 and is uninformative. Exact fitting would order positives above negatives; current small average losses do not prove near-perfect selection.

A stronger distinction concerns new data: the MSE-optimal predictor is E[v|x], whereas expected Mixed selection wants eta(x)=P(Y=1|x). For example:

- TypeA: Mixed probability.7; when Mixed, v=.16. Then E[v|A]=.112.
- TypeB: Mixed probability.5; when Mixed, v=.24. Then E[v|B]=.120.

Perfect mean-variance regression prefersB, despiteA having a higher Mixed chance. Additional distributional assumptions would be needed to rule out this reversal.

For contrast, binary groupBCE targets eta directly. On a fixed feature pool, if E_i is excess conditional natural-logBCE, a Pinsker/Cauchy argument gives expected precision regret at most sqrt(sum_i E_i/K). This uses true conditional risk, not observed trainingBCE, and provides no automatic neural-network generalization guarantee.

### Step2. K-specific margin construction and exact empirical bound

Define the best score sum among label-optimal selections:

\[
 C_K^*(s)=
 \sum\operatorname{Top}_{b}\{s_i:y_i=1\}
 +\sum\operatorname{Top}_{K-b}\{s_i:y_i=0\}.
\]

Empty sums are0. Define

\[
 L_K(s,y)=\frac{
 \gamma b+\sum\operatorname{Top}_K(s-\gamma y)-C_K^*(s)
 }{\gamma K}.
\]

The numerator is nonnegative: substitute a label-optimal set into the first maximum. ReLU is mathematically redundant and selects zero derivative at an exact zero in the implementation.

LetS be the actual scoreTopK andt its positive count. Then

\[
 \sum\operatorname{Top}_K(s-\gamma y)
 \ge \sum_{i\in S}s_i-\gamma t,\qquad
 \sum_{i\in S}s_i\ge C_K^*(s).
\]

Consequently,

\[
 \boxed{R_K(s)=\frac{b-t}{K}\le L_K(s,y)},\qquad
 \boxed{P@K(s)\ge\frac{\min(K,M)}K-L_K(s,y)}.
\]

This is an upper-bound relationship, not a theorem that every decrease in L_K increases observedP@K. The surrogate is generally nonconvex and piecewise linear. It is not bounded above.

Its exact zero condition is that every sizeK setT obeys

\[
 C_K^*(s)-\sum_{i\in T}s_i
 \ge \gamma\left(b-\sum_{i\in T}y_i\right).
\]

Thus correctly selected groups need only achieve a sufficient margin; there is no requirement to push a sigmoid probability arbitrarily close to0/1. C*_K MUST retain its gradient. Detaching the oracle term destroys the intended gradient cancellation.

For M>=K this is the precision-at-the-top ramp construction studied by Kar etal.; the above proof explicitly includes the M<K oracle-floor case. Source: [Kar,Narasimhan,Jain2015](https://proceedings.mlr.press/v37/kar15.html). The finite-pool proof here does not import their conditional-consistency or learning guarantees for this neural training procedure.

### Step3. Simpler all-pair bound

For the dimensionless SUM over all positive-negative group pairs,

\[
 H_\gamma=\sum_{i:y_i=1,j:y_j=0}
 [1-(s_i-s_j)/\gamma]_+,
\]

every missed-positive/selected-negative crosspair contributes at least1. Hence

\[
 H_\gamma\ge(M-t)(K-t)=r(r+|M-K|)\ge r^2.
\]

So R_K<=sqrt(H_gamma)/K. A mean pairloss requires multiplying by M(N-M) before applying this bound. These are pairs of QUESTION GROUPS, never correctness ranking of individual trajectories.

The K-specific loss focuses on the actual retained portion and avoids requiring every low-ranked positive to outrank every negative.

### Step4. Dead-zone regression is a separate option

The proposed tolerance loss

\[
 L_\epsilon=\frac1N\sum_i[|\hat v_i-v_i|-\epsilon]_+^2
\]

has zero gradient inside the band. If epsilon<d/2, the disjoint reversed-pair argument gives

\[
 R_K\le\frac{2N L_\epsilon}{K(d-2\epsilon)^2}.
\]

A vanishing dead-zone loss is sufficient for correct empirical class ordering in that regime; arbitrary tolerance and nonzero average loss do not automatically imply highTopK.

For a tolerance on the structured loss, [L_K-tau]_+, the guarantee becomes R_K<=[L_K-tau]_++tau. A nonzero tolerance must not be omitted from the reported bound.

### Step5. Historical-score window with current-only gradients

For the human-proposedN2018/K500 window, combine1954 historical detached scores with64 current group scores. In a global two-GPU64-group update, each rank contributes32 current groups. The cache stores raw pre-update logit, actual ownG5 view/label, questionID and actualAdam modelversion. It stores neither hidden features nor an autograd graph.

Within a window, questionIDs remain unique. When a question returns with a new ownG5, remove its previous cached version before insertion. Eviction is chronological and label-independent. Save the queue and sampler/RNG state in the remote checkpoint.

During startup the queue may warm-fill; report the actual N_eff and use ceil(N_eff*500/2018), not a fictitious full2018. Development results remain the same fixed2048/G5 and Top512.

The empirical bound inStep2 remains exact for the mixed old/new vector tilde s. For comparison to the all-fresh current vector s, suppose

\[
 \|s-\tilde s\|_\infty\le\delta.
\]

Each sizeK score sum changes by at mostK*delta, including both maxima inL_K. Therefore

\[
 |L_K(s)-L_K(\tilde s)|\le2\delta/\gamma,\qquad
 \boxed{R_K(s)\le L_K(\tilde s)+2\delta/\gamma}.
\]

If the cached TopK boundary gap exceeds2delta, its selected set also equals the fresh selected set. Unknown score drift is not zero drift. Epoch-end fresh-window audits measure delta and selection changes at that saved model, without mutating the training queue.

Only current scores receive gradients. Loss>0 can involve exclusively cached rows and yield zero current gradient. Skip actualAdam/EMA updates in this case, preserving moments; also skip them when the global loss is exactly0. Report batchattempts separately from actualAdam/EMA updates and distinguish margin-satisfied from cache-only-inactive batches.

The optional multiplier N_eff/B compensates for smaller current-only gradient magnitude. With stale chronological rows, this is a declared scaling heuristic, not an unbiased-gradient claim. Log scaled trainingloss separately from the unscaled empirical-bound loss.

## Remarks and Interpretation

- The new controls use real Prefix80, block15 and an identical2048-position group head. No80-token prefix is fabricated from20-token assets.
- Matched MSE/BCE controls compare fixedG5 and randomM3..10 at the same initialization/EMA/LR/batch/budget. The cache experiment has its own matched global64 no-cache/window2018 pair.
- All current2048-question results are development results. They are not an untouched final generalization proof.
- Freeze the leading checkpoint, score/threshold rule and primarymetric before two-seed replication and the two untouched2048 confirmation pools.
- Raw receives gradients; EMA follows only actual raw updates. No EMA-to-raw reset or lost Adam state is allowed.

## Boundaries and Non-Claims

An empirical upper bound is not a statistical correlation coefficient, SGD monotonicity theorem or guarantee against overfitting. Smaller trainingloss can coexist with worse development selection. Cached-score theorems do not make historical logits current; finite sampled drift checks do not bound every future update. An online changing actor requires additional versioning and distribution checks.

Prefix-group ordering changes a causal packed input despite the label being permutation-invariant. The model's order sensitivity must be measured, not assumed absent. No confirmed improvement is claimed until the selected scheme is reproduced and evaluated on previously unused questions.

## Open Risks

Generalization, label noise conditional on short prefixes, raw/EMA lag, stale-score drift, current-only inactive constraints, the difference between randomM and G5, and training-subset comparability remain measured questions. The margin mechanism removes unnecessary optimization pressure after sufficient ordering; it does not prove that memorization cannot occur.


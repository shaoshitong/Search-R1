# Finite-pool Mixed TopK margin bound

Fix N labeled question groups, binary true-Mixed labels y, real scores s, K in{0,…,N}, and margin gamma>0. Select the K highest scores with any deterministic tie-breaking rule. Write P={i:y_i=1}, M=|P|, t=TP@K and r=min(K,M)−t. The optimal binary-Mixed yield is min(K,M), so r is the missed yield.

Define the complete positive-negative pairwise hinge sum

L = sum_{i in P, j not in P} max(0, 1−(s_i−s_j)/gamma).

Let A=P outsideTopK and B=TopK outsideP. Their sizes are a=M−t and b=K−t. Every selected negative j∈B has s_j≥s_i for every missed positive i∈A, including tied scores. Each of these a×b hinge terms is at least1. Therefore

L ≥ (M−t)(K−t) = r(r+|M−K|) ≥ r².

Consequently r≤sqrt(L), with the stronger bound

r ≤ [sqrt((M−K)²+4L)−|M−K|]/2.

Integer r can additionally be bounded by the floor. If all pairwise margins are satisfied then L=0 and TopK has optimal binary-Mixed yield. When0<M<N and the implementation uses mean pairwise loss l=L/[M(N−M)], the correct normalized statement is r²≤M(N−M)l. An unweighted sum over only sampled pairs is not the complete-pool sum in this theorem. In the all-positive/all-negative edge cases r=0 and the sum bound is trivial; the normalized mean is undefined.

For one pair, the hinge has zero gradient once its score gap reaches gamma (using the usual zero ReLU subgradient at the boundary). This says nothing about gradients from another active pair or an additional nonzero regression/calibration anchor. The score unit and gamma must match; a bounded score may make a requested margin infeasible.

## Variance MSE does not monotonically order Mixed TopK quality

Take true population strict-EM variances(.16,.24,0), so the first two groups are Mixed under G5 and the third is nonMixed. Let K=1.

- Prediction a=(.16,0,0) has MSE=(0+.24²+0)/3=.0192 and selects a true Mixed group.
- Prediction b=(.07,.08,.09) has MSE=(.09²+.16²+.09²)/3=.0139333333 but selects the nonMixed group.

Thus a strictly lower pure variance MSE can produce a worse binary-Mixed TopK selection, even when every prediction lies in[0,.25]. Scaling targets/predictions by4 multiplies both MSEs by16 and leaves the counterexample unchanged. This does not claim that exact variance fitting has no ranking guarantee: at zero error every positive G5 variance is at least.16 while every nonMixed variance is0, so the correct ordering follows. It refutes monotonic equivalence between MSE improvement and Mixed TopK improvement.

## Scope

The hinge bound is an exact deterministic statement about one fixed finite labeled pool. It does not prove unseen-question generalization, convergence of SGD, feasibility under ambiguous Prefix features, a monotonic increase at each update, or correctness under labels changed by a future student snapshot. Training-batch pairwise loss only supplies the same bound for that batch's ranking; it is not automatically the complete future candidate-pool loss. No new training recipe, margin value, threshold or final candidate is selected by this note.

## K-specific cost-augmented loss

For the same finite pool and1≤K≤N, let b=min(K,M). Define

C* = sum of the b largest positive scores + sum of the K−b largest negative scores,

A = sum of the K largest values of(s_i−gamma*y_i),

L_K = [gamma*b + A − C*]/(gamma*K).

Empty sums are0. Let T* be a label-optimal K-set achieving C*, and S the original score-TopK set with t true positives. By the two maximization definitions,

A ≥ sum_S s − gamma*t, and sum_S s ≥ C*.

Therefore gamma*b+A−C*≥gamma*(b−t)=gamma*r, hence L_K≥r/K. Substituting T* into the maximization defining A also proves that the numerator is always nonnegative. This proof covers M<K and deterministic score ties. The loss is generally nonconvex and piecewise linear, with convex or identically-zero special cases.

The exact zero-margin condition is: for every K-set T,

C*−sum_T s ≥ gamma*[b−sum_T y].

When this structure margin holds, L_K=0 and the finite-pool yield is optimal. With strict relevant margins and consistent active sets, its local gradient is zero. Away from ties the score gradient is the indicator of the cost-augmented topK set minus the indicator of the constrained label-optimal set, divided by gamma*K. C* must retain its gradient through the selected score values; detaching it destroys this cancellation and continued-score-stop behavior.

At topK/margin ties, an automatic differentiation implementation may select unequal internal active sets even when the final value is0. Applying ReLU to the mathematically nonnegative numerator leaves the real-valued objective/bound unchanged and selects zero ReLU gradient at exact0. Introducing an additional epsilon-insensitive band changes the bound: for L_epsilon=max(0,L_K−epsilon), r/K≤L_epsilon+epsilon. A no-slack bound must not be claimed for that modified objective.

This K-specific statement remains empirical to the scored labeled pool. It neither proves generalization to new questions nor makes a training optimizer batch128 equivalent to the actual2048-candidate/512-selection pool. No literature attribution is asserted or checked by this independent algebraic verification.

The displayed L_K is not generally bounded or clipped: for N=2,M=1,K=1,gamma=1 it reduces to max(0,1+s_negative−s_positive), which is unbounded. Its flat region for sufficiently good scores does not imply a flat region for extremely wrong scores. The cost-augmentation label y here must remain binary trueMixed; substituting continuous4v changes the utility being optimized and does not preserve the TP@K count-regret interpretation.

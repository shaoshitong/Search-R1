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

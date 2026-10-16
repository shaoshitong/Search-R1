# Derivation Package

Date: 2026-10-16, Asia/Shanghai. This note records the new human-authorized
fresh probe experiment. It does not alter the completed replay70 experiment.

## Target

Train the existing question-plus20-token trajectory correctness probe with both
its own trajectory label and its original G5 group's mixed/nonmixed label.
The acquisition score remains a deterministic function of the five predicted
trajectory probabilities. No additional classification head is introduced.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.

The observed group target is a binary event, not a measured conditional
probability. The product score represents its probability under conditional
independence of the five future continuations. Joint supervision is a valid
multiobjective training choice; improved held-out accuracy is an empirical
question rather than a consequence of the algebra.

## Invariant Object

The probability that the same five prefix-conditioned trajectories contain
at least one correct and at least one incorrect final answer.

## Assumptions

- The actor, official frozen encoder, tokenizer, generation protocol and strict
  EM labels are fixed for the retained trajectories.
- Inputs contain actual question/context and the same trajectory's first20
  retained policy-token positions. No future answer or final label enters the
  encoder or probe input. Encoder output is detached at index27 post-block.
- Each G5 group is identified using its actual question and trajectory IDs.
  The five members remain together within one rank and optimizer update.
- To interpret the product score as the true event probability, future
  continuations are conditionally independent given the question, their
  respective prefixes, actor version and generation protocol. Marginal
  correctness probabilities alone do not establish this assumption.
- Reusing a cached trajectory preserves its original label; replaying it does
  not create a new independent label observation.

## Notation

- i: question/group index; j in {1,...,5}: trajectory within its own group.
- C_ij in {0,1}: that trajectory's final strict EM.
- K_i = sum_j C_ij: observed correct count. G=5 is the fixed group size.
- Y_i = 1{0<K_i<5}: observed mixed label. K_i=1 is mixed as well.
- z_ij: probe logit; p_ij = sigmoid(z_ij): predicted trajectory correctness.
- Q_i = 1 - product_j p_ij - product_j(1-p_ij): predicted mixed score.
- lambda: group-loss weight, explicitly 1 for this experiment.
- B: number of complete question groups in an optimizer batch.

## Derivation Strategy

Define the mixed event from the five own labels, derive its complement
probability under the stated factorization, and add its binary log loss to
the existing per-trajectory binary log loss. Separate the probability
interpretation from the finite-model optimization choice.

## Derivation Map

1. Identity: Y_i is a deterministic function of the five observed C_ij.
2. Probability identity: a nonmixed group is either all correct or all wrong.
3. Conditional-independence assumption: replace those two joint probabilities
   by products of the five corresponding marginal probabilities.
4. Proposed objective: separately average the trajectory and group log losses.
5. Exact differentiation: the group term sends gradient through the same five
   logits; no detached or second group prediction is needed.
6. Implementation: stable log-event probabilities and complete-group batching
   preserve the mathematical objective in eight-rank training.

## Main Derivation

### Step 1. Event label — identity

K_i = sum_j C_ij and Y_i = 1{0<K_i<5}. The condition 1<K_i<5 would incorrectly
exclude a group with exactly one correct trajectory. One retained G5 supplies
one observed Y_i, not the population probability P(0<K_i<5 | inputs).

### Step 2. Predicted event probability — conditional factorization

The complement of mixed is the disjoint union of all-correct and all-wrong:

Q_i = 1 - P(C_i1=...=C_i5=1 | inputs)
        - P(C_i1=...=C_i5=0 | inputs).

Under the stated conditional-independence model this becomes

Q_i = 1 - product_j p_ij - product_j(1-p_ij).

The five p values need not be equal. Averaging them before applying the
identically distributed formula would change the modeled event.

### Step 3. Joint objective — proposed training choice

L_traj = (1/(5B)) sum_i sum_j BCEWithLogits(z_ij, C_ij).

L_group = (1/B) sum_i [-Y_i log Q_i - (1-Y_i) log(1-Q_i)].

L = L_traj + lambda L_group, with lambda=1.

Both components have their own mean reduction. Merely summing five trajectory
losses and one group loss would silently change their relative weight.
Y_i reuses the original five EM labels; it adds no new teacher observations.
It does explicitly make errors in the final group score part of the objective.

If the factorization is correct and model capacity and optimization suffice,
true conditional trajectory probabilities minimize both population log losses.
If the factorization or representation is inadequate, the two objectives can
trade off trajectory calibration against group discrimination. Improvement of
both is not guaranteed.

### Step 4. Gradient — exact algebra

For A_i=product_k p_ik and D_i=product_k(1-p_ik),

dQ_i/dz_ij = p_ij D_i - (1-p_ij) A_i.

The group BCE derivative for 0<Q_i<1 is

dL_group_i/dz_ij = ((Q_i-Y_i)/(Q_i(1-Q_i)))
                  * (p_ij D_i - (1-p_ij) A_i).

At p_i1=...=p_i5=0.5, Q_i=15/16 and every dQ_i/dz_ij is zero.
Trajectory BCE still has derivative p_ij-C_ij. The trajectory term anchors
which particular response is correct; the group label alone cannot do so.
The group score is unchanged by permutation of trajectories and by complementing
all five probabilities. These symmetries are another reason to retain the
own-trajectory objective.

### Step 5. Stable computation — equivalent event log probabilities

For each binary pattern a in {0,1}^5 define

ell_i(a) = sum_j [a_j log sigmoid(z_ij)
                    + (1-a_j) log sigmoid(-z_ij)].

log Q_i = logsumexp_{a: 0<sum(a)<5} ell_i(a),
log(1-Q_i) = logsumexp_{a: sum(a) in {0,5}} ell_i(a).

There are30 mixed patterns and2 nonmixed patterns. Their log-probability
difference is an event logit. BCEWithLogits on that event logit gives the group
loss without probability clipping or loss of gradients at rounded0/1 outputs.

### Step 6. Complete groups and DDP — exact mean scaling

An epoch contains5120 complete groups,25600 trajectories. A strict256-trajectory
batch cannot simultaneously consist solely of complete G5 groups. The chosen
100-update plan uses80 batches of51 groups (255 trajectories) and20 batches of52
groups (260 trajectories), averaging256 trajectories per update.

Each rank receives6 or7 complete groups. Rotating the extra groups gives640
groups and3200 trajectories per rank per epoch. There is no padding, duplication,
dropped tail or gradient accumulation. For n_r local groups and N global groups,
multiply the local mean loss by 8*n_r/N before DDP's mean gradient reduction.
This is the gradient of the intended global mean for both loss components.

## Remarks and Interpretation

- Human-authorized fresh run:100 epochs; start with official copied block35
  and a new standard-initialized Linear. No old probe, Adam, optimizer step or
  RNG checkpoint is imported. The official frozen encoder remains pretrained.
- Epoch data order: cached sources1..70, then1..30. Every epoch uses one source
  batch exactly once. There is no new rollout or retrieval work.
- Frozen-encoder extraction occurs inside each training batch. Already consumed
  features may remain in per-rank CPU RAM until that epoch's post-update train
  evaluation, then are released. No full-epoch precomputed feature stage or
  persistent training-feature tensor cache is required.
- Reuse the unchanged fixed512G5 test features. Report train and test group
  ordinary and balanced accuracy, both class recalls, and the separate losses.
  The threshold remains0.5. Train batches differ across epochs.
- For actual Adam update t=0,...,9999, use
  lr(t)=3e-6 + (3e-5-3e-6)/2 * [1+cos(pi*t/9999)].
  The first applied LR is3e-5 and the last is3e-6; no extra warmup is added.

## Boundaries and Non-Claims

This objective does not provide new EM labels, establish conditional
independence, guarantee perfect fitting, prove calibration, or demonstrate
better held-out selection or QA performance. It changes the probe's objective;
it does not update actor/PPO or change cached trajectories. A fixed test used
repeatedly for diagnosis is not an untouched confirmation set for later tuning.

## Open Risks

Finite information in early policy tokens, conditional-factorization error,
competing calibration and group objectives, and generalization from repeated
cached question clusters remain empirical issues. Numerical stability is handled
by equivalent log-domain algebra, not by speculative stopping gates or changes
to the scientific labels.

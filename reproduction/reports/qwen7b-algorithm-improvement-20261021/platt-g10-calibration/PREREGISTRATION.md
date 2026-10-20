# Frozen-probability calibration check

Preregistered before any calibration input scoring completed or calibration
parameters were fitted. The observed failure motivating this check is that
allcorrect questions have mean individual probability0.38773 and mean Mixed
score0.74355, exceeding true Mixed questions' mean score0.63149. This does not
prove that calibration will fix discrimination. The experiment tests that
specific hypothesis with two learned scalars instead of another large fork.

## Inputs and scope

- The already accepted8192 NQ questions, ten authentic trajectories each, with
  exact own trajectory IDs/strict EM, source/feature SHA bindings.
- Frozen E4 EMA and peer8192-CP256 EMA are each scored once on those exact
  features. There are no actor, native block or existing optimizer updates.
- Those questions were seen by original warmup, and by the peer fine-tune.
  Internal holdout refers only to fitting the two calibration parameters;
  it is not an unseen-native-probe generalization claim.
- All tensors, full predictions, learned coefficients and full checkpoints
  remain remote. Local artifacts contain code/configuration/small audits.

## Fixed finite protocol

1. Identity-only hash order with seed2718 separates exactly6144 fitting and
   2048 parameter-holdout questions. All views of a question share one split.
   This exact split supersedes the early approximate80/20 sketch before any
   fit/input-scoring outcome was available, to match Top512/2048 quotas.
2. For holdout assessment, one sorted five-subset per question is frozen by
   question hash/seed2718 without labels. Fitting uses all252 real G5 subsets,
   averaging each original question to total weight1.
3. Probability-only Platt transform uses canonical
   `z=logit(clip(stored_p_j,1e-6,1-1e-6))` for both fitting and external input.
   The transform is `sigmoid(a*z+b)`, with `a=exp(3*tanh(u))>0` and
   `b=10*tanh(v)`. It preserves trajectory ordering up to declared clipping
   ties. The uncalibrated comparison uses original, unclipped probabilities.
4. Six fits per source: trajectory BCE or trajectory BCE+three-event NLL,
   crossed with identity regularization0/.01/.1 on `log(a)^2+b^2`.
   Torch LBFGS is configured for40 iterations/max_eval60, with source-question
   chunk128 and CPU2. Global wall/RSS ceilings are26min/3GiB.
5. At fixed Mixed threshold0.5, a candidate must improve both internal BA
   and internal Mixed@512 versus its uncalibrated source. Among qualifying
   candidates across both sources, choose exactly one by largest
   `BA+Mixed@512_fraction`, then lexical candidate ID. If none qualifies,
   do not open external predictions.
6. Freeze selected source, model/checkpoint SHA, two parameters, internal
   metrics and protocol first. Only then transform the single chosen source's
   existing same-HF fixed2048 predictions and evaluate once. No external
   threshold fitting, second winner selection or final4096 access.
7. Save each fit's raw scalars, complete LBFGS state, Torch/NumPy/Python RNG,
   input/split/protocol bindings and phase. Immediately load/compare all
   components. Synthetic next-update restoration is separately tested.

## Mathematical qualification

The new three-state log-space dynamic program computes the same G5
allwrong/Mixed/allcorrect probabilities as the32-pattern reference. Focused
tests compare double-precision values and gradients, including extreme
logits, all252 G10 views and checkpoint next-update restoration. It changes
temporary computation size, not the probability model or label definition.

This remains a diagnostic until actual external BA and Mixed@512 jointly
improve and a frozen candidate is confirmed on untouched question pools.

# Derivation Package

Date: 2026-10-16, Asia/Shanghai. This is a proposed mathematical reframing; it does not change the stopped experiment or authorize restarting it.

## Target

Explain why a question's mixed/nonmixed label changes across groups of five responses, and derive objectives that estimate its propensity to produce an informative G5 group. Distinguish finite-cache fitting, new-sampling prediction, and cross-question generalization.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.

The stable target is a probability conditional on the question, actor version, and generation protocol. A particular realized G5 label is a random observation of that probability.

## Invariant Object

q_t(x): probability that a future G5 from question x and actor version t contains both correct and incorrect responses. This is the acquisition utility; it is not a deterministic semantic class of a question.

## Assumptions

- Fix actor parameters, tokenizer, retrieval environment, sampling settings, and strict binary EM scorer while analyzing one question.
- For the binomial formulas, correctness indicators within G5 are conditionally independent and identically distributed Bernoulli(p_t(x)). Independent repeated G5 groups use fresh sampling randomness.
- Independence is a modeling assumption, not established by the aggregate metadata alone. Correlated sampling changes the binomial formulas.
- Across actor updates, p_t and q_t can change. Labels from different versions are not interchangeable stationary repetitions.
- Existing cached teacher G5 labels are fixed observations. Replaying them for ten epochs does not create ten independent G5 observations.

## Notation

- C_j: binary correctness of response j, j=1,...,5.
- K=sum_j C_j: observed number correct.
- Y=1{0<K<5}: observed mixed-group label.
- p=P(C_j=1 | x,t): single-response success probability.
- q=P(Y=1 | x,t): future mixed-group probability.
- v=p(1-p): population correctness variance.
- S²=K(5-K)/20: observed unbiased sample variance for G5.
- r: model's predicted mixed-group probability.

## Derivation Strategy

Derive the acquisition utility from the G5 sampling distribution, then derive expected-loss optima. Separate exact identities under the binomial assumptions from proposed finite-sample fitting choices.

## Derivation Map

1. Conditional independence gives K~Binomial(5,p).
2. Complement events K=0 and K=5 give q.
3. Independent repeated labels give their flip probability and the Bayes classification ceiling.
4. Expected BCE identifies q as its optimum; fixed-cache interpolation does not prove estimation of q.
5. Count likelihood uses all K information to fit p; q is then computed analytically.
6. The sample-variance identity gives an alternative continuous target; taking square roots changes the regression target.
7. Fitting capacity, regularization, and repeated-sampling evaluation remain necessary; a loss change alone is not a proof of improved generalization.

## Main Derivation

### Step 1. Mixed propensity — identity under the binomial model

K~Binomial(5,p), therefore

q(p)=1-P(K=0)-P(K=5)=1-(1-p)^5-p^5.

q is symmetric around p=1/2, maximized there, and q(1/2)=15/16=0.9375. Under these assumptions even the most mixed-prone question does not produce a mixed G5 with certainty.

### Step 2. Repeat-label flips — proposition under independence

For two independent G5 labels Y and Y' for the same fixed question and actor,

P(Y≠Y')=q(1-q)+(1-q)q=2q(1-q).

Examples:

- p=0.5: q=0.9375 and flip probability=0.1171875.
- p=0.1: q=0.4095 and flip probability=0.4836195.

For heterogeneous questions, average 2q_i(1-q_i) over questions. Substituting the mean q into this expression is generally incorrect.

The optimal deterministic binary prediction for one question has accuracy max(q,1-q), not necessarily 100%. Across a question distribution the ceiling is E[max(q(x),1-q(x))]. It cannot be inferred by treating a selected overlap subset's flip rate as the population rate.

### Step 3. What BCE estimates — expected-loss identity

E[BCE(r,Y)|x,t]=-q log(r)-(1-q)log(1-r)

=H(Bernoulli(q))+KL(Bernoulli(q)||Bernoulli(r)).

The minimum occurs at r=q. Thus the existing binary BCE is a mathematically proper objective for the mixed probability. Its problem is not that stochastic labels invalidate BCE. A high-capacity model trained repeatedly on one fixed Y per question can interpolate those realizations instead of estimating their probabilities.

### Step 4. Count-based fitting — proposed model with a proper likelihood

Predict p_theta(x,t)=sigmoid(z_theta(x,t)) and fit

L_count=-K log(p_theta)-(5-K)log(1-p_theta),

ignoring the binomial coefficient, which is constant with respect to theta. Dividing by five gives BCEWithLogits(z_theta,K/5). Use the fused logit formulation for numerical stability.

Then acquire questions with score

q_theta=1-p_theta^5-(1-p_theta)^5.

This uses whether K was 0,1,2,3,4,5, rather than collapsing it to one binary event. The selection prefers estimated p near 1/2. It uses a shared predictive model as a smoothing assumption and still needs capacity control; it does not eliminate finite-sample noise or guarantee generalization.

Do not treat 1-(K/5)^5-(1-K/5)^5 as an unbiased soft label. q''(p)=-20[p^3+(1-p)^3]<0, so Jensen's inequality gives E[q(K/5)]≤q(p), strictly inside (0,1). For p=1/2, E[q(K/5)]=0.78 although q(p)=0.9375.

### Step 5. Variance relation — exact identities and a regression distinction

E[S²]=p(1-p)=v.

Since p+(1-p)=1, algebra gives p^5+(1-p)^5=1-5v+5v². Consequently

q=5v(1-v), 0≤v≤1/4.

This is increasing on [0,1/4], so ranking true v and true q is equivalent under the binomial assumptions. A second possible model therefore predicts v and regresses against S². The conditional expected squared-error optimum is v, even though individual S² observations can be as large as 0.3.

The original sample-standard-deviation target is sqrt(S²). Its squared-error optimum is E[sqrt(S²)|x,t], which is generally smaller than sqrt(E[S²])=sqrt(v). Replacing it with sqrt(p_theta(1-p_theta)) silently changes the target and should not be described as the same sample-std regression.

### Step 6. Direct repeated-G5 alternative — fewer distributional assumptions

With M independently sampled G5 groups at the same actor version, estimate q from their mean mixed indicator. Its conditional variance is q(1-q)/M. A smoothed estimate such as (sum_m Y_m+a)/(M+a+b) introduces an explicit Beta prior rather than claiming to be assumption-free.

This supports keeping the mixed-probability head while fitting soft frequency targets. It costs more generation. No extra repeated G5 generation is authorized by this note; the current requested diagnostic remains one new random512×5 set.

## Remarks and Interpretation

- Minimal proposed experiment: retain the requested native-block architecture initially, use K-based stable logit loss for p, derive q for selection, and remove the independent std output or clearly redefine its auxiliary target as sample variance. This is a proposal requiring an explicit algorithm update before execution.
- Reduce head learning rate and add capacity regularization; if needed, compare fewer trainable block parameters or a frozen-block readout. These are empirical choices, not mathematically guaranteed fixes. Avoid simply increasing fixed-cache epochs until training accuracy approaches 100%.
- Keep binary accuracy as requested, but interpret it with the actual class mix and majority-class baseline. Acquisition quality is assessed by the selected groups' future mixed fraction, not training accuracy alone.
- Same-question resampling and unseen-question evaluation answer different questions. Keep actor versions and generation protocols fixed when comparing repeated labels.

## Boundaries and Non-Claims

- The real CPU comparison verified all 14 tensors/78,076,418 elements in the original Head200 and formal initial Head200 are exactly equal, as are Adam and private RNG.
- A fresh inference-only control reproduced 5114/5120=99.8828125% on the original cached training labels; BCE=0.0366393862. That verifies the cached-fit number, not generalization.
- In the selected first formal batch, 53 questions overlapped the teacher set. All 53 had cached mixed labels; 37 were mixed on fresh G5 and 16 flipped. This selected subset is biased and does not estimate the whole dataset's flip rate.
- The first formal selection was correctly sorted by its saved probabilities. Only eight candidate probabilities were exactly one, so large saturation ties do not explain that first selection's low mixed fraction.
- After the first online Head update, 1,412 candidate probabilities were exactly one. The selected512 were all tied at one. This is a separate observed confidence/ranking problem, not evidence that the initial weights differed.
- On the original5120 control, every std prediction was exactly zero. Its regression MSE equaled mean target sample variance, 0.0884375. The std auxiliary head was therefore not producing useful question-dependent predictions at this checkpoint.
- New random512 results are pending. This note does not invent them or diagnose all degradation as either code error or overfitting.

## Open Risks

- Within-group sampling correlation, changing retrieval behavior, or actor drift can violate the fixed-p binomial approximation.
- One G5 per question cannot accurately identify each question's propensity without shared structure, a prior, or additional sampling.
- The native block has substantial capacity relative to 5,120 fixed noisy observations. Count-based fitting can also memorize noisy K values if unconstrained.
- The cause of the online confidence explosion and dead std output still requires focused analysis; observed saturation is not by itself a complete causal diagnosis.
- Selected-only online replay changes the training distribution. Calibration on that distribution need not transfer to the full candidate pool.

Evidence: preparation/qprobe-native-new512-cpu-weight-compare-20261016.json; preparation/qprobe-native-head200-control-read-1792083940623678000.json; preparation/qprobe-native-new512-existing-ranking-overlap-20261016.json; preparation/qprobe-native-same-question-G5-label-and-std-audit-20261016.json. All raw model, features, labels, and predictions remain remote.

# New transparent three-objective family phase

This new phase extends the currently frozen-E1 Count-head capacity experiment
with two same-cache/same-actual-CP0 head-only objective controls. It does not
rewrite the original Count12-node protocol or change the active Count source.
It is frozen before any current Count prediction file or completed epoch node.

The36 predeclared nodes are Count, binary and three_event, each raw/EMA at
epoch0/1/2/4/8/20. Every recipe uses the same frozen rawE1 cache, actual small
CP0 raw/EMA/Adam0/RNG, original30016 questions,938 fixedB32 updates per epoch,
20 epochs, four unique G5 views per question with weight0.25, LR1e-4 and EMA0.99.
No new GPU is submitted by this protocol or metadata preparation.

The fixed internal5120 holdout keeps its committed own-newseed five indices5..9
and own strictEM. NQ2560 plus the same two label-blind1280 halves determines
eligibility: NQ BA and Top640 trueMixed count must both strictly exceed the
same-cache frozen-E1 product-q baseline; each half BA and Top320 count must not
fall. Hotpot2560 and pooled5120 remain separately reported and do not gate.

Wait for every one of36 nodes. Choose at most one across the entire family by
NQTop640 descending, BA descending, earlier epoch, EMA before raw, then fixed
recipe order Count/binary/three_event. Original Count12-node eligibility cannot
bypass this new family budget. No eligible node means no external candidate.
A passing family screen only permits preparing one candidate; actual external
scoring still requires independent loaded-candidate same-five online micro8
qualification and a separate root GO. Final4096 outcomes stay unopened.

The two new objectives use sourcee155860790cde71fc2d0edd5cfdc54ce6fe882f4:
CE(event logits+log(train prior/reference prior),true coarse event), with the
exact original938/four-view train-only prior receipt and no validation prior.
Inference never adds that offset: raw eventsoftmax Mixed equals the sum of
raw softmax6 latent slots1..4. For these coarse objectives it is a reference-
balanced decision score, not an original-distribution calibrated probability.
Six latent slots do not imply learned six-count probabilities; countNLL and
six-way accuracy will not be used as training-correctness evidence.

Input resolution is fail-closed: absent actual cache/CP0/epoch0 receipt returns
WAIT, never fabricated descriptors. Metadata and small ZIP checks are permitted
on the controller; actual CP0 four-component restoration occurs only in the
future fork allocation. The large hidden cache is never loaded on login.

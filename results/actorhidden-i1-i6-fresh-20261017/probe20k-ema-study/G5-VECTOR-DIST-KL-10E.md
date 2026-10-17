# G5 vector DIST + mean KL + diversity KL, 10 epochs

Human approved: 5120 existing training questions G5, fixed actor0050, question+20 readout, LR5e-5, EMA.99, 10 epochs, no individual-trajectory BCE. Same matched source head40/Adam200/eight RNG; new EMA initialized from that source. No new rollout. All full outputs remain remote.

Let p=sigmoid(z), c be the ordered own-EM vector of the same five trajectories. Continuous hidden mixup uses lambda=max(u,1-u)>.5, u~Beta(.2,.2), with each paired row retaining its own primary sample. No token patch replacement. The sole soft target vector is r=lambda*c_A+(1-lambda)*c_B; B has its own-primary symmetric target. Only compatible mask/position layouts are paired. Unpaired rows keep real labels.

For any five-vector v, define mu(v)=sum(v)/5 and D(v)=4/25 sum_{j!=k} v_j(1-v_k). On real binary labels D=4K(5-K)/25, giving 0/.64/.96 for K=0,5 / 1,4 / 2,3. On soft vectors D is the expected finite-G strength, not 4mu(1-mu), and is not the binary mixed-event probability.

The loss is mean_question[1-rho(p,r)] + KL(Bern(mu(r)) || Bern(mu(p))) + 5 KL(Bern(D(r)) || Bern(D(p))). The Pearson term is omitted for constant target vectors; the two KL terms still separate all-correct and all-wrong. This is within-question five-dimensional Pearson relation alignment, not a separate five-class head or inter-question correlation. Stable log-domain mean/diversity logits use the 32 Bernoulli outcomes; targets and outputs share the same functional. No individual BCE, binary mixed-event BCE, count-classification, MSE, probability clamp, or independent mixed-label interpolation enters training.

Eight-rank natural once-per-epoch traversal: 100 synchronous Adam updates/rank/epoch (80 global255 and20 global260 trajectories, fullG5). Final actual Adam1200/rank includes source200; new1000, EMA1000. Mixup is training-only. After every epoch the same raw and EMA heads evaluate unaugmented fulltrain5120G5 and fixedtest512G5.

Report all epochs0..10, raw/EMA, train/test: each objective term and total, vector MAE/within-mixed-group Pearson, groupmean MAE, diversity MAE and threshold.5 ordinary/balanced mixed accuracy. Also retain the historical q=1-prod(p)-prod(1-p) threshold.5 mixed accuracy for comparison, explicitly separate from D. Existing trajectory/binarymixed BCE values are diagnostics only. No formal actor training resumes.

Status: implementation prepared; actual launch/results pending. CPU-only mathematical fixture checked all32 hard-label patterns, common soft-target optimum, constant groups, stable endpoint gradients, and continuous dominant hidden interpolation. This was not a GPU/model test or training gate.

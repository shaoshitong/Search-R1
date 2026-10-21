Existing E0 only: independent probability-record audit

The stdlib CPU auditor independently rederived all four existing raw/EMA x diagnostic512/fixed-selection2048 records. No model/tensor load, new predictions, training edits or final4096 reads occurred. Heldout own-G5/EM identities match across raw/EMA.

EMA fixed-selection2048: Mixed BA55.7714%, trueMixed Top512=196/512=38.28125%; TP448/FN215/TN609/FP776. Allwrong540/1055=51.1848% were calledMixed; allcorrect236/330=71.5152% were calledMixed; Mixed recall448/663=67.5716%. Probability-record-derived clipped MixedBCE: allwrong1.06160, Mixed0.61752, allcorrect1.48182. These are before-training values, not an improvement claim.

Each future completed epoch is independently checked for checkpoint/config/fixedG5/ownID+EM stability and confusion/Top512 equality. Its true-class BCE and p-range diagnostics expose whether nonMixed-only loss reduction hides Mixed recall deterioration. Missing epoch1/2/3/4 remain pending. The existing joint goal gate requires both BA>56.2504968663% and Top512>196, with a later untouched4096 confirmation.


Existing E0 Top512 composition independently recomputed (not new training gain)

EMA selected K0..K5=[209,44,38,46,68,107]:209allwrong+196Mixed+107allcorrect. Cutoff0.8720808029174805 has one question, so no large tie at selection boundary. Class selection rates are209/1055=19.81%allwrong,196/663=29.56%Mixed,107/330=32.42%allcorrect. Old EMA ranks allcorrect questions above Mixed too frequently; this is a specific failure mode to compare after each new epoch. Scores are ordered first without labels; class counts are diagnostic only. Actual ownG5 score/payloadSHA verified, no GPU/model load and untouched4096 not opened.

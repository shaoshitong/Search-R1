# Existing predictions: domain and original warmup membership

The internal confidence holdout and external fixed-HF validation are **both entirely NQ**. This comparison does not measure NQ-to-other-domain shift. All 8192 internal-cohort questions (including the scalar-fit 6144 and heldout 2048) belong to the original probe warmup's question set. The external 2048 has zero matching original warmup question IDs and zero matching canonical prompt hashes. These facts distinguish seen-question diagnostic performance from unseen-question validation performance; they do not alone prove the cause of the gap.

The original canonical warmup pool contains 169028 questions: 78871 NQ and 90157 HotpotQA. Its producer identity and ordered-ID digest were independently reproduced from the parquet's `prompt`, `data_source`, and `extra_info.index` projections. No reward, target, EM, features, model, or tensor column was read. The canonical full-file SHA is inherited from the accepted producer; this audit checks the projected identities, exact ordered membership digest, and unchanged pre/post file stat, and does not claim a new full-file SHA pass.

## Fixed external NQ validation

All models use the same 2048 actual fixed-G5 question/trajectory IDs and own strict EM labels, with 663 Mixed questions (32.373%). The stored label-blind q formula, prediction SHA, original fixed-G5 plan, canonical question hashes, and accepted append512 audit were rechecked. Threshold remains 0.5. Top512 is selected once globally by `(-q, qid)`, then partitioned by domain; there is no new per-domain selection.

| Existing model | BA | Mixed recall | NonMixed recall | AllWrong FPR | AllCorrect FPR | AUC | Global Top512 Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2, same HF | 56.2505% | 74.8115% | 37.6895% | 56.3981% | 81.2121% | 0.569053 | 181/512 (35.35%) |
| E4, same HF | 55.7714% | 67.5716% | 43.9711% | 51.1848% | 71.5152% | 0.576476 | 196/512 (38.28%) |
| Append cursor512 | 55.6880% | 70.4374% | 40.9386% | 50.9005% | 85.1515% | 0.566600 | 185/512 (36.13%) |

Whole-pool true K=0..5 counts are `[1055,181,136,132,214,330]`. The original global Top512 K composition is:

- E2: `[227,35,29,46,71,104]`.
- E4: `[209,44,38,46,68,107]`.
- Append512: `[197,44,30,48,63,130]`.

The append512 all-correct false-positive rate rose from E4's 71.52% to 85.15%. Its top selection contains 130 all-correct questions versus E4's107, while retaining fewer Mixed questions. The all-wrong FPR changes little. This is a descriptive decomposition of accepted scores, not a newly selected recipe.

TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue and Bamboogle have zero questions in this validation, so their BA/recalls/FPR/AUC are **NA**; these results do not establish performance on those domains.

## Internal confidence NQ diagnostic

The original E4 score on the heldout new-seed five trajectories uses 2048 warmup-seen questions: 655 Mixed (31.982%), BA64.0519%, Mixed recall76.4885%, NonMixed recall51.6152%, all-wrong FPR42.50%, all-correct FPR68.6901%, AUC0.680658, Top512 Mixed238 (46.48%). Its global Top512 K composition is `[180,59,39,47,93,94]`.

The scalar calibration holdout is held out only from fitting the small calibrator. It is not an unseen-question holdout for the original probe. No new scalar predictions or calibrator state were opened in this domain audit; it uses only existing original E4 probabilities.

## Actual execution and bindings

The v1 read-only attempt stopped after the external three-model checks because the independent split check initially confused hash-ranked selection order with the producer's original-cohort storage order. The original frozen split was correct. The isolated v2 implementation verifies the same hash-only selected set and original-cohort order. V1 files/intent/log/exit remain preserved; no training, fitting or sampling was repeated.

Source v2: `6d2b773b` (remote verified before staging). Two focused stdlib fixtures passed; an additional synthetic fixed-membership/original-order check passed. Actual worker completed in4.082 seconds, peak RSS338993152 bytes, two CPU cores and zero GPUs. A second independent status read reproduced the same terminal receipt SHA.

Remote terminal receipt: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/existing-predictions-domain-cursor512-v2/complete.json`, 51622 bytes, SHA256 `7b9d3fe8f4b1d36cb23812c69a7fc1ed68221dd10195ab194805a1907749f570`.

All probability rows, original questions and trajectories remain remote. No fitting, threshold changes, GPU, model/optimizer tensors, new candidate scores, per-domain reselection, or final-pool access occurred.

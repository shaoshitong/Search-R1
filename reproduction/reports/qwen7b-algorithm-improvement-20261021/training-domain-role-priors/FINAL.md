# B128 real-G10 domain and role priors

The current 169028-question source has a class–domain association under its global50:50 Mixed/nonMixed-role sampler. NQ has a45.7964% Mixed-role prior; Hotpot53.1767%. Global class balance does not imply per-domain balance.

| Domain | Questions | K10=1..9 positive-eligible | G10 heterogeneous | Uniform own-G5 natural Mixed |
|---|---:|---:|---:|---:|
| nq | 78871 | 34023 | 43.1375% | 32.6784% |
| hotpotqa | 90157 | 52277 | 57.9844% | 45.9013% |

G10 positive eligibility and natural G5 Mixed probability are different quantities. UniformG5 probability is averaged per original question using `1-[C(K10,5)+C(10-K10,5)]/252`; labels are not resampled.

| Domain | P(domain given Mixed role) | P(domain given nonMixed role) | P(Mixed role given domain) | Share of balanced stream |
|---|---:|---:|---:|---:|
| nq | 39.4241% | 46.6615% | 45.7964% | 43.0428% |
| hotpotqa | 60.5759% | 53.3385% | 53.1767% | 56.9572% |

Positive pool:86300 heterogeneous G10 questions. Negative pool:all169028 questions. Each role supplies half the question occurrences; each occurrence supplies four own-class G5 views at weight1/4. Thus the equations are `P(d|+) = M_d/M`, `P(d|-) = N_d/N`, and `P(+|d) = (M_d/M)/[(M_d/M)+(N_d/N)]`. These are early uniform-role pool marginals, not an observed count of saved training batches.

The sampler selects positive questions first and prevents repeated qids inside B128. A hypothetical fresh-uniform-permutation first-batch correction yields NQ45.7949% rather than45.7964% (Hotpot53.1780% rather than53.1767%); the adjustment is immaterial to this mechanism. Shuffled-cycle skipping and later hard-pool sampling mean these formulas must not be presented as exact realized batch-domain frequencies. No per-qid batch trace was available/read or reconstructed.

A population BCE predictor using only domain would emit approximately0.458 for NQ and0.532 for Hotpot; at the unchanged0.5 threshold it would call NQ nonMixed. This gives a possible directional mechanism for NQ mean score near0.47 and low recall. It does **not** show that the actual model learned this shortcut, nor that this alone explains its failure. Its hypothetical balanced-role training BA would be53.62%, while a single-domain constant predictor has BA50%. These are algebraic illustrations, not a new fitted model or validation score.

K10 histogram (K0 throughK10):

- NQ: `36492,5400,3458,2894,2617,2638,2907,3415,4480,6214,8356`
- Hotpot: `31793,7488,5661,5087,5008,5089,5360,5722,6218,6644,6087`

Source config: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1/config.json`, SHA256 `e5f0027dfcff4b6db442b38e9b2452e86cc2c9adcabd97a194971dc4d8f8d837`. Current groups127068810B, SHA256 `9b6616b041170e7b3fc3de5128376e12f971842360b7a79b178991941038915f`; full canonicalqid-set and accepted domain-count metadata agree. Fixed6656 evaluation exclusions were checked as identity metadata only: qid/hash overlap0. No evaluation outcome, feature, rollout, model or checkpoint was read.

Actual CPU run2.109s, peakRSS451.5MiB; a second independent small-receipt readback has the sameSHA. Main746266 and all training files remain unchanged.

Remote complete: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/training-domain-role-priors-v1/complete.json` (6127B), SHA256 `e9b67c1dc80e8a8def53746a6fbfc78167330cc153cd9955eda2dd5f0571c8eb`.

Source: `568bbceaae43885663e01db91f3d1d7d8daa11bb`, `codex/training-domain-role-priors-20261022`.

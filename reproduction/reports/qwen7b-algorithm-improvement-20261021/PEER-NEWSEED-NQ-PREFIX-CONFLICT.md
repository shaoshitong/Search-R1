# Frozen peer newseed: NQ-only Prefix ambiguity

Actual fixed scope: 244 sealed batches, 31,232 NQ questions and 156,160 real
own-G5 trajectories. This partial source snapshot is entirely NQ; it is not
an IID sample of the169028 train pool or evidence about seven-domain gains.
Actor official7B revision a09a35458c702b33eeacc393d103063234e8bc28 is frozen.

All diagnostic512, viewed-selection2048 and frozen-final4096 identity lists
were projected to qid/hash and excluded **before accessing own EM**. Actual
intersection is zero by qid/hash, including the original169028 train identities.
The copied source train SHA and ordered-qid legacy SHA match the accepted B
canonical train provenance. Source prefixes/ownEM were freshly hashed and
joined by own trajectory/qid/prefix SHA; no archive, full rollout or tensor scan
was repeated. No p_j or q_i is available or inferred.

K0..5 counts: `[15887,2725,2115,2225,3222,5058]`; real Mixed=10287/31232=32.94%.

| Metric and denominator | Prefix5 | Prefix10 | Prefix20 |
|---|---:|---:|---:|
| Question has any repeated prefix (/31232) | 90.96% | 47.96% | 8.03% |
| All five identical (/31232) | 6.70% | 0.76% | 0.01% |
| Identical-prefix unordered pair (/312320) | 31.37% | 9.49% | 1.03% |
| Same-prefix/opposite-EM pair (/49828 correct-wrong pairs) | 29.20% | 9.15% | 0.92% |
| Real Mixed contains a same-prefix/opposite-EM pair (/10287) | 63.41% | 27.14% | 3.65% |

Token-only empirical trajectory-BCE minima (nats/trajectory): 5=0.082927, 10=0.030590, 20=0.003597.
These empirical minima apply only to a deterministic scorer of question plus
exact prefix tokens. They are not a Mixed balanced-accuracy bound and do not
promise that a neural probe reaches them. All three lengths found zero equal-
token clusters with different recorded causal contexts/positions. Natural short
prefixes are kept unpadded (0/2/6 trajectories shorter than5/10/20).

The output supports Prefix20 removing most literal input/label ambiguity;
it does not establish probe calibration, Mixed BA, AUC or top512 selection.
Those require real fitting and NQ heldout evaluation. Existing mixed-domain6528
controls remain intact; seven-domain generalization is not claimed.

Raw data stayed on NAS10. Source exporter complete SHA:
`b533f3275da2d747cd3eea8f1993e57e73fc36b69d245c37ac95db816504e8d6`.
Aggregate: `actual-launches/peer-newseed-prefix-conflict-aggregate.json`.
Diagnostic source: `reproduction/operations/qwen7b_warmup/peer_newseed_prefix_diagnostic.py`.

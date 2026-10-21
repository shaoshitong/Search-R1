# Head-LR-only candidate: fixed update 768

The fixed checkpoint and evaluation passed the independent typed audit in job 743809’s CPU-only allocation step. The audit checked the full checkpoint SHA, all nine state components, original four-rank RNG archive, exact question/trajectory identities, own strict-EM labels, direct-count Mixed score and aggregate metrics. Configuration remains `71c3ee90…`; head Adam LR is 1e-4 and legacy LR is 5e-6.

| Update | Mixed BA | Mixed recall | NonMixed recall | AUC | True Mixed in Top512 |
|---:|---:|---:|---:|---:|---:|
| 256 | 50.00% | 0.00% | 100.00% | 0.552572 | 188/512 (36.72%) |
| 512 | 50.00% | 0.00% | 100.00% | 0.561871 | 195/512 (38.09%) |
| 768 | 50.00% | 0.00% | 100.00% | 0.571551 | 190/512 (37.11%) |

At update 768, all-wrong and all-correct false-positive rates are both 0 at the fixed 0.5 threshold: the model still predicts no Mixed questions. Selected K=0…5 counts are `[226,45,38,44,63,96]`; the unchanged pool contains 663 Mixed questions out of 2048. The joint BA/Top512 improvement gate fails. Ranking AUC improved, while Top512 fell from 195 at update512 to190. This is not a successful classifier or a promoted candidate.

CP768 is 3,754,035,012 bytes with SHA256 `b08cc209dcb49032b65476bf293f1b2c005c06e17f9094b4c266c51935b1b72b`; legacy Adam is11232 and head Adam768. Its matched-binding hash is distinct from the outer typed configuration-file SHA. The finite observer subsequently measured cursor783, so training continued. Update938 remains pending. No threshold fitting, bootstrap, new inference, or final-pool access was introduced.

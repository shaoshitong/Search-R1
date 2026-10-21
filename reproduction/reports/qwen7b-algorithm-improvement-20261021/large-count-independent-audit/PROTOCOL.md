# Independent large-head node acceptance

This CPU-only auditor independently recomputes the already frozen screen on the
5120 own-newseed G5 groups. It reads metadata, completed q/K prediction rows,
and the 1.4 MB frozen float32 legacy-logit cache. It does not open head/model/Adam
payloads, hidden-feature tensors, external outcomes, or final4096 outcomes.

The exact source config is `0be60cc806a177790dce39fa28269303087a1b43ede90a439cce8a63f9e3a0fc`.
The pre-score NQ halves receipt is
`3c1c89c44bb90884a7c7ba5f5b9b5c2292f1a4547bd7e25cfa082c24eed0c9e9`.
For each actual completed epoch 0/1/2/4/8/20 and raw/EMA, independently join
strict own EM and IDs to source indices 5..9, verify finite q, exact row order,
domain, K, source SHA and completed-checkpoint metadata, then recompute BA,
TPR/TNR, separate AllWrong/AllCorrect FPR, tie-aware AUC and exact Top25%.
The producer q/K rows omit five trajectory IDs: executed GPU forward identities
remain explicitly inherited from the immutable scorer/config, rather than
misrepresented as a new independent per-row forward-ID observation.
The full ordered cache trajectory SHA and frozen own-five metadata are checked.

The baseline uses stable float64 sigmoid of the exact saved float32 Z and the
original product score; no new model forward, fitting or threshold selection.
The NQ 2560 question gate and two qid-only 1280 halves exactly follow the
predeclared screen. All 12 nodes must finish before selecting one internal
candidate. Even a passing candidate never authorizes external scoring here.

Each node acceptance is immutable and scored once. Later invocations preserve
accepted nodes, bind their unchanged completed-source receipt, and process only
new actual node files. Missing cache/node files are reported as absent, never
reported as scored. Local artifacts contain only source/hash/progress/metrics.

Five synthetic fixtures cover tied-score AUC/Top25, missing class denominators,
invalid probabilities, deterministic label-blind NQ halves, the strict internal
gate, own-EM/ID/row-order joins, and duplicate rejection. Both package and flat
remote-bundle import checks passed locally. Remote fixture qualification remains
required before the first actual node acceptance.

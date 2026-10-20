# DynRPO / Post Selection iter200 NQ grounding

Completed2026-10-20, HPC3 evaluation742551 / ACD1-16 / singleH100, Slurm COMPLETED0:0.
Original training733898, checkpoint/run cc0de51f2d68495c9664aa888074839c actor0200.
Recovered previous calculation from pinned 审查七种算法实现 (2), session01a14ab9-289c-7e30-ad26-542dea4f30ed.
Frozen v4/v5 scientific files match all original six-method recorded SHA hashes.

| Protocol | Effective originally-correct searched NQ | TC | D-Inv |
|---|---:|---:|---:|
| Previous table: changed-answer rule replay on frozen v4 | 1663 | 0.9334936861 | 0.09500901984 |
| Corrected conservative typed v5, newly generated | 273 | 0.8128205128 | 0.1501831502 |

Original NQ EM1706/3610=47.2576%; response-side parsed Search/Q3947/3610=1.0933518006.
Both procedures preserve all3610 original occurrences/order/golds and the same iter200 model.
No normal inference rerun or retrieval was needed. All raw data remain on HPC3.

Calculation: replace gold-bearing spans in actually seen truncated documents, then greedy
BF16 TP1 decode same model with context8192/new512/seed1729/terminal answer stop, no retrieval.
TC=(Direct+True+0.6Amb+0.3Hall)/(Direct+True+Amb+Hall), DInv=Hall/(True+Amb+Hall).
The requested historical changed-answer replay assigns True for changed answers (1505),
Hall for invariant original correct answers (158), Amb0. It is a diagnostic rather than a
complete official evaluator reproduction. v5 requires actual injected-answer equivalence:
True176/Amb56/Hall41, includes only273 eligible cases (16.00% of1706 correct searched),
with1433 exclusions. v4 has43 unperturbable cases. Coverage differences preclude treating
conditional TC as a full-population grounding ranking. Single measured run, no seed SD.

Both full checkpoint readbacks and independent source/prompt/classification recomputations
passed. v4 inference1663 and v5 inference273 each had zero errors/interrupted attempts.
13 required child phases exited0. Permission repair incident and optional vLLM telemetry
warning are retained in result.json; no training source/checkpoint content changed.

Initial742547 failed before decoding because four model shards were owner-only0600.
Exact4 shards were granted shared-group read0640, preserving bytes andmtime, and failure
namespace was retained. Only then was terminal-job recovery742551 submitted once.

Full source identities, manifests, exclusion reasons, SHA receipts and token totals: result.json.
Remote artifact root: /data/user/jxu957/experiments/dynrpo-nq-grounding-20261020.

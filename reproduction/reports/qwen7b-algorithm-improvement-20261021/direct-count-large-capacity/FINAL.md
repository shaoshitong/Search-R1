# Frozen-E1 head-only diagnostic: completed 20 epochs

Job **743978 completed 0:0 in 1:22:35**, with the full 351,360-trajectory native cache and 18,760 real head Adam updates. E1 block27 and its trajectory head stayed frozen: raw/buffer hashes before and after cache generation match, and actor/legacy update counts are 0. Peak allocated GPU memory was 50.67GB.

The independent auditor accepted all 12 raw/EMA nodes using the held-out questions’ own fixed newseed G5 labels. This is an **internal diagnostic**, and the question texts were part of the original warmup pool. It is not an external benchmark result.

| Epoch | Head | Fit all4 NLL | Fit primary BA | NQ held BA | NQ AUC | True Mixed in Top640 |
|---:|:---:|---:|---:|---:|---:|---:|
| 0 | raw | 1.8272 | 50.002% | 50.000% | 0.49825 | 208 |
| 0 | EMA | 1.8272 | 50.002% | 50.000% | 0.49825 | 208 |
| 1 | raw | 1.3881 | 50.031% | 50.033% | 0.56388 | 242 |
| 1 | EMA | 1.3877 | 50.018% | 50.033% | 0.56135 | 244 |
| 2 | raw | 1.3795 | 50.082% | 50.033% | 0.57112 | 246 |
| 2 | EMA | 1.3769 | 50.057% | 50.033% | 0.57033 | 246 |
| 4 | raw | 1.3714 | 50.115% | 50.033% | 0.57921 | 246 |
| 4 | EMA | 1.3663 | 50.108% | 50.062% | 0.57802 | 249 |
| 8 | raw | 1.3599 | 50.267% | 50.161% | 0.58664 | 248 |
| 8 | EMA | 1.3526 | 50.460% | 50.331% | 0.58565 | 254 |
| 20 | raw | 1.3505 | 52.348% | 52.251% | 0.58551 | 255 |
| 20 | EMA | 1.3295 | 51.570% | 51.565% | 0.59130 | 256 |

The table’s fit NLL uses all four related views per fit question; fit BA uses one precommitted primary view. The complete JSON separately preserves both NLL definitions. These views are not independent questions. Held-out metrics use exactly one own newseed G5 per question.

The same-cache frozen E1 product-q baseline on NQ has BA50.47602% and207 true Mixed questions in Top640. At epoch 20, both raw and EMA improve NQ BA and Top640 and pass the two fixed-half noninferiority checks. Count12’s declared tie order chooses EMA because its Top640 count is 256 versus raw 255. The new 36-node objective family remains incomplete, so no external candidate or external scoring is permitted.

| Epoch20 NQ head | Mixed recall | NonMixed recall | All-wrong FPR | All-correct FPR |
|:---|---:|---:|---:|---:|
| raw | 9.248% | 95.254% | 4.854% | 4.435% |
| EMA | 5.302% | 97.827% | 2.080% | 2.439% |

The recall remains low: ranking improved more than balanced classification. For EMA, the two fixed NQ halves have BA50.97177%/52.14227% and Top320 counts119/140, compared with baseline50.36056%/50.59918% and96/111. HotpotQA and pooled metrics remain separate in the complete aggregate and do not enter NQ selection.

The actual first update passed exact raw/EMA/Adam/RNG replay. An independent CPU-only step inside the original allocation verified CP0/CP1 and all four new-head components, source four-rank archive, cache/config and cursor; it did not load the old 3.75GB model. Files CP1–CP6 really exist and subsequent real updates continued. The six scheduled small checkpoints were all freshly SHA-verified after completion; their full-state reload proofs come from the immutable producer. No final-check tensor load or 50GB hidden-cache rehash was performed.

The final head-only recovery checkpoint is 8,281,973 bytes, SHA256 `c6451a2e61d54e8fdb9e40d4706bfe932fb865fbd15759c112891a2dc52878c5`; it references the preserved frozen E1 and cache. The independent 12-node snapshot is 45,317 bytes, SHA256 `9552c4666235f182f31980278d3d3e93d135c8030ed80beede3f14efec44b893`.

Forward identities are bound through the verified cache ordering and fixed own-ID manifest; the CPU auditor did not independently observe every GPU prediction’s execution IDs. Fitting used a common cached-head batch geometry, so online micro8 numerical compatibility remains a separate prerequisite. No external/final rows were opened. The allocation and finite observer have ended normally; source, cache, checkpoints and prediction rows are retained remotely.

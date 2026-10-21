# Count12 independent terminal audit

All 12 actual nodes (epochs 0/1/2/4/8/20, raw and EMA) passed independent own-newseed-G5 truth joins, K/domain identity, fixed decision and ranking arithmetic, and frozen NQ-halves metric recomputation. Every one of the 12 saved node audit receipts and the final snapshot was then freshly SHA-checked once. No model/head/optimizer payload, external prediction, or final outcome was opened.

The same-cache frozen E1 product baseline on NQ2560 is BA **50.4760%**, Mixed recall **98.1504%**, nonMixed recall **2.8016%**, AUC **0.518256**, and **207/640 (32.34375%)** selected Mixed.

| Epoch | Weights | NQ BA | Mixed recall | nonMixed recall | AUC | Mixed / Top640 | Internal node gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | raw | 50.0000% | 100.0000% | 0.0000% | 0.498248 | 208 | False |
| 0 | EMA | 50.0000% | 100.0000% | 0.0000% | 0.498248 | 208 | False |
| 1 | raw | 50.0331% | 0.1233% | 99.9428% | 0.563884 | 242 | False |
| 1 | EMA | 50.0331% | 0.1233% | 99.9428% | 0.561351 | 244 | False |
| 2 | raw | 50.0331% | 0.1233% | 99.9428% | 0.571124 | 246 | False |
| 2 | EMA | 50.0331% | 0.1233% | 99.9428% | 0.570326 | 246 | False |
| 4 | raw | 50.0331% | 0.1233% | 99.9428% | 0.579214 | 246 | False |
| 4 | EMA | 50.0617% | 0.1233% | 100.0000% | 0.578017 | 249 | False |
| 8 | raw | 50.1608% | 0.4932% | 99.8285% | 0.586637 | 248 | False |
| 8 | EMA | 50.3306% | 1.2330% | 99.4282% | 0.585651 | 254 | False |
| 20 | raw | 52.2511% | 9.2478% | 95.2544% | 0.585508 | 255 | True |
| 20 | EMA | 51.5647% | 5.3021% | 97.8273% | 0.591304 | 256 | True |

E20 raw and EMA pass the frozen Count12 internal gate. EMA is Count12's provisional internal selection by Top640 first (256 versus raw 255), with BA 51.5647%. Its fixed halves have BA 50.9718%/52.1423% and Top320 Mixed 119/140, versus baseline BA 50.3606%/50.5992% and 96/111. Low Mixed recall remains a material limitation. Ranking gain is a heldout internal observation relative to the identical frozen E1 cache, not proof of superiority to the stronger external E2/E4 probes.

The later, transparently preregistered Count/binary/three-event **36-node family remains incomplete**. This Count12 provisional selection does not authorize external scoring, select the family winner, or complete the overall goal. Both external and final outcomes remain unopened.

Producer prediction rows lack explicit trajectory IDs. The independent audit binds their truth to the frozen actual own-five IDs and cached 351,360-trajectory lineage; actual GPU forward IDs remain inherited from the immutable producer/config binding, rather than independently observed in each prediction row.

Final snapshot: 45,317 bytes, SHA256 `9552c4666235f182f31980278d3d3e93d135c8030ed80beede3f14efec44b893`. The final existing-receipt readback did not invoke a new forward or fit. One redundant deploy-helper `run` call subsequently failed its wrapper receipt idempotence check because its deterministic result filename also contains a variable execution-duration field. It did not alter the accepted node audits or snapshot; no rerun was needed. Use existing proofs/status when all 12 nodes are complete.

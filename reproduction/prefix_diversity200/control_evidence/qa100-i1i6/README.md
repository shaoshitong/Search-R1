# Actor100 dev512 isolated QA

Prepared for run `4a1cc2d500fe40979ef8ca4792959438`, actual complete joint100 SHA `b1db2307b6a4170f0ab91b7aeb6d037f99ee8e7c6a000673794b907e7d2f00b4`, actor/probe2000 Adam per rank. Native QA100 exited1; its evidence remains on retained18.

Reuse accepted QA implementation `2b0a9f7b809d63880dcb0e30bebc96ab11267f45`: 8 TP1 BF16 workers, the same fixed512/strict scorer and interaction contract, full8GPU FP16 FAISS/E5 loopback on authorized same-cluster h100-11. Model/data/index stay on shared retained18 storage; only code/config/small receipts are local. Existing40GiB initial/12GiB runtime headroom and36GiB own-memory guards remain.

This commit contains configuration preparation only. Deployment, actual progress, scores, independent saved-output acceptance and owned-unit cleanup must be recorded separately. Original3433/frozen33ef training and native grounding100 remain untouched.

Actor50 fixed512 QA deployment

Actual training job3433 completed50 outers; actor and probe each1000 Adam/rank. Frozen trainer33ef232e605f7864086d1685d9a4d8ea0d67f4ab remains unchanged. Native milestone QA failed rank2 CUDA OOM; its original logs stay retained.

The independently reviewed 2b0a9f7b809d63880dcb0e30bebc96ab11267f45 isolated evaluator was deployed once to h100-11 with eight TP1 workers, same-node FP16 FAISS/E5 and retained18 NFS actor50. No model/data/index transferred through Mac. Deployment/submission receipts are not QA completion. Actual scores, saved-raw acceptance and owned cleanup remain pending at this publication.

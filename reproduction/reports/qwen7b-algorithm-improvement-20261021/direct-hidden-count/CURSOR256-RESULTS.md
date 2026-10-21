# Direct hidden count head — accepted update256

The actual same-HF fixed2048 score is scientifically negative at this early milestone. The original938-update experiment continues unchanged. This is the first supervised evaluation of the new six-class head, not a final or untouched-confirmation result.

| Metric | Update256 |
|---|---:|
| Mixed balanced accuracy at fixed0.5 |49.8917%|
| True Mixed recall |0.0000%|
| NonMixed recall |99.7834%|
| AllWrong false-positive rate |0.0948%|
| AllCorrect false-positive rate |0.6061%|
| Mixed ROC AUC |52.6156%|
| True Mixed among selected512 |171/512 =33.3984%|

The2048 pool contains663 Mixed questions (32.3730%). Selected K0..5 counts are `[249,42,37,36,56,92]`. At the preregistered0.5 threshold the head currently calls every true Mixed question NonMixed and falsely calls three NonMixed questions Mixed. Top512 and AUC also give no improvement: accepted same-HF E2 BA is56.2505%, and E4 Top512 is196. Both required joint-frontier conditions fail here. No threshold or selection rule was tuned to these results.

The independent CPU audit joined the actual fixedG5 IDs and own strict EM, verified six-class probabilities and `q_direct=sum P(K=1..4)`, recomputed all metrics, freshly hashed and reloaded the full nine-state CP256, and checked evaluation did not change the training states. This uses the direct count head; the five-trajectory probability-product audit would be the wrong check.

CP256:3754035012 bytes, SHA `1d5cbb7cfbc42e5fc91fc7f1430c65d0f8c19476b4e4069be424f96464276850`. Independent audit:5001 bytes, SHA `ccf56fac722568392a77e2c1342daf03bc6397b4236fbb1cec088139e61dfd31`. Observed checkpoint-publication to evaluation-complete file interval:87.6612 seconds. Read-only observations confirmed512 then2048 finished questions, continued owned-process I/O/CPU, no failure, and updates after256.

The original2h TimeLimit increase was denied. Exact continuation source31e930 passed its four actual remote CPU fixtures, but no future checkpoint or continuation GPU launch is claimed. The current writer remains unchanged; a resource handoff must use the highest actual saved nine-state checkpoint and terminal/lock/intent gates.

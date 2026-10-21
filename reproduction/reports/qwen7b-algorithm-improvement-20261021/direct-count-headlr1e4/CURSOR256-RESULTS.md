# Matched count-head LR1e-4: accepted update256

At the fixed0.5 threshold the candidate still classifies every true Mixed
question as NonMixed. The new-head LR increase has improved ranking relative
to the matched LR5e-6 control at256, but this milestone fails the scientific
BA/Top512 gate. The unchanged938-update single-pass experiment continues.

| Same fresh-G5 HF2048 metric | Control headLR5e-6 | Candidate headLR1e-4 |
|---|---:|---:|
| Mixed balanced accuracy |49.8917%|50.0000%|
| Mixed recall |0.0000%|0.0000%|
| NonMixed recall |99.7834%|100.0000%|
| AllWrong FPR |0.0948%|0.0000%|
| AllCorrect FPR |0.6061%|0.0000%|
| Mixed AUC |52.6156%|55.2572%|
| Top512 true Mixed |171/512 (33.3984%)|188/512 (36.7188%)|

The pool has663/2048 true Mixed questions (32.3730%). Candidate selected K0..5
counts are `[196,41,29,36,82,128]`. The matched increase is17 selected Mixed
questions and2.6416 percentage points of AUC; Mixed recall remains zero. The
accepted same-HF E2 BA56.2505% and E4 Top512196 remain unexceeded. Diagnostic512
BA is50%, Mixed recall0%, NonMixed recall100%, AUC54.0415%; this does not turn
the result into a success. No threshold, label selection or bootstrap was fitted.

The candidate's exact new namespace/config71c3… and head Adam LR1e-4 were checked,
with legacy LR5e-6 unchanged. Independent CPU work in the existing allocation
freshly hashed/reloaded the complete9-state CP256, joined actual fixed five
trajectory IDs/own EM, verified six count probabilities and their direct Mixed
sum, and recomputed every reported metric. Evaluation preserved all nine states
and the historical four-rank archive. Full CP256 is3,754,035,012 bytes, SHA
`dc57d7acff277b043bf3960a183bec0dcad25dc75ca0c8d6be41867dd776d9bf`.
The independent receipt is5186 bytes, SHA
`05375d0645fdb9d94ddc8e67d2ca753027533a3c933ec873ffeacaba7537f502`.

The first external CPU audit wrapper stopped before executing the audit due to
a generated newline-string SyntaxError. Its original controller/log remain.
The published c8457521 fix uses chr(10), compiles the generated wrapper before
launch and allows the independently namedv2 attempt only after the known prior
PID has exited. That audit completed exit0; the training runtime/data/GPU job
were untouched. Actual accepted updates continued through288 after the node.
No final pool was read and no new GPU was submitted.

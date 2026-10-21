# Direct hidden count head — accepted update512

Same-HF fixed2048 remains below both preregistered scientific frontiers. The original938-update experiment continues unchanged; no extra bootstrap, threshold fitting, selection change or new evaluation pool was used.

| Metric | Update256 | Update512 |
|---|---:|---:|
| Mixed balanced accuracy, fixed0.5 |49.8917%|49.9639%|
| True Mixed recall |0.0000%|0.0000%|
| NonMixed recall |99.7834%|99.9278%|
| AllWrong false-positive rate |0.0948%|0.0000%|
| AllCorrect false-positive rate |0.6061%|0.3030%|
| Mixed ROC AUC |52.6156%|54.3864%|
| Selected true Mixed |171/512|177/512|

At512, all663 true Mixed questions are classified NonMixed. Exactly one false Mixed prediction comes from330 AllCorrect questions; none comes from1055 AllWrong questions. The selected512 contain K0..5 counts `[220,37,39,37,64,115]`, hence177 Mixed (34.5703%) versus663/2048=32.3730% in the pool. This weak ranking result does not beat accepted same-HF E4 Top196; BA does not beat same-HF E2 56.2505%. The original diagnostic512 gives BA50.0000%, AUC54.4387%, with0 Mixed recall.

Independent engineering acceptance passed: actual fixedG5 IDs and own strict EM, six-class normalization and direct `q=sum P(K=1..4)`, independently recomputed metrics, unchanged training nine-state hashes through evaluation, full checkpoint SHA and CPU reload, both Adam clocks10976/512, and the original four-rank historical archive. This is not a claim of scientific improvement.

CP512:3754035012 bytes, SHA `16f48ff1994c9934a6034e12a98490e3a999153d271825a03826fb2360c3e13f`. Independent audit:4967 bytes, SHA `4b7177271c89caa13fee3e3fcdb5037bbe5530a5dfea4f3f3d9185a318541996`. Owned-allocation read-only observations showed512 then2048 finished rows, positive I/O/CPU and no failure. The actual checkpoint-publication-to-evaluation-complete interval is recorded in `actual-evaluation-512-duration.json`.

Current job743619 remains the only writer. The independent exact continuation operator is CPU-qualified, with no continuation submitted while the source is active. Its post-exit gate requires the highest actual accepted cursor, complete nine-state checkpoint, matching intent/progress/data position, terminal source and free lock; no rollback or optimizer reset is permitted. The published safety fallback can request finish-and-save before768 if only5..10min remain.

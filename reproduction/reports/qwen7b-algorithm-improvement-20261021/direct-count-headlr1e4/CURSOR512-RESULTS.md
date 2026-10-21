# Matched headLR1e-4: accepted update512

The fixed0.5 Mixed recall remains zero. Ranking improves relative to update256,
but the result still fails the pre-existing BA/Top512 joint gate; no threshold
is tuned and the original938-update run continues.

| Same fresh-G5 HF2048 metric | Update256 | Update512 |
|---|---:|---:|
| Mixed BA |50.0000%|50.0000%|
| Mixed recall |0.0000%|0.0000%|
| NonMixed recall |100.0000%|100.0000%|
| AllWrong FPR |0.0000%|0.0000%|
| AllCorrect FPR |0.0000%|0.0000%|
| Mixed AUC |55.2572%|56.1871%|
| Top512 true Mixed |188/512 (36.7188%)|195/512 (38.0859%)|

Pool Mixed is663/2048 (32.3730%). Selected K0..5 is
`[198,45,33,46,71,119]`. All663 real Mixed questions receive scores below0.5;
all1385 NonMixed do too. The direct count head's six probabilities and their
Mixed sum, own fixed IDs/EM, full checkpoint SHA, full9-state CPU reload, two
Adam learning rates and evaluation state preservation passed independent audit.
This is exact candidate namespace/config71c3…, not the original control audit.
Diagnostic512 BA49.8371%, Mixed recall0%, NonMixed recall99.6743%, AUC55.7543%;
one AllWrong question is falsely called Mixed there.

The independent receipt is5284 bytes, SHA
`eb9ec05b328f70fdd3012cb46a424a193c44c78fef5f4fc083289bdbc1793f45`.
Actual Adam clocks are legacy10976/head512. Further actual updates reached528
with no failure. No bootstrap, extra candidate, external tuning or final pool
access occurred. The separate frozen-E1 large-cohort diagnostic does not modify
this training source, dataset or938-step budget.

# Full169028 L15 Prefix80 MSE — first completed epoch

Actual job750190 (second seed) completed1321 raw/Adam/EMA updates, saved its E1 boundary metadata, completed the fixed diagnostics, and proceeded into E2. No failure files. Seed1 job750189 remains pending; this result does not choose the family checkpoint.

| Fixed pool | State | Mixed BA | True Mixed@512 | Variance MSE |
|---|---|---:|---:|---:|
| Train B,2048/888Mixed | raw | 66.6663% | 367/512 (71.6797%) | 0.00738013 |
| Train B,2048/888Mixed | EMA | 62.2655% | 316/512 (61.7188%) | 0.00858189 |
| Viewed development,2048/663Mixed | raw | 66.0406% | 286/512 (55.8594%) | 0.00714388 |
| Viewed development,2048/663Mixed | EMA | 60.5228% | 249/512 (48.6328%) | 0.00850443 |

The development set has been repeatedly viewed and is not final generalization evidence. Fixed train B differs from historical E1 train A; do not compare their training scores directly. The blind4096 remains untouched. Actual CPU checkpoint acceptance is deferred until the whole job completes. No claim of GPU restore consistency.

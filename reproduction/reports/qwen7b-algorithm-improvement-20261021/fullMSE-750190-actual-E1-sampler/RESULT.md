Actual full MSE E1 natural G5 sampler

Job750190 / init19137 / plan2026102302; actual epoch1 plan, no plan regeneration. 169,028 unique questions and 845,140 distinct selected own trajectories. Every selected label/ID matches canonical G10 and the physically accepted fixture.

| G5 class | Actual questions | Actual fraction | Exact uniform5-of10 expectation |
|---|---:|---:|---:|
| all_wrong | 77,652 | 45.940318% | 77632.738 |
| Mixed | 67,169 | 39.738387% | 67157.016 |
| all_correct | 24,207 | 14.321296% | 24238.246 |

Among 86,300 G10-Mixed questions, this actual G5 contains 19,131 homogeneous hard negatives: 9,367 all-wrong and 9,764 all-correct. Exact hypergeometric expectation is 19142.984, residual-0.111 standard deviations. These negatives are 11.3182% of all questions / 22.1680% of G10-Mixed questions.

| K10 correct | Questions | G5 all-wrong | G5 Mixed | G5 all-correct | Expected homogeneous |
|---:|---:|---:|---:|---:|---:|
| 0 | 68,285 | 68,285 | 0 | 0 | 68285.000 |
| 1 | 12,888 | 6,407 | 6,481 | 0 | 6444.000 |
| 2 | 9,119 | 2,049 | 7,070 | 0 | 2026.444 |
| 3 | 7,981 | 681 | 7,300 | 0 | 665.083 |
| 4 | 7,625 | 192 | 7,433 | 0 | 181.548 |
| 5 | 7,727 | 38 | 7,657 | 32 | 61.325 |
| 6 | 8,267 | 0 | 8,088 | 179 | 196.833 |
| 7 | 9,137 | 0 | 8,376 | 761 | 761.417 |
| 8 | 10,698 | 0 | 8,323 | 2,375 | 2377.333 |
| 9 | 12,858 | 0 | 6,441 | 6,417 | 6429.000 |
| 10 | 14,443 | 0 | 0 | 14,443 | 14443.000 |

Question order is shuffled. Trajectory-index inclusion counts range 84,070–84,923, expected84,514 each; standardized deviations range-2.160–1.990. Empirical counts are consistent with natural sampling; a single plan does not by itself prove random-number independence.

Plan rows digest 23f931ee5e8923399c35b3469a2ee8c72ce190c15f0ee1be2506549c7530a28c; plan fileSHA 3c0dd27fb666738459e35cf4b02baa86001c86ca9c762280268551a37c9a32bf. Small resultSHA 8824e5d4fcf3c937520a1622c03db7d91c20604e59e6029c6e594f9407096172. Analysis6.36sec, peak970MB. Independent readback confirms result unchanged and analysis process exited cleanly; training continued to actual rank cursors 754/754.

No checkpoint, tensor, RAW rollout, final4096, sampler regeneration, training mutation or Slurm change.

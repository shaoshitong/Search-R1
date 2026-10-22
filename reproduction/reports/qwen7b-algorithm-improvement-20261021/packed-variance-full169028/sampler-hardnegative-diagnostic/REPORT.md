# Full own-G10 sampler diagnostic

Random M uses the exact published seed 2026102227 and sampler functions; all five reconstructed plan hashes match the actual completed MAE epoch coverage. Fixed G5 is a CPU-only paired metadata control using the published `view(fixed_m=5)` function, identical seed and question order. It is not a claim that a new training run was submitted.

Every epoch contains all169,028 question IDs exactly once, with five distinct epoch question orders. Labels are computed from only the selected own trajectories. No G10 Mixed label is substituted for a homogeneous selected subset.

| Epoch | Sampling | Mixed | All wrong | All correct | G10 Mixed→all wrong | G10 Mixed→all correct | G10 Mixed→homogeneous |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | random_G5 | 67187 (39.75%) | 77561 | 24280 | 9276 | 9837 | 19113 |
| 1 | random_M | 72084 (42.65%) | 75267 | 21677 | 6982 | 7234 | 14216 |
| 2 | random_G5 | 67184 (39.75%) | 77577 | 24267 | 9292 | 9824 | 19116 |
| 2 | random_M | 72151 (42.69%) | 75167 | 21710 | 6882 | 7267 | 14149 |
| 3 | random_G5 | 67229 (39.77%) | 77619 | 24180 | 9334 | 9737 | 19071 |
| 3 | random_M | 72043 (42.62%) | 75208 | 21777 | 6923 | 7334 | 14257 |
| 4 | random_G5 | 67099 (39.70%) | 77595 | 24334 | 9310 | 9891 | 19201 |
| 4 | random_M | 71969 (42.58%) | 75309 | 21750 | 7024 | 7307 | 14331 |
| 5 | random_G5 | 67014 (39.65%) | 77714 | 24300 | 9429 | 9857 | 19286 |
| 5 | random_M | 72150 (42.69%) | 75090 | 21788 | 6805 | 7345 | 14150 |

The G10 pool has86,300 Mixed questions. RandomG5 keeps roughly19,100–19,300 G10-Mixed questions per epoch as legitimately homogeneous hard negatives; randomM retains roughly14,100–14,300. These samples are correctly labeled negative for their selected subset, not relabeled positive because their full G10 was Mixed.

Reversing selected trajectory order leaves each group target unchanged for all questions in both sampling protocols. This is a metadata target identity check; model prediction permutation invariance was not tested.

Existing P80 availability is recorded separately in `actual-P80-fixture-availability.json`: 8192ownG10 training questions plus the same2048fixedG5 diagnostic questions, four existing native causal token shards,92,160 trajectories. The native fixture stores real causal segments up to80, not a packed `policy80` projection. No projection or new rollout was performed. Its dev token subset has only5 trajectories/question; randomM training is available, while devM>5 requires CPU extraction of the other5 from existing full journals. Compact full169k P20 data cannot reconstruct missing P80 tokens.

No checkpoint, feature tensor, final4096, GPU/Slurm mutation or new generation was involved. Token shard existence/size was freshly checked against their sealed refs; this availability audit did not reread their token payload SHA.

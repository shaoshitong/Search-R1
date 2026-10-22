# Natural G5 control: accepted L21 result and partial L27 result

Original job 749459 (L21/P20) completed all three epochs and 192 updates with Slurm `COMPLETED 0:0`. The independent CPU audit loaded the actual final 3,770,208,838-byte checkpoint, checked its SHA256, separate finite raw/EMA native block and head parameters, all Adam moments and 192-step clocks, and both rank RNG records. The checkpoint SHA256 is `36347e600b174970d72ea5a9fe81a2874b098f097b4749cb122a04211b73b287`. CUDA RNG bytes are present; a GPU resume replay was not performed or claimed.

The original job 749460 (L27/P20) was still running at the last successful read, with both ranks at update 128. E0/E1 evaluations were available, and E2 was being saved/evaluated. Its final completion and checkpoint acceptance are not yet established. Later control reads failed intermittently; one authenticated login could not find the formerly verified `/data/modulefiles` and experiment paths. That is an infrastructure-read incident, not evidence that the experiment files were deleted or the training failed. No missing remote directories were created, and neither job was changed or resubmitted.

The sampling intervention is label blind: each question's five trajectories are drawn uniformly from its own ten before the five true EM labels determine the target. It preserves 851/780/825 formerly excluded G10-Mixed-to-G5-homogeneous cases across the three epochs. All other question IDs, epoch order, online actor prefilling, architecture, initialization, LR, EMA, optimizer budget and fixed evaluation inputs match the conditional-G5 control. Initial evaluation logits are exactly equal for both layers, both evaluation pools and both raw/EMA models.

The following are **previously viewed independent 2048-question validation results**, paired using identical questions, own G5 trajectories and labels. They are not final 4096 scores. No final 4096 labels, scores or aggregates were read.

| Layer/model | Epoch | Natural BCE | Natural BA | Natural true Mixed in top512 | Conditional BCE | Conditional BA | Conditional true Mixed in top512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L21 raw | 1 | 0.64719 | 50.34% | 189 (36.91%) | 0.64588 | 51.32% | 198 (38.67%) |
| L21 raw | 2 | 0.70388 | 56.14% | 193 (37.70%) | 0.73528 | 55.52% | 210 (41.02%) |
| L21 raw | 3 | 0.71141 | 56.50% | 204 (39.84%) | 0.71192 | 57.23% | 198 (38.67%) |
| L21 EMA | 1 | 0.75051 | 50.22% | 169 (33.01%) | 0.74730 | 50.40% | 177 (34.57%) |
| L21 EMA | 2 | 0.71041 | 53.66% | 188 (36.72%) | 0.70553 | 55.12% | 201 (39.26%) |
| L21 EMA | 3 | 0.68514 | 55.67% | 198 (38.67%) | 0.68501 | 56.34% | 207 (40.43%) |
| L27 raw | 1 | 0.64795 | 50.56% | 182 (35.55%) | 0.64432 | 50.72% | 181 (35.35%) |
| L27 EMA | 1 | 0.70813 | 51.52% | 176 (34.38%) | 0.71278 | 52.06% | 175 (34.18%) |

L21 raw epoch 3 gains six Mixed questions against the same conditional epoch, but its best natural top512 count remains below the conditional epoch-2 count (204 versus 210). L21 EMA does not improve over its matched control. Thus this fix restores the missing natural negative examples but the completed L21 pilot does not demonstrate a consistent ranking improvement. L27 requires final receipts before a conclusion.

Exact fixed-training and independent-validation results are in the paired metric summaries; `actual-CPU-checkpoint-audit-L21.json` contains the actual checkpoint and remote audit-file hashes. Large tensors and all model/data artifacts remain remote.

# Completed natural G5 paired control

Original jobs 749459 (L21/P20) and 749460 (L27/P20) both completed with Slurm COMPLETED 0:0. Each trained three epochs, 192 real optimizer updates, global batch128 with two GPUs and eight accumulation microbatches. Actual total elapsed times were 10:20 and 12:40 respectively. No job was resubmitted or changed by this observer.

The only intervention was label-blind uniform selection of five of each question’s own ten trajectories before deriving the real Mixed label. It restored 851/780/825 G10-Mixed→G5-homogeneous examples across the three epochs. All 8192 questions, epoch question order, frozen actor, native block/head initialization, online causal prefilling, LR5e-6, EMA0.99, optimizer budget, fixed training2048 and previously viewed independent validation2048 inputs match the conditional control. All eight initial raw/EMA layer/pool comparisons have exactly zero maximum logit difference.

The tables compare saved predictions on identical question IDs, own G5 trajectories and labels. No final4096 labels, prediction scores or aggregates were read. These are previously viewed validation results, not fresh final-test results.

## Previously viewed independent validation2048

| Layer/model | Epoch | Natural BCE | Natural Mixed BA | Natural true Mixed top512 | Conditional BCE | Conditional Mixed BA | Conditional true Mixed top512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L21 raw | 1 | 0.64719 | 50.34% | 189 (36.91%) | 0.64588 | 51.32% | 198 (38.67%) |
| L21 EMA | 1 | 0.75051 | 50.22% | 169 (33.01%) | 0.74730 | 50.40% | 177 (34.57%) |
| L21 raw | 2 | 0.70388 | 56.14% | 193 (37.70%) | 0.73528 | 55.52% | 210 (41.02%) |
| L21 EMA | 2 | 0.71041 | 53.66% | 188 (36.72%) | 0.70553 | 55.12% | 201 (39.26%) |
| L21 raw | 3 | 0.71141 | 56.50% | 204 (39.84%) | 0.71192 | 57.23% | 198 (38.67%) |
| L21 EMA | 3 | 0.68514 | 55.67% | 198 (38.67%) | 0.68501 | 56.34% | 207 (40.43%) |
| L27 raw | 1 | 0.64795 | 50.56% | 182 (35.55%) | 0.64432 | 50.72% | 181 (35.35%) |
| L27 EMA | 1 | 0.70813 | 51.52% | 176 (34.38%) | 0.71278 | 52.06% | 175 (34.18%) |
| L27 raw | 2 | 0.70672 | 54.11% | 193 (37.70%) | 0.72624 | 54.62% | 187 (36.52%) |
| L27 EMA | 2 | 0.68336 | 53.26% | 175 (34.18%) | 0.68340 | 53.04% | 181 (35.35%) |
| L27 raw | 3 | 0.71988 | 55.08% | 202 (39.45%) | 0.77607 | 53.88% | 200 (39.06%) |
| L27 EMA | 3 | 0.67518 | 54.24% | 191 (37.30%) | 0.67493 | 53.90% | 198 (38.67%) |

## Fixed training2048 diagnostic

| Layer/model | Epoch | Natural BCE | Natural Mixed BA | Natural true Mixed top512 | Conditional BCE | Conditional Mixed BA | Conditional true Mixed top512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L21 raw | 1 | 0.67308 | 51.67% | 288 (56.25%) | 0.66790 | 52.63% | 294 (57.42%) |
| L21 EMA | 1 | 0.71872 | 50.22% | 238 (46.48%) | 0.71632 | 50.25% | 246 (48.05%) |
| L21 raw | 2 | 0.67282 | 59.96% | 313 (61.13%) | 0.67700 | 59.21% | 318 (62.11%) |
| L21 EMA | 2 | 0.69066 | 56.01% | 289 (56.45%) | 0.68525 | 57.05% | 298 (58.20%) |
| L21 raw | 3 | 0.65714 | 63.37% | 326 (63.67%) | 0.64459 | 63.92% | 332 (64.84%) |
| L21 EMA | 3 | 0.66851 | 60.12% | 304 (59.38%) | 0.66191 | 61.90% | 316 (61.72%) |
| L27 raw | 1 | 0.67627 | 50.55% | 287 (56.05%) | 0.67530 | 50.75% | 282 (55.08%) |
| L27 EMA | 1 | 0.70021 | 51.23% | 235 (45.90%) | 0.70226 | 51.08% | 237 (46.29%) |
| L27 raw | 2 | 0.67925 | 58.57% | 298 (58.20%) | 0.67987 | 59.34% | 308 (60.16%) |
| L27 EMA | 2 | 0.68174 | 56.58% | 280 (54.69%) | 0.68042 | 56.16% | 272 (53.12%) |
| L27 raw | 3 | 0.66741 | 62.18% | 333 (65.04%) | 0.68165 | 59.01% | 337 (65.82%) |
| L27 EMA | 3 | 0.66809 | 59.61% | 298 (58.20%) | 0.66256 | 59.22% | 303 (59.18%) |

## Interpretation

The completed L21 natural run reaches its best top512 count of 204/512 (39.84% raw E3), below conditional L21’s 210/512 (41.02% raw E2). L27 natural raw E3 reaches 202/512 (39.45%), versus conditional E3’s 200/512 (39.06%); its EMA top512 result is lower (191 versus198). The natural sampling fix therefore does not show a consistent ranking gain in this pilot.

At L27 raw E3, natural sampling reduces all-correct false alarms from 299/330 (90.61%) to253/330 (76.67%), and all-wrong false alarms from828/1055 (78.48%) to678/1055 (64.27%). It also increases real-Mixed misses from72/663 (10.86%) to150/663 (22.62%). This explains the modest balanced-accuracy improvement (53.88%→55.08%) while ranking barely changes. The full classwise counts and BCE, confusion matrices, recall, false-positive rates and paired deltas are preserved in actual-matched-metric-summary-complete.json.

## Checkpoint acceptance

- L21: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-natural-g5-8192-20261022-v1/layer21-prefix20/checkpoint-epoch-03.pt`, 3770208838 bytes, SHA256 `36347e600b174970d72ea5a9fe81a2874b098f097b4749cb122a04211b73b287`.
- L27: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-natural-g5-8192-20261022-v1/layer27-prefix20/checkpoint-epoch-03.pt`, 3770208838 bytes, SHA256 `397b797cd08d9cc8fde65b6633ba4ca919081840997d04fefc60b17a76742849`.

Both actual final checkpoints fully deserialized on CPU. Raw and EMA are separate, finite and different in 46 native/head tensors; all Adam moments are finite and all clocks equal192; both rank Python/NumPy/CPU torch RNG states load correctly and CUDA RNG bytes are present. A GPU resume next-update replay was not performed or claimed. All checkpoint tensors, optimizer state, actor/model assets and input data remain remote.

An HPC3 login-node restart briefly interrupted read access and shared-mount visibility. The jobs continued on their compute nodes; fresh accounting, final receipts and full checkpoint SHA acceptance after the incident establish completion. Missing remote paths were never recreated.

The independent matched report has status NATURAL_VS_CONDITIONAL_G5_MATCHED_METRICS_COMPLETE, zero missing files and eight complete paired layer/epoch combinations. Its remote SHA and path are in actual-matched-comparison-complete.json.

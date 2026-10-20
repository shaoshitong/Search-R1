# 7B native probe warmup: epoch1 accepted

Only the probe layer27 block and Linear are trained on frozen official7B layer26 features. Prefix5 fit; same-trajectory Prefix20 is retained. These accuracies classify trajectory correctness; they are not actor QA EM.

Full original train169028 questions/845140 trajectories and independent diagnostic512/2560 were evaluated without resampling.

| Split | Epoch | Trajectory BCE | Mixed BCE | Joint loss | Trajectory accuracy | Mixed balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| train | 0 | 0.863486 | 1.251443 | 2.114929 | 46.09% | 49.43% |
| train | 1 | 0.686821 | 0.891335 | 1.578156 | 66.12% | 59.63% |
| diagnostic | 0 | 0.831646 | 1.221775 | 2.053420 | 47.73% | 49.44% |
| diagnostic | 1 | 0.755895 | 0.933344 | 1.689239 | 62.73% | 54.67% |

Epoch1 completed4573 genuine Adam updates. The balanced fit used233175 G5 groups (169028 unique;64147 repeated whole groups). Actor updates remain0.
All epoch1 prediction identity/own EM/p_j/Mixed formula and confusion metrics were independently accepted; the full checkpoint was SHA/CPU/raw/EMA/Adam/four-rank RNG/shape/source/config/bank accepted.
Checkpoint: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/probe/epoch-0001.pt
SHA256: a6a7a255588b50d65860710c513203de341a8b7f686b9ecbf18717bb6dca716b
Acceptance: /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/preparation/epoch1-acceptance.json
Fit source:96df5a23f23f6da0f24e79e67dcf0e3039f22de0. Auditor source:fe0e65058ff3dba9deec831f9fc8dec18153caea.

Epoch2 is running. The20-epoch goal and final EMA selection are not complete.

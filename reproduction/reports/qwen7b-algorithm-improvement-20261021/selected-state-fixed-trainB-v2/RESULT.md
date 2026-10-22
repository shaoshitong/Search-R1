# Unified fixed trainB scoring and E1 adapter: actual completion

Root-owned job750334 completed0:0 onACD1-20 in1m58s. Sourcev2ab8060368cc49c1313493463642350dd5912dc0e. Both ranks finished1040/1040 genuine groups (1024trainB+16fixed viewed-dev replay each), forward scoring32.62/34.56s after loading. No failure receipt. The observer saw loading, completed coverage, and final completion/accounting; it does not claim two intermediate progress-growth snapshots that were not sampled.

E1 same-CP first32 sorted existing dev ownG5 replay passed: maximum probability difference5.2098113711984695e-08, mean1.966877333052508e-08, fixed tolerance0.001. Legacy strict_em schema was normalized with exact ordered trajectory/EM identity. This validates adapter consistency, not new-question performance or a new model selection.

All models use fixed trainB2048ownG5:888Mixed,950all-wrong,210all-correct; SHAe6e74d92160efb0740ba5c0316c4ee7686f06a499c90cafd5a545f325b62209b. OldtrainA remains separate. Z E10/raw predictions were reused only after exact saved identities and accepted CP/source proof checks. No Z inference or rollout was repeated.

| model/state | fixed diagnostic loss | Mixed BA | Mixed recall | all-wrong FPR | all-correct FPR | Top512 realMixed |
|---|---:|---:|---:|---:|---:|---:|
| E1_EMA | direct_Mixed_BCE 0.66632308 | 55.8287% | 25.4505% | 11.3684% | 24.7619% | 292/512 (57.0312%) |
| Z_seed1 | variance_MSE(v) 0.00416854 | 82.1660% | 67.0045% | 2.1053% | 5.2381% | 500/512 (97.6562%) |
| Z_seed2 | variance_MSE(v) 0.00474452 | 79.8831% | 64.0766% | 3.2632% | 9.0476% | 490/512 (95.7031%) |

E1 uses direct Mixed probability threshold0.5; Z uses population-variance threshold0.08. Variance MSE is not BCE or probability calibration. These are training-diagnostic results, not evidence of unseen-question generalization. This job neither selected an epoch/state/threshold nor scored final4096. Actor/probe optimizer updates remain0.

[Actual terminal proof](actual-terminal-proof.json) preserves complete aggregate metrics, source/prediction hashes, paired-input identity proofs, adapter tolerance and Slurm accounting. Small loading/final observations are retained as live-*.json. The observer never loaded any CP/tensor, changed Slurm or training, or returned per-question predictions to Mac.

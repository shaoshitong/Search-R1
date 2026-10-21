# Easy2560 + Co rollout: official Qwen2.5-7B-Instruct

This is a fresh 7B actor experiment, adapted from the 3B Easy implementation at
6956d9ffa0ec497c13a66061bd6460a471e9aa0d. The original 3B checkout and live job
734689 are unchanged. No other thread is messaged or modified.

Official source: Qwen/Qwen2.5-7B-Instruct, revision
a09a35458c702b33eeacc393d103063234e8bc28. The complete existing HPC3 asset is
`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/assets/qwen25-7b-instruct-a09a35458c70`.
Its 11 files / four BF16 shards total 15,242,788,168 bytes. The exact pinned
manifest is `reproduction/easy2560_joint4/official7b.json`. No learned selection
head, teacher labels or actor checkpoint is inherited from the source thread.

## Same scientific recipe

- 2560 candidate questions, five complete original trajectories per question.
- Real EM chooses K=1..4 first, then K=5 in original order to 512. Never choose K=0.
- Reuse the selected 512x5 tokens and actual sampling log-probabilities; no second rollout.
- Same grounding-augmented GRPO, virtual0=false; no question-selection probe.
- Same 4096 total context, max four searches, 500 action/observation limit and
  500 final-answer reserve. Same temperature1/top_p1 and original retrieval.
- Four H100, 200 outer, global mini128, physical micro8/rank, logical micro4,
  accumulation4, 20 optimizer opportunities per outer. LR1e-6, fixed57-outer
  warmup, PPO epoch1, clip.2, dual clip3, grad clip1, entropy/ref-KL coefficients0.
- Same BF16 compute and FP32 master/optimizer, FSDP offload, TP1. Preserve the
  original vLLM cache ratio0.1. In this vendored worker it allocates a fraction of
  *free memory after profiling*, not a total GPU cap including model weights.

7B-specific work is identity / architecture / shard binding. The grounding
calibration recipe is unchanged: train-derived fixed300, G5, layers21/24/27,
correction feedback, two-fold grouped CV. Round0 is genuinely regenerated and
fitted using official7B, rather than inheriting incompatible3B labels/heads.
Rounds50/100/150 retain the same actor-dependent refits. All these layers exist
in the28-layer7B; features and classifier input dimensions are determined natively.

Full native4 model/Adam/scheduler/RNG/driver state is saved after every completed
outer; publish the new complete boundary before retiring the preceding ordinary
checkpoint. Preserve milestones5/50/100/150/200, fixed512 QA and final native51713
per-source/macro/weighted EM/F1, and perform actual full-state CUDA restore.
There must be no actor update201. Raw outputs/checkpoints stay remote.

## Launch and current limitations

Dedicated base: `/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021`.
The existing once-only `control submit --base BASE --commit FULL40 --actor-micro 8`
uses jxu957/emergency_acd, four GPUs,48CPUs,512GiB. Code is published and deployed
before submission. A job id or RUNNING is not an accepted training result.

The retained personal storage budget is decimal2TB. The last full Oct20
cross-account visible measurement was2,354,561,650,688 bytes (not a fresh quota
reading). New training writes cannot be claimed budget-compliant while this
remains unresolved. A new cleanup question is pending; never delete other active
jobs' assets or silently raise the budget. Preparation and publication continue.

Local focused tests cover real Hydra composition in the existing HPC runtime,
model/reference binding, native4 geometry, real7B round0 routing and pinned shards.
The inherited common-contract fixture test `test_reader_preserves_eight_logical_micro_mass_for_physical32`
already fails identically in the unchanged3B checkout because it omits tensor
hashes. This is retained as a pre-existing test issue, not relaxed in runtime.

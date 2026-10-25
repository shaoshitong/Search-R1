# CP50 objective revision and ordered handoff

User instruction: fix logical loss normalization, all-correct rewards/advantages, and fallback eligibility; resume the complete CP50 on four training GPUs. **Start the new experiment before stopping old job 751884.** The stop gate is a fully restored and committed outer51 plus observed outer52 progression. Pending allocation does not satisfy it. Keep retrieval751011 and all parent CP50 assets.

Runtime source: `7cfef2495d6e82e6a5af7a88d9029ec1c7beba70`, pushed to `codex/dynrpo7b-formal-20261023`; 2221 remotely deployed source files matched SHA. New job761991, base `/data/user/jxu957/experiments/dynrpo7b-formal-20261025-cp50-mathfix-r1`.

Changes:
- Physical micro8, logical micro4; match the original four-row masked-token mean reduction. Global optimizer budget and scheduler remain as saved.
- Disable virtual-zero GRPO. Only all-correct G5 groups receive five rewards of1 and therefore zero group advantages. Mixed retains its Soft Reward. Identify groups by uid after DP row balancing, not adjacency.
- Select Mixed first, then all-correct; never include all-wrong fallback. If eligible groups are insufficient, continue the original frozen candidate trajectories in128-question batches. Persist completed batches for recovery.
- Initial fork imports complete CP50 state only; the old run's outer51+ collection and updates are excluded. Later recovery inside the revised experiment remains supported.

Validation: local29 passed,1 skipped; remote real EcoRewardManager/GRPO integration3 passed. Actual CP50 producer metadata validated: complete boundary, four ranks at1000 Adam updates. Actual parent-budget comparison changed only logical micro2→4 and its reduction descriptor. GPU restore and first full iteration are pending; submission alone is not acceptance.

Old job continues until the stop gate is met. All outputs/checkpoints/caches stay remote. The read-only observer now accepts explicit base/training/retrieval IDs to prevent confusing the old and new experiment.

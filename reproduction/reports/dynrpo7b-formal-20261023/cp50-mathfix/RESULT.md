# CP50 objective revision: ordered handoff completed

The revised four-GPU job761991 restored the complete CP50, completed outer51, saved and verified CP51, and demonstrably advanced outer52 **before** old training751884 was cancelled. Retrieval751011 stayed running. Current allocation is4 training H100 +2 retrieval H100; temporary overlap was10 H100.

## Implemented differences

- Physical micro8, logical micro4; match the original four-row masked-token mean reduction. Global optimizer budget and scheduler continue from CP50.
- Disable virtual-zero GRPO. Only all-correct G5 groups receive five rewards of1 and zero group advantages; Mixed keeps its Soft Reward. Group by uid after DP balancing, not adjacency.
- Select Mixed first, then all-correct. Never use all-wrong fallback. If eligibility is insufficient, continue original frozen-pool trajectories in128-question batches and journal the completed batches.
- Import complete CP50 state; exclude the old branch's outer51+ collection and updates. Retain all old assets, including CP50 and the probe files referenced by the new run.

## Actual verification

Runtime source `7cfef2495d6e82e6a5af7a88d9029ec1c7beba70` was pushed and2221 deployed files matched SHA. New base: `/data/user/jxu957/experiments/dynrpo7b-formal-20261025-cp50-mathfix-r1`.

Local tests:29 passed,1 skipped. Remote real EcoRewardManager/GRPO integration:3 passed. Actual parent-budget comparison changed only logical micro2→4 and its reduction descriptor.

All four ranks restored Adam1000, scheduler50 and their RNG with equal readback. Driver RNG, source cursor243234, reward version50 and probe update50 state were restored. Raw/EMA/Adam/RNG continued from the published probe checkpoint.

Outer51 selected512 questions:487 Mixed,25 all-correct,0 all-wrong. Runtime reward metrics reported25 constant-reward groups/125 rows. Train trajectory EM was0.567 at logged precision; this is selected training data, not validation accuracy. Four ranks advanced to1020 Adam updates. CP51 passed16 actor-file metadata checks and5 fresh boundary-state SHA checks. Outer52 started; its proxy log grew213561 bytes across pre-stop observations.

Only then, at Unix1792912966.491394, old751884 was cancelled. Its last complete checkpoint was117 and rank Adam counters were2340; no unsaved actor update was discarded. Old source1960 files matched published commit `ee74bfa3c6d1141859abb19e882b65c93ea6f0ba` before release. Slurm confirmed cancellation, the writer lock was free, and CP50 remained present. After stopping old training, new outer52 advanced another294599 proxy-log bytes without a trainer error. The old branch is intentionally retired and must not be automatically resumed.

## Ongoing monitoring

The existing30-minute monitor now follows761991/new base plus retrieval751011. It preserves the three revised objective choices when recovering. Follow-up acceptance for resumed outers52–55 remains pending; only51 is currently fully accepted. Training target remains200 complete outers, all scheduled probe updates, fixed512 validation, final51713 evaluation, and resource cleanup. The overall training goal is not complete.

All tensors, checkpoints, optimizer state and rollouts remain remote. This report contains only small audit summaries.

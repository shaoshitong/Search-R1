# Retained-input cold handoff: prepared, human deferred

Latest actual human decision (October19): **保持运行，暂不重启**. The following is a prepared plan only; it must not be executed without a new explicit authorization. Current jobs continue. Decision evidence: `work/prefix5-ab-retained-handoff-human-decision-20261019.json`.

Both current A/B native jobs are healthy on h100-11. Inputs/runtime still use the original source8 private mount namespace. The shared-storage memory records the source8 service release on October20, without an exact time. The goal remains each200/no201 plus all QA/final51713 and actual resource closure.

The existing storage instruction explicitly says not to stop or hot-mount healthy training merely to apply storage policy. The latest human B-only CP50 rollback was a different, already-consumed operation and does not authorize this new two-arm stop. This plan therefore keeps current training running until an explicit cold-handoff approval, or a genuine fault requiring already-authorized recovery. No new stop/submit intent has been consumed.

Prepared artifacts:

- All53 fixed retained inputs are complete under the qualified NAS10 retained namespace; the qualified `retained-cache-v2` private wrapper preserves legacy path names without changing other tasks. Common runtime and code are retained too. Completed large hashes must not be repeated.
- Recovery runtime `d68173cea039f5949e6bd799ead7ed590ee87b71` was published, independently read back and staged as1393 runtime files under `/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/search-g1-b-progressive-code-retained-20261018-d68173cea039f5949e6bd799ead7ed590ee87b71`.
- Frozen A uses `ab_frozen_latest_resume.prepare_latest`; progressive B uses `b_progressive_resume.prepare_latest`. The existing22 focused CPU recovery cases passed. Core actor/reward/loss implementations remain byte-identical to each arm's protected scientific source.
- The new stop record requires an explicit `HUMAN_APPROVED_RETAINED_INPUT_COLD_HANDOFF` authorization bound to both exact source runs/config SHAs and each original leader PID/birth. It rejects the archived B-only stop record, a different execution and every consumed stop intent. Only the approved leader receives a pidfd SIGTERM; its own existing cleanup handles its children.

After approval, the concrete sequence is:

1. Record the actual approval/time/scope. Re-read both current native source identities, latest complete joint and current stage. Prefer completing B101 first, retaining its real40-token model/optimizer state and avoiding duplicate QA100; never fall back to101 if a later complete joint exists.
2. Stop each exact approved native leader once, preserve original inode/log SHA and actual OS-exit records, then verify only its owned process births/GPU allocations and services are released. The overall goal stays active; foreign tasks remain untouched.
3. Read the actual final latest joint after writers stop. Reuse an existing matching CPU source proof; otherwise independently hash/load only that new complete point with the staged CPU reader. Preserve actor/Adam/scheduler, raw+EMA probe/Adam/EMA count, four-rank RNG, active grounding100 and the exact question permutation/cursor/pending reservation. Unsaved updates cannot be stitched into a completed outer.
4. Prepare new NAS10 namespaces with source names suffixed `-retained-inputs-r1`; reconcile any existing intent/receipt instead of replaying. Use a fresh process with prefix5 for A and the completed stage's20/40/80 environment for B. The new processes run through the qualified retained wrapper. No warmup, official0 reset, scientific parameter or remaining-budget change.
5. Submit each prepared execution once. Bind a new passive exit recorder to its new exact leader births. Personally verify five new complete outers and the next real prefix generation in each arm, then resume the normal150/200 observer. Preserve all source runs and historical branches; models, optimizer, trajectories and full outputs remain remote.

The exact future checkpoint numbers are intentionally not fixed while live training advances. Existing source CP100 proof does not authorize rolling back a newer completed state. Actual GPU restoration/handoff/five-round acceptance has not happened yet.

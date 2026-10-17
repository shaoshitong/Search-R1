# Thread5 I1-I6 evidence and fresh restart

Scope: native Qwen periodic25 experiment. The consolidated incident report audits other runs; its findings are not automatically findings against this implementation. Original human message `01a14849-d31c-7a80-b223-6a46bc3a5a08`, 2026-10-17 05:16 UTC, authorizes stopping the old run, correcting I1-I6, fresh official-actor training, skipping accepted warmup, and scoped deletion of erroneous weights/logs.

Job725286 is CANCELLED by21930; fresh accounting and an empty account queue confirm termination. Old run `1cd20aedc5e94511be8ea8b7db76933c` completed126 outers. They do not count toward the new200 and must not be automatically resumed.

| Item | Actual implementation and disposition |
|---|---|
| I1 | Train, live QA and final_eval use GenerationConfig/configure_generation, LLMGenerationManager and common8 async_rollout_runtime. Shared context_budget plans4096 total,500 action/reserve, information-wrapper protection and early final. Strict scorer unchanged. Update obsolete CPU environment-tokenization fixtures and add wrapper/reserve regression. |
| I2 | Preserve dp_actor._run_minibatch: each local micro4 has its own masked mean, then equal gradient accumulation. No global token-mass mean. This four-rank topology is not the other experiment's eight-rank micro2 topology. |
| I3 | Preserve ray_trainer checkpoint/refresh_after_checkpoint and reward-grounding rounds0/50/100/150. Separate from periodic25 question probe. Fresh actor may reuse only verified pristine R10 round0, with all original model/data/producer/CV/head/dump bindings; never old run round100. |
| I4 | Preserve before/after-minibatch monitor, existing KL/tail calls, finite checks, dualclip and gradclip. No new runtime risk gate. |
| I5 | Preserve native4_bf16_varlen_fp32_master: actor/ref/rollout/live QA BF16, FP32 master/reduction. Independent final_eval uses the same resolved configuration. |
| I6 | Train and QA enter the same async controller/action-cut path. Original tokens, logprobs, masks and positions remain authoritative. Selection chooses complete questions before generation; do not substitute prefix selection. |

`initial-calibration-reuse.json` distinguishes human termination of a trained run from independently pristine historical calibration reuse. The source still requires an authenticated FAILED/OOM/TIMEOUT job, pristine_student, idle locks, completed cleanup and all existing source/data/artifact SHA checks. The126-outer cancelled run is never called a zero-update failure.

Fresh run retains official aa8e725 actor, accepted warmup20/Adam400 question-probe state, refits0/25/50/75/100/125/150/175 using5120x5 and20epochs, no refit200, fixed512 QA and final51713, four GPUs, low-priority acd_u first, and no dependency on the cancelled job. CPU checks/source publication are not evidence of successful GPU training; verify five complete new outers and the next round after launch.

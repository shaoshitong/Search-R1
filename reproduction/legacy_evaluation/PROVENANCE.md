# Baseline512 evaluation change provenance and strict legacy isolation

Read-only Git/source audit; no worktree, runtime, state, job, or evaluation was changed. Git author metadata identifies commits, not the specific AI conversation that wrote them. No other thread was contacted.

## Introduction commits (all original timestamps UTC+08)

| Change | Introduction commit | Author date | Parent | Relationship |
|---|---|---|---|---|
| `context_budget.py`: answer reserve, early no-search final stage, bounded observation room, preserve information wrapper | [d00e4f8811cb4a2119c7d088280f7a97cd149d7e](https://github.com/shaoshitong/Search-R1/commit/d00e4f8811cb4a2119c7d088280f7a97cd149d7e) | 2026-10-14T20:38:41+08:00 | f7d9f1488d3c1373098b9401463812b24fe94b63 | Already an ancestor of user-specified upstream1f75a186; absent old3364/2de313e |
| Independent async generation plus enabling it in `_validate` (not just train) | [1f75a186eea7b769c37a919096ea0b54372e5160](https://github.com/shaoshitong/Search-R1/commit/1f75a186eea7b769c37a919096ea0b54372e5160) | 2026-10-14T23:41:37+08:00 | c10c295e465e13cd038adf8e4f4f0840a5722d91 | This is the specified upstream commit itself |
| Original four-GPU BF16 precision profile and `rollout.dtype=bfloat16` override | [7dcbc695be205e5d61db10f946efb6a3dbbf9524](https://github.com/shaoshitong/Search-R1/commit/7dcbc695be205e5d61db10f946efb6a3dbbf9524) | 2026-10-08T14:24:34+08:00 | 9f25fa4759d79569695465b352c87b95f587bb6d | Code already existed at old2de313e, but old3364 did not activate this profile |
| Native eight-GPU BF16/varlen profile | [fb1c850dd00d0e6bcca6b650ffdb5f6e28267d6f](https://github.com/shaoshitong/Search-R1/commit/fb1c850dd00d0e6bcca6b650ffdb5f6e28267d6f) | 2026-10-12T01:21:18+08:00 | 09e59943c1fcf4563d8251a4df729aa5e112b5e3 | Already inherited by upstream1f75a186; after old3364 |
| Explicit fresh H1008 BF16/localmicro4 profile admission/activation | [a8c311685af6bf2d71ba6d977515106ff164f127](https://github.com/shaoshitong/Search-R1/commit/a8c311685af6bf2d71ba6d977515106ff164f127) | 2026-10-15T01:21:47+08:00 | 8bc6bfe8c7b71e7c770e3f096d2fe15f2d01f412 | H100 branch addition after specified upstream |
| Permit existing async contract on8 engines rather than4 | [3b57edb38024bca88915b79c7f9b951aa24d781a](https://github.com/shaoshitong/Search-R1/commit/3b57edb38024bca88915b79c7f9b951aa24d781a) | 2026-10-15T13:46:18+08:00 | ec6856e5aa68cf06e79b8aa700519352516a9e94 | H100 branch adaptation; no new wrapper/context semantics |

All six commits list Git author `shaoshitong <shaoshitong@users.noreply.github.com>` and matching author/commit dates. This is not reliable evidence of which agent or human typed each change. Current containing branches include the named upstream and H100 branches; Git does not store an immutable original branch-of-authorship, so ancestor/blob evidence is stronger than claiming a unique originating branch.

## Exact locations at frozen3399 source545d4cf

- [answer reserve / early final / wrapper clipping, context_budget.py lines6–47](https://github.com/shaoshitong/Search-R1/blob/545d4cf00da9bed3a6e61f3ab286e48ceac8573c/reproduction/common8/context_budget.py#L6).
- [validation enables rollout phase at ray_trainer.py line487](https://github.com/shaoshitong/Search-R1/blob/545d4cf00da9bed3a6e61f3ab286e48ceac8573c/verl/trainer/ppo/ray_trainer.py#L487).
- [common8 async routing, generation.py line233](https://github.com/shaoshitong/Search-R1/blob/545d4cf00da9bed3a6e61f3ab286e48ceac8573c/search_r1/llm_agent/generation.py#L233).
- [BF16 override including rollout dtype, precision_contract.py](https://github.com/shaoshitong/Search-R1/blob/545d4cf00da9bed3a6e61f3ab286e48ceac8573c/reproduction/common8/precision_contract.py#L36).

`context_budget.py` Git blob40f9cf3152ace63ff75a2f3632bfac320ba2bfe4 and `async_rollout_runtime.py` blobb83657634a36e50641f95f4b02f7ad646ebb9e3b are exactly equal between upstream1f75a186 and frozen3399/545d4cf. Of context_budget/generation_behavior/async_runtime/generation routing, H100 did not change any bytes; only rollout_phase_contract world-size admission was widened.

## Generation parameters

Old and new baseline both use validation `do_sample=False`, enforced greedy temperature0/top_p1/top_k-1/n1/actionmax500. Overall training config temperature1 did not imply sampled validation. The async implementation in1f75a186 added a second equivalent neutral-greedy parameter construction for individual phase requests; this is a route/precision/context change rather than a temperature increase. Initial limits2048/4096/500/500/turn4, official generation-config asset, and raw-prefix-to-first-closing-tag behavior remain the same. The new context policy is separate and changes when environment input is clipped/search execution stops.

## Current fallback is NOT strict legacy

Setting `async_phase_enabled=False` in current545d4cf still calls the post-d00 `generation_behavior._run`, which uses newcontext protocol, wrapper clipping and earlyfinal. Simply switching off async cannot honestly be called old3364 evaluation.

## Existing independent evaluator: reusable structure, not ready-made strict CP5 evaluator

Local separate repo `/Users/shaoshitong/Documents/Search-G1/mixed-first-200`, HEAD28aa93a75d6c982673f17cc74fe853538565bcdb contains `reproduction/common8/final_resume_runner.py` and pipeline/journal. That repo's generation_behavior Git blobc3c0e941231c527fa3bdf86b5ff7c2f6c8e38856 is exactly identical to old3364/2de313e. Its runner imports `_run as original_loop` explicitly, so its loop is truly pre-d00 synchronous.

However it is hardbound toCP200/51713, bs256/cache.35, carries source dtype unchanged, and has no generic CP5/512 binding. It therefore cannot be launched unchanged and described as strict historical evaluation.

Precision trap: FinalRolloutWorker setsrole='rollout' (runner85–95); fsdp_workers.init_model lines317–323 sets an emptyfsdp_config for a rollout-only worker; `_build_model_optimizer` lines145–149 defaults non-actor HF load toBF16. Changing only vLLM dtype toFP32 can still round HF weights toBF16 before synchronization. Strict old evaluation must explicitly load HF/FSDP parameters inFP32 as well as vLLMFP32 and verify actual dtypes. No optimizer should be constructed. Use_remove_paddingFalse restores the old SDPA HF initialization path (HF forward isn't the vLLM generation engine).

## Minimal isolated eval-only implementation path (proposal, not executed)

1. Publish a distinct legacy evaluator package and namespace; pin the byte-identical old2de313e generation_behavior loop (or vendor it under a uniquely named module). Explicitly bypass current manager's async routing; do not overwrite the training module or global environment of active workers.
2. Reuse owned-process/native8GPU/retriever lifecycle and rollout-only structure from the existing final runner, adapting binding/journal to explicit checkpointstep and validation/final split/count. Never rewrite the existingCP200/final journal or labelCP5 asCP200.
3. Bind remote checkpoint HF files/hash/revision and same validationSHA. CP5/50/etc weights are read-only; optimizer/head are not loaded for answer generation. Official-zero should point to the same official weights.
4. Explicit FP32 HF loader, FP32vLLM, old greedy/stop settings and old sync batch64/cache.1/template/observation/turn rules, plus actual8UUID/readback proofs. Checkpoint save metadata may differ but effective EOS/stop fields must match original official generation config.
5. Keep asyncBF16 training unchanged. A separate evaluator needs an actual availableGPU allocation/owned runtime. Current8GPUtraining cannot silently be oversubscribed or stopped merely to fit an evaluator. Queue isolated evaluation or coordinate only within already-authorized sharing; do not heat-patch active training to switch precision.
6. Record a recipe/source hash in every result. Current inline before0/dev5 are the new recipe and must remain labeled as such. User requested the old recipe, so future official comparison requires this separate exact legacy evaluator, not a partial wrapper-only rollback.

No jobs were started, stopped, or changed during this audit. No full logs/rows/checkpoints were transferred toMac.

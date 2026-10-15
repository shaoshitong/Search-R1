# Isolated old3364 evaluator: deployment and CLI

Implementation is evaluation-only. Canonical GPU execution remains **pending**. The existing3399/3400 inline evaluations retain their original values and are **new-recipe diagnostics**, not canonical legacy results. Nothing here starts, changes, stops or hot-edits3400.

## Layout and runtime dependencies

Keep two different remote code roots:

- `OLD_CODE`: original3364 checkout `/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011/repo`. Its12 pinned files must pass `contract.verify_source_tree`; its oldretrieval server has additionally fixed SHA `b0e70e06bbd7d4d820ff7c550142fa6b22725789711f927bdc4960dcdc766ca1`.
- `CONTROL`: a newly published immutable control checkout containing `reproduction/legacy_evaluation/` and the unique top-level `legacy_eval_control/`. Bind its independent Git publication/fullSHA proof separately from oldsource2de313e. Never put these controls into active3400's frozen checkout.
- `EVAL_BASE`: a new remote execution namespace containing only evaluation configuration, process ledger, cache and remote scalar-row journals. Symlink an existing remote environment; do not copy an environment or model toMac.

The remote environment needs the original training/evaluation dependencies: Python3.10, PyTorch, Transformers, OmegaConf, PyArrow/Pandas, Ray with `runtime_env.worker_process_setup_hook`, vLLM **0.6.3**, CUDA/NCCL, existing GPUFAISS and retrieval-server packages. No local dependency installation is needed. Actual torch/Transformers versions are recorded; their historical versions have not independently been established, so no bitwise historical-result reproduction is claimed.

`legacy_eval_control.entry` explicitly extends the `reproduction` package search path, even if the oldtree has a regular `__init__.py`. The driver and everyRay worker execute the same old-source bootstrap; the Ray worker setup hook runs before actor deserialization. The worker's HFloader receives explicitFP32 config with nooptimizer, independent of the checkpointconfig dtype. ActualHF/FSDP and actualvLLM parameter dtypes, vLLM0.6.3 and scheduler values are captured after initialization.

## Remote CPU preparation: real existing checkpoint only

After the controls are committed/pushed/fullSHA verified and staged remotely, run this **CPU-only preparation** with `PYTHONPATH` containing only the newcontrol root. The helper activates oldcode internally; there is no GPU initialization or Slurm submission.

```bash
export PYTHONPATH="$CONTROL"
"$REMOTE_ENV/bin/python" -m reproduction.legacy_evaluation.prepare \
  --base "$EVAL_BASE" \
  --legacy-source-root "$OLD_CODE" \
  --control-commit "$PUBLISHED_CONTROL_COMMIT" \
  --control-publication-proof "$REMOTE_CONTROL_FULL_SHA_PROOF" \
  --checkpoint /mnt/data/shaoshitong/search-g1-qprobe3-stdloss-545d4cf-20261015-v2/checkpoints/search-g1-common8/actor/global_step_5 \
  --checkpoint-receipt /mnt/data/shaoshitong/search-g1-qprobe3-stdloss-545d4cf-20261015-v2/artifacts/question-probe/completed-pairs/global_step_5/source-receipt.json \
  --step 5 --split validation \
  --source-base /mnt/data/shaoshitong/search-g1-qprobe3-stdloss-545d4cf-20261015-v2 \
  --official-model /dataset/shaoshitong/Search-G1-common8/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725 \
  --reference-config /mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011/logs/resolved-training-config.json \
  --environment-root "$REMOTE_ENV" \
  --sharing-bindings "$ACTUAL_SHARING_JSON" \
  --gpu-lock-directory /mnt/data/shaoshitong/.search-g1-gpu-uuid-locks \
  --retrieval-index "$EXISTING_VERIFIED_INDEX" \
  --retrieval-corpus "$EXISTING_VERIFIED_CORPUS" \
  --retrieval-encoder "$EXISTING_E5_DIRECTORY" \
  --retriever-gpu-lib "$EXISTING_GPU_FAISS_DIRECTORY" \
  --retrieval-url http://127.0.0.1:18210/retrieve \
  --gpu-uuid "$ACTUAL_UUID0" --gpu-uuid "$ACTUAL_UUID1" \
  --gpu-uuid "$ACTUAL_UUID2" --gpu-uuid "$ACTUAL_UUID3" \
  --gpu-uuid "$ACTUAL_UUID4" --gpu-uuid "$ACTUAL_UUID5" \
  --gpu-uuid "$ACTUAL_UUID6" --gpu-uuid "$ACTUAL_UUID7"
```

Variables denote real remote paths/inventory, not pre-existing artifacts claimed by this document. Do not use a currenttraining service port or invent a foreign-process birth. CP5's actualreceipt was inspected: status `QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS`, step5, run `c025948ebc0947f69d86af84871a9237`,25 totalfiles, threeHFshards and savedadded_tokens/special_tokens_map. HFfiles are derived dynamically from the actualindex and all recognized tokenizer-side files, not a hardcoded nine-file count.

ForCP50/100/150/200, use the actual savedreceipt/path when it exists; no fabricatedfuture receipts. Use `--step 200 --split final` for51713. Forofficial0, use the realofficial nine-file modelmanifest as `--checkpoint-receipt`, the officialmodel path as `--checkpoint`, `--step 0 --split validation` and explicit `--run-id` for the logical experiment.

Preparation performs cheap CPU binding/path/config checks before its once-onlyintent, then writes:

- `preparation/binding.json`: role/run/step/split, actualcheckpoint/receipt SHA, allHFinput file hashes, officialmodel path, oldresolvedconfig SHA, datasetSHA/canonical occurrence binding, recipehash, old12-source map, independently publishedcontrolcommit/map, localretrievalURL.
- `preparation/legacy-evaluation-manifest.json`: above identities plus sourcebase/controltree/oldtree, uniqueexecutionID, CPU96/memory512GiB/world8, actual8UUIDs, sharing and forbiddenjob bindings, globalOFDlease root, existingretrievalasset paths and oldadmissionreference SHA, isolatedRay/tmp/journal paths.
- `prepare-intent.json`, `prepare-receipt.json` and `launch.sh`. Existing namespace means inspect; never blindly replay preparation. It reports `submitted=false`, `actual_gpu_execution=false`.

The oldresolved configuration is canonically pinned to `12e4b148cfbfbb8baa67baf80aa0f3796ac93f6c33cb056b1e8e9f155489104e`, not merely checked for a few selected fields. The protocolJSON hash is unchanged by this implementation.

## Sharing bindings

Only these sharingJSON keys are allowed: `allocation_mode`, `preserved_foreign_allocations`, `preserved_foreign_gpu_processes`, `forbidden_training_job_ids`, `gpu_lock_filename`. They cannot override recipe/run/source/UUID/path bindings. Global leasefilename must stay `{uuid}.lock`.

A shared template when **no other permittedallocation/process is present** is:

```json
{
  "allocation_mode": "shared",
  "preserved_foreign_allocations": [],
  "preserved_foreign_gpu_processes": [],
  "forbidden_training_job_ids": ["3400"],
  "gpu_lock_filename": "{uuid}.lock"
}
```

If an authorizedvideoallocation remains, populate its actualjob ID and exact `scontrol UserId` string in `preserved_foreign_allocations`; GPU processes additionally require actualUUID and originalPID/start_ticks/SID/UID identity objects in `preserved_foreign_gpu_processes`. Unknownallocations/GPUpids are refused and preserved, never killed. Sharedmode checks96CPU/512GiB and all8actualCUDAUUIDs without requiring a secondGPU GRES reservation. The active3400 allocation must not overlap this evaluator. `exclusive` is an optional explicitmode, not a newdefault user restriction.

## Future safe allocation only

When an authorizedavailable8GPU allocation is actually available, its owned launcher may run:

```bash
"$EVAL_BASE/preparation/launch.sh"
```

Equivalent stable entry point:

```bash
PYTHONPATH="$CONTROL" "$REMOTE_ENV/bin/python" -m legacy_eval_control.entry pipeline --base "$EVAL_BASE"
```

This is **not** authorization to submit an additionalGPU job now, silently oversubscribe3400, stoptraining, or rerun abaseline. The pipeline verifies nativeallocation/OFDleases, then only within that evaluator's allocation performs remoteCPU fullSHA reads of fixedindex/corpus and oldofficialE5file map/revision. The retriever reuses that exactpreflight/stat receipt; no duplicatefullindex hashing in its child. The runner verifies savedHFbytes before loading, setsFP32HF and vLLM, uses byte-pinnedoldsynchronousloop/batch64/cache.1, recordsactualengine scheduler values and versions, and emits remote64-questionbatchjournals.

Completejournal re-entry needs the originalcanonical result and matchingactual8rankloaded-workerproof; it verifies alljournal rows and checkpoint/recipe binding, then returns the originalresultunchanged without GPU initialization. Missingactualproof refusescanonical aggregation. Partialjournal resume remains the samecheckpoint/recipe and exactcanonicaloccurrenceprefix; it never inheritsoptimizer/head/trainingstate.

Resourceclosure distinguishes evaluatorleader exit0 from cleanup. KnownownedRaychildren enter `EXIT0_CLEANUP_PENDING`; ownedpidfd/birth cleanup and GPUrelease must finish before the pipeline reports success. The supervisor is itself still in the nativecgroup, so nativejob actualexit0 and cgroupdisappearance remain **post-exit observer requirements**, never falsely claimed inside the livejob.

## Verification scope

Static compilation has been performed. FormalCPU verification belongs to a separately staged remote diagnostics directory, with no localenvironment installation:

```bash
SEARCH_G1_LEGACY_SOURCE_ROOT="$OLD_CODE" PYTHONPATH="$CONTROL" \
  "$REMOTE_ENV/bin/python" -m unittest \
  reproduction.legacy_evaluation.test_contract \
  reproduction.legacy_evaluation.test_runtime \
  reproduction.legacy_evaluation.test_pipeline
```

Tests are syntheticscalar/codefixtures, not actualGPUacceptance or modelquality. The packagefixture reads authenticatedoldsource files from `SEARCH_G1_LEGACY_SOURCE_ROOT`, so deployeddiagnostics need no`.git`. LocalGit fallback is for development only. RemoteCPU tests, actualmodel loading, actualoldcanonical512/51713 outputs, and nativeexit/cgroupclosure must each be recorded as separate evidence. Do not interpret code/specification/CPUtests as completedcanonicalevaluation.

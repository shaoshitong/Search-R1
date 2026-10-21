#!/usr/bin/env bash
# Explicit once-only CP1 transition: prepare -> isolated numerical workers ->
# fresh production processes.  The ordinary external submit claim owns the job.
set -euo pipefail
umask 007
BASE="${1:?explicit remote run base required}"
COMMIT="${2:?immutable source commit required}"
ACTOR_MICRO="${3:?explicit actor micro required}"
RESTORE_RECEIPT="${4:?explicit CP1 restore receipt required}"
INFRA_PROFILE="${5:?explicit infrastructure profile required}"
EXTERNAL_RETRIEVER_RECEIPT="${6:?explicit external retriever receipt required}"
RAW_POOL="${7:?explicit original outer1 raw pool file required}"

# Match train.sh's environment, source binding, and ordinary preparation.
export SEARCH_G1_BASE="$BASE" SEARCH_G1_SOURCE_COMMIT="$COMMIT" SEARCH_G1_PROFILE=common8
export QPROBE_SOURCE="$BASE/repo" SEARCH_G1_TRAINING_WORLD_SIZE=4
export SEARCH_G1_EXECUTION_PROFILE=native4_bf16_varlen_fp32_master
export PATH="$BASE/env/bin:$PATH" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
export TMPDIR="$BASE/tmp" TRITON_CACHE_DIR="$BASE/cache/triton"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch-extensions" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset SEARCH_G1_SLURM_SHARED_AUTHORIZATION
mkdir -p "$BASE/cache/triton" "$BASE/cache/torch-extensions" "$BASE/cache/inductor" "$BASE/artifacts"
cd "$BASE/repo"
PREPARE_ARGS=(--base "$BASE" --commit "$COMMIT" --actor-micro "$ACTOR_MICRO")
PREPARE_ARGS+=(--infra-profile "$INFRA_PROFILE" --restore-receipt "$RESTORE_RECEIPT")
PREPARE_ARGS+=(--external-retriever-receipt "$EXTERNAL_RETRIEVER_RECEIPT")
python -B -m reproduction.easy2560_joint4.prepare_training "${PREPARE_ARGS[@]}"

mkdir -p "$BASE/diagnostics" "$BASE/logs"
NUMERICAL_OUTPUT="$BASE/diagnostics/cp1-sparse-numerical-qualification.json"
NUMERICAL_LOG="$BASE/logs/cp1-sparse-numerical-qualification.log"

record_transition() {
    python -B - "$BASE" "$COMMIT" "$ACTOR_MICRO" "$RESTORE_RECEIPT" "$INFRA_PROFILE" \
        "$EXTERNAL_RETRIEVER_RECEIPT" "$RAW_POOL" "$1" "$2" "$NUMERICAL_OUTPUT" "$NUMERICAL_LOG" <<'PY'
import json, os, sys, time
from pathlib import Path
base, commit, micro, restore, profile, external, raw, phase, code, report, log = sys.argv[1:]
value = dict(role='explicit_once_only_cp1_numerical_transition', base=base,
    source_commit=commit, actor_micro_batch_per_rank=int(micro), restore_receipt=restore,
    infra_profile=profile, external_retriever_receipt=external, original_raw_pool_file=raw,
    phase=phase, numerical_exit_code=None if code == 'pending' else int(code),
    numerical_report=report, numerical_log=log, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
    checked_unix=time.time(), numerical_workers_reused_for_production=False,
    numerical_optimizer_state_restored=False, numerical_training_rng_restored=False,
    formal_full_state_restore_still_required=True, automatic_retry_or_relaxation=False,
    production_dispatch_allowed=phase == 'STARTING_FRESH_PRODUCTION')
path = Path(base)/'logs/cp1-numerical-transition.json'
temp = path.with_name(path.name + '.partial')
with temp.open('w') as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
os.replace(temp, path)
print('EASY7B_CP1_NUMERICAL_TRANSITION=' + json.dumps(value), flush=True)
PY
}

record_transition NUMERICAL_DIAGNOSTIC_RUNNING pending
# torch.distributed.run owns/reaps all four diagnostic ranks before it returns.
# A later production process never inherits these workers or their CUDA/RNG.
if python -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    --module reproduction.easy2560_joint4.qualify_sparse_cuda \
    --base "$BASE" --raw-pool-file "$RAW_POOL" --output "$NUMERICAL_OUTPUT" \
    >"$NUMERICAL_LOG" 2>&1; then
    record_transition STARTING_FRESH_PRODUCTION 0
else
    NUMERICAL_CODE=$?
    record_transition NUMERICAL_DIAGNOSTIC_FAILED "$NUMERICAL_CODE"
    exit "$NUMERICAL_CODE"
fi

# No preparation is repeated; the run manifest and restore/external/source
# bindings are the same ones used by the completed explicit diagnostic.
exec python -B -m reproduction.easy2560_joint4.native_ray_resources --base "$BASE"

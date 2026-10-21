#!/usr/bin/env bash
set -euo pipefail
umask 007
BASE="${1:?explicit remote run base required}"
COMMIT="${2:?immutable source commit required}"
ACTOR_MICRO="${3:-8}"
RESTORE_RECEIPT="${4:-}"
INFRA_PROFILE="${5:-baseline}"
EXTERNAL_RETRIEVER_RECEIPT="${6:-}"
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
PREPARE_ARGS+=(--infra-profile "$INFRA_PROFILE")
if [[ -n "$RESTORE_RECEIPT" ]]; then PREPARE_ARGS+=(--restore-receipt "$RESTORE_RECEIPT"); fi
if [[ -n "$EXTERNAL_RETRIEVER_RECEIPT" ]]; then PREPARE_ARGS+=(--external-retriever-receipt "$EXTERNAL_RETRIEVER_RECEIPT"); fi
python -B -m reproduction.easy2560_joint4.prepare_training "${PREPARE_ARGS[@]}"
exec python -B -m reproduction.easy2560_joint4.native_ray_resources --base "$BASE"

#!/usr/bin/env bash
set -euo pipefail
umask 007
BASE="${1:?explicit remote run base required}"
COMMIT="${2:?immutable source commit required}"
RESTORE_RECEIPT="${3:-}"
EXTERNAL_RETRIEVER_RECEIPT="${4:-}"
export SEARCH_G1_BASE="$BASE" SEARCH_G1_SOURCE_COMMIT="$COMMIT" SEARCH_G1_PROFILE=common8
export QPROBE_SOURCE="$BASE/repo" SEARCH_G1_TRAINING_WORLD_SIZE=4
export SEARCH_G1_EXECUTION_PROFILE=native4_bf16_varlen_fp32_master
export PATH="$BASE/env/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$BASE/retriever-gpu-lib:$BASE/repo"
export HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache" TMPDIR="$BASE/tmp"
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCH_EXTENSIONS_DIR="$BASE/cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NCCL_NVLS_ENABLE=0
export LD_PRELOAD="$BASE/preparation/flash3-runtime/libcudart.so.12"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
if [[ -n "$RESTORE_RECEIPT" ]]; then export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; fi
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset SEARCH_G1_SLURM_SHARED_AUTHORIZATION
mkdir -p "$BASE/cache/triton" "$BASE/cache/torch-extensions" "$BASE/cache/inductor" "$BASE/artifacts"
cd "$BASE/repo"
ARGS=(--base "$BASE" --commit "$COMMIT")
if [[ -n "$RESTORE_RECEIPT" ]]; then ARGS+=(--restore-receipt "$RESTORE_RECEIPT"); fi
if [[ -n "$EXTERNAL_RETRIEVER_RECEIPT" ]]; then ARGS+=(--external-retriever-receipt "$EXTERNAL_RETRIEVER_RECEIPT"); fi
exec python -B -m reproduction.dynrpo7b.train "${ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail
umask 007
ROOT="${1:?evaluation root required}"
COMMIT="${2:?published evaluation source required}"
BASE=/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023
SOURCE="$ROOT/source/$COMMIT"
export LD_PRELOAD="$BASE/preparation/flash3-runtime/libcudart.so.12"
export PYTHONPATH="$SOURCE" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1
export XDG_CACHE_HOME="$ROOT/cache" HF_HOME="$ROOT/cache/hf" TRITON_CACHE_DIR="$ROOT/cache/triton"
unset RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset SEARCH_G1_BASE SEARCH_G1_PROFILE SEARCH_G1_SOURCE_COMMIT
mkdir -p "$ROOT/cache/triton"
cd "$SOURCE"
for STEP in 100 150; do
  echo "MILESTONE_EVALUATION_START step=$STEP job=$SLURM_JOB_ID"
  "$BASE/env/bin/python" -B -m reproduction.easy2560_joint4.milestone_eval \
    --source-base "$BASE" --checkpoint "$BASE/checkpoints/search-g1-common8/actor/global_step_$STEP" \
    --output "$ROOT/eval-$STEP" --commit "$COMMIT" --step "$STEP" \
    --retrieval-url http://ACD1-4:18026/retrieve
  echo "MILESTONE_EVALUATION_COMPLETE step=$STEP job=$SLURM_JOB_ID"
done

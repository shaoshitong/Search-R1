#!/usr/bin/env bash
# Independent numerical comparison only; never starts Ray or production training.
set -euo pipefail
umask 007
BASE="${1:?private diagnostic base}"
COMMIT="${2:?published source commit}"
SOURCE_BASE="${3:?unchanged live production base}"
export LD_PRELOAD="$BASE/preparation/flash3-runtime/libcudart.so.12"
export SEARCH_G1_BASE="$BASE" SEARCH_G1_SOURCE_COMMIT="$COMMIT" SEARCH_G1_PROFILE=common8
export QPROBE_SOURCE="$BASE/repo" SEARCH_G1_TRAINING_WORLD_SIZE=4
export SEARCH_G1_EXECUTION_PROFILE=native4_bf16_varlen_fp32_master
export PATH="$BASE/env/bin:$PATH" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
export TMPDIR="$BASE/tmp" TRITON_CACHE_DIR="$BASE/cache/triton"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch-extensions" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NCCL_NVLS_ENABLE=0
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset SEARCH_G1_SLURM_SHARED_AUTHORIZATION WG_PREFIX WG_BACKEND DISABLE_WORKER_INIT
mkdir -p "$BASE/diagnostics" "$BASE/cache/triton" "$BASE/cache/torch-extensions" "$BASE/cache/inductor"
cd "$BASE/repo"
python -B -m reproduction.operations.easy7b_20261021.prepare_fa3_model_probe \
  --base "$BASE" --commit "$COMMIT" --source-base "$SOURCE_BASE"
exec python -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  --module reproduction.easy2560_joint4.qualify_fa3_model_cuda \
  --base "$BASE" --output "$BASE/diagnostics/fa3-model-comparison.json"

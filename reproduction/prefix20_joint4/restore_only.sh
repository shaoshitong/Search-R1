#!/usr/bin/env bash
set -euo pipefail
umask 007
BASE="${1:?explicit remote restore-only target required}"
export SEARCH_G1_BASE="$BASE" QPROBE_SOURCE="$BASE/repo" SEARCH_G1_PROFILE=common8
export SEARCH_G1_TRAINING_WORLD_SIZE=4 SEARCH_G1_EXECUTION_PROFILE=native4_bf16_varlen_fp32_master
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo"
export HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache" TMPDIR="$BASE/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0
unset PYTHONHOME RAY_ADDRESS SEARCH_G1_SLURM_SHARED_AUTHORIZATION
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$BASE/repo"
python -m reproduction.prefix20_joint4.restore_controller check-execute --base "$BASE"
exec python -m reproduction.prefix20_joint4.restore_only execute --base "$BASE"

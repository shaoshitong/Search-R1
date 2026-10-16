#!/usr/bin/env bash
set -euo pipefail
umask 007
BASE="${1:?explicit remote run base required}"
COMMIT="${2:?immutable source commit required}"
export SEARCH_G1_BASE="$BASE" SEARCH_G1_SOURCE_COMMIT="$COMMIT" SEARCH_G1_PROFILE=common8
export QPROBE_SOURCE="$BASE/repo" SEARCH_G1_TRAINING_WORLD_SIZE=4
export SEARCH_G1_EXECUTION_PROFILE=native4_bf16_varlen_fp32_master
export PATH="$BASE/env/bin:$PATH" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
export TMPDIR="$BASE/tmp" TRITON_CACHE_DIR="$BASE/cache/triton"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch-extensions" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset SEARCH_G1_SLURM_SHARED_AUTHORIZATION
mkdir -p "$BASE/cache/triton" "$BASE/cache/torch-extensions" "$BASE/cache/inductor" "$BASE/artifacts"
cd "$BASE/repo"
python -m reproduction.prefix20_joint4.prepare_training --base "$BASE" --commit "$COMMIT"
if [[ "$BASE" == /data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727148-20261016 ]]; then
  python -B -m reproduction.prefix20_joint4.recovery727148_node_cleanup --recovery-base "$BASE" --commit "$COMMIT"
fi
exec python -B -m reproduction.prefix20_joint4.native_ray_resources --base "$BASE" --role train

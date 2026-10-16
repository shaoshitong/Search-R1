#!/usr/bin/env bash
# Run only owned failed-allocation cleanup and the isolated zero-GPU Ray fixture.
set -euo pipefail
umask 007
SOURCE_BASE="${1:?failed independent execution required}"
TARGET_BASE="${2:?new independent diagnostic namespace required}"
COMMIT="${3:?published immutable source required}"
FAILED_JOB="${4:?actual terminal source job required}"
export QPROBE_SOURCE="$TARGET_BASE/source/$COMMIT" PYTHONDONTWRITEBYTECODE=1
export PATH="$TARGET_BASE/env/bin:$PATH" TMPDIR="$TARGET_BASE/tmp"
export HF_HOME="$TARGET_BASE/cache/hf" XDG_CACHE_HOME="$TARGET_BASE/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$QPROBE_SOURCE"
export SEARCH_G1_BASE="$SOURCE_BASE"
python -u -m reproduction.hpc3_dynrpo_periodic.reconcile_cleanup --failed-job "$FAILED_JOB"
export SEARCH_G1_BASE="$TARGET_BASE"
exec python -u -m reproduction.hpc3_dynrpo_periodic.ray_thread_smoke \
  --base "$TARGET_BASE" --output "$TARGET_BASE/preparation/ray-thread-smoke.json"

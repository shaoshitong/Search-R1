#!/usr/bin/env bash
set -euo pipefail
PROBE_ROOT="${1:?diagnostic root}"
PROBE_PYTHON="${2:?original training Python}"
PROBE_INTERFACE="${3:?pinned FA3 interface}"
PROBE_CUDART="${4:?process-private compatible CUDA runtime}"
export LD_PRELOAD="$PROBE_CUDART" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR="$PROBE_ROOT/cache/triton" XDG_CACHE_HOME="$PROBE_ROOT/cache"
exec "$PROBE_PYTHON" -B "$PROBE_ROOT/qualify_fa3_kernel_cuda.py" \
  --interface "$PROBE_INTERFACE" --output "$PROBE_ROOT/result.json"

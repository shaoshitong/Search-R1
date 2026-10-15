#!/usr/bin/env bash
set -euo pipefail
umask 007
PHASE="${1:?phase required}"
BASE="${2:?independent remote base required}"
COMMIT="${3:?immutable source required}"
export SEARCH_G1_BASE="$BASE" SEARCH_G1_SOURCE_COMMIT="$COMMIT"
export SEARCH_G1_PROFILE=common8 SEARCH_G1_TRAINING_WORLD_SIZE=4
export SEARCH_G1_EXECUTION_PROFILE=native4_bf16_varlen_fp32_master
export QPROBE_SOURCE="$BASE/source/$COMMIT" PYTHONDONTWRITEBYTECODE=1
export PATH="$BASE/env/bin:$PATH" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
export TMPDIR="$BASE/tmp" TRITON_CACHE_DIR="$BASE/cache/triton"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch-extensions" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$QPROBE_SOURCE"
case "$PHASE" in
  preparation)
    exec 9>"$BASE/preparation/preparation.lock"
    flock -n 9
    python -m unittest reproduction.hpc3_dynrpo_periodic.test_continuity
    exec python -u -m reproduction.hpc3_dynrpo_periodic.warmup
    ;;
  formal)
    exec 9>"$BASE/preparation/formal.lock"
    flock -n 9
    if [[ -f "$BASE/logs/run-manifest.json" ]]; then
      python -c 'import json,os; from pathlib import Path; r=json.loads((Path(os.environ["SEARCH_G1_BASE"])/"logs/run-manifest.json").read_text()); assert r.get("full_state_restore"), "Existing fresh run cannot be launched twice"'
    else
      python -u -m reproduction.hpc3_dynrpo_periodic.bind
    fi
    exec python -u -m reproduction.common8.run_pipeline --base "$BASE" --export-port 18016
    ;;
  *) exit 2 ;;
esac

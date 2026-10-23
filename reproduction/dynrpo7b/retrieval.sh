#!/usr/bin/env bash
set -euo pipefail
umask 007
BASE="${1:?explicit remote run base required}"
COMMIT="${2:?immutable source commit required}"
export PATH="$BASE/env/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$BASE/retriever-gpu-lib:$BASE/repo"
export HF_HOME="$BASE/retrieval/cache/hf" XDG_CACHE_HOME="$BASE/retrieval/cache" TMPDIR="$BASE/retrieval/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
for LIB in "$BASE"/env/lib/python3.10/site-packages/nvidia/*/lib "$BASE"/env/lib/python3.10/site-packages/torch/lib; do
  if [[ -d "$LIB" ]]; then export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$LIB"; fi
done
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$BASE/repo"
exec python -B -m reproduction.dynrpo7b.retrieval --base "$BASE" --commit "$COMMIT"

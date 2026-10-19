#!/usr/bin/env bash
set -euo pipefail
BASE=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019
SHARED=/data/user/jxu957/experiments/cosearch-continue300-x203-20261008
PY="$SHARED/runtime/retriever-python310-candidate/bin/python3.10"
CODE_ROOT="${QWEN7B_CODE_ROOT:-$BASE/repo}"
export PYTHONPATH="$CODE_ROOT:/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019/env/overlay:/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019/retriever-gpu-lib:$SHARED/source/env/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="$SHARED/runtime/retriever-python310-candidate/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SEARCH_G1_PHYSICAL_WORLD_SIZE=4
export SEARCH_G1_B_PREFIX_TOKENS=20
export QWEN7B_QUESTION_MAX_TOKENS=8
if [[ -f "$BASE/config.json" ]]; then
  export QWEN7B_QUESTION_MAX_TOKENS
  QWEN7B_QUESTION_MAX_TOKENS="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["question_max_tokens"])' "$BASE/config.json")"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$BASE/cache/huggingface"
export PIP_CACHE_DIR="$BASE/cache/pip"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions"
export TMPDIR="$BASE/tmp"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
unset PYTHONHOME http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
cd "$CODE_ROOT"
exec "$PY" "$@"

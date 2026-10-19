#!/usr/bin/env bash
# Python environment entry only; does not launch a trainer or allocate GPUs.
set -euo pipefail
export SEARCH_G1_PHYSICAL_WORLD_SIZE="${SEARCH_G1_PHYSICAL_WORLD_SIZE:-8}"
export SEARCH_G1_B_PREFIX_TOKENS="${SEARCH_G1_B_PREFIX_TOKENS:-5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019/repo:/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019/env/overlay:/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019/retriever-gpu-lib:/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/env/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/retriever-python310-candidate/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019/cache/huggingface"
export PIP_CACHE_DIR="/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019/cache/pip"
export TORCH_EXTENSIONS_DIR="/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019/cache/torch_extensions"
export TMPDIR="/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019/tmp"
unset PYTHONHOME
exec /data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/retriever-python310-candidate/bin/python3.10 "$@"

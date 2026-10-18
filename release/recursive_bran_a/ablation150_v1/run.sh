#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "$0")/../../.." && pwd)"
runtime="${RBA_ABLATION_RUNTIME:-/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009}"
base="${RBA_ABLATION_BASE:?Set the independent condition remote output directory}"
export RBA_H100_ALLOCATION_ID="${RBA_H100_ALLOCATION_ID:-${SLURM_JOB_ID:-}}" RBA_H100_SHARED_ALLOCATION=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="$runtime/env/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$source_root/release/recursive_bran_a/ablation150_v1:$source_root:$runtime/retriever-gpu-lib:$source_root/release/recursive_bran_a/retriever_overlap_candidate:$source_root/release/recursive_bran_a/h100_migration/oracle_source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0
export CPATH=/root/miniconda3/pkgs/python-3.10.21-h17756b0_0/include/python3.10
export TRITON_LIBCUDA_PATH="$runtime/lib/driver" LIBRARY_PATH="$runtime/lib/driver"
export VLLM_ATTENTION_BACKEND=XFORMERS HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export TMPDIR="$base/tmp" XDG_CACHE_HOME="$base/cache" HF_HOME="$base/cache/hf"
unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$base/logs"
cd "$source_root"
exec "$runtime/env/bin/python" -B -u -m search_r1.baselines.preflight -- \
    "$runtime/env/bin/python" -B -u -m release.recursive_bran_a.ablation150_v1.entry \
    --receipt "$base/receipts/configuration.json" "$@"

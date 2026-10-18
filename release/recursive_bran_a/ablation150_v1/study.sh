#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "$0")/../../.." && pwd)"
runtime="${RBA_ABLATION_RUNTIME:-/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009}"
base="${RBA_ABLATION_BASE:?Set the independent study remote output directory}"
export RBA_H100_ALLOCATION_ID="${RBA_H100_ALLOCATION_ID:-${SLURM_JOB_ID:-}}" RBA_H100_SHARED_ALLOCATION=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_root" PATH="$runtime/env/bin:/usr/local/bin:/usr/bin:/bin"
mkdir -p "$base"
cd "$source_root"
exec "$runtime/env/bin/python" -B -u -m release.recursive_bran_a.ablation150_v1.study \
    --base "$base" \
    --template "${RBA_ABLATION_TEMPLATE:-/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013/receipts/resolved.yaml}" \
    --source-commit "${RBA_ABLATION_SOURCE_COMMIT:?Published immutable source SHA required}" \
    --control-token-file "${RBA_ABLATION_CONTROL_TOKEN:-/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013/retriever/receipts/private-probe-token}" \
    --original150-manifest "${RBA_ABLATION_ORIGINAL150:-/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013/checkpoints/training_state/global_step_150/post-evaluation/manifest.json}" \
    "$@"

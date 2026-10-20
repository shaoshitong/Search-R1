#!/bin/bash
#SBATCH --job-name=dynrpo-nq-ground
#SBATCH --partition=emergency_acd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
: "${GROUNDING_SOURCE:?}"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$GROUNDING_SOURCE"
cd "$GROUNDING_SOURCE"
exec /data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python -B -m reproduction.grounding_eval.operations.dynrpo_grounding_controller

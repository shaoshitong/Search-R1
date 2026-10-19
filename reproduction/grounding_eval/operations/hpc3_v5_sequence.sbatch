#!/bin/bash
#SBATCH --job-name=grounding-strict-v5
#SBATCH --partition=emergency_acd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
: "${GROUNDING_SOURCE:?explicit immutable deployed runtime}"
: "${GROUNDING_JOB:?explicit frozen sequence job}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTHONPATH="$GROUNDING_SOURCE"
cd "$GROUNDING_SOURCE"
exec /data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python -B \
  -m reproduction.grounding_eval.operations.hpc3_v5_sequence run --job "$GROUNDING_JOB" --max-batch 256

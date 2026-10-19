#!/bin/bash
#SBATCH --job-name=grounding-normal-nq
#SBATCH --partition=emergency_acd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=12:00:00
# Submit only after normal/CF pilot acceptance and fresh storage budget checks.
# sbatch --output=/ABS/TASK/slurm-%j.log --export=ALL,GROUNDING_SOURCE=/ABS/CODE,GROUNDING_JOB=/ABS/JOB.json normal_hpc3.sbatch
set -euo pipefail
: "${GROUNDING_SOURCE:?explicit deployed source required}"
: "${GROUNDING_JOB:?explicit frozen job JSON required}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
RUNTIME=/data/user/jxu957/experiments/topd-20261008/runtime
cd "$GROUNDING_SOURCE"
exec "$RUNTIME/env/bin/python" -B reproduction/grounding_eval/normal_hpc3.py --job "$GROUNDING_JOB"

#!/bin/bash
#SBATCH --job-name=grounding-nq-cf
#SBATCH --partition=emergency_acd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
# Prepared entry only. Human/parent acceptance gate precedes any sbatch.
# sbatch --output=/ABS/PIPELINE/slurm-%j.log --export=ALL,GROUNDING_SOURCE=/ABS/IMMUTABLE_RUNTIME,GROUNDING_JOB=/ABS/PIPELINE.json pipeline_hpc3.sbatch
set -euo pipefail
: "${GROUNDING_SOURCE:?explicit immutable deployed runtime required}"
: "${GROUNDING_JOB:?explicit frozen pipeline JSON required}"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1 PYTHONPATH="$GROUNDING_SOURCE"
RUNTIME=/data/user/jxu957/experiments/topd-20261008/runtime
cd "$GROUNDING_SOURCE"
exec "$RUNTIME/env/bin/python" -B -m reproduction.grounding_eval.pipeline --job "$GROUNDING_JOB"

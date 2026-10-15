#!/bin/bash
#SBATCH --job-name=cosearch200-final-dual
#SBATCH --account=mou176
#SBATCH --partition=emergency_acd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --time-min=00:30:00
#SBATCH --requeue
#SBATCH --output=/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-final200-four-gpu-20261015/slurm-%j.log
set -euo pipefail
TASK_BASE=/data/user/jxu957/experiments/cosearch-continue300-x203-20261008
EVAL_ROOT="$TASK_BASE/runtime/cosearch-final200-four-gpu-20261015"
export PYTHONPATH="$EVAL_ROOT/tools:$EVAL_ROOT/source:$EVAL_ROOT/source/verl:$TASK_BASE/runtime/flash-attn-281-hpc-build/lib:$TASK_BASE/source/author-native/env/lib/python3.11/site-packages"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$EVAL_ROOT/cache/hf" HF_DATASETS_CACHE="$EVAL_ROOT/cache/datasets" TMPDIR="$EVAL_ROOT/cache/tmp"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$TMPDIR"
exec "$TASK_BASE/source/author-native/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11" -B "$EVAL_ROOT/tools/run.py"

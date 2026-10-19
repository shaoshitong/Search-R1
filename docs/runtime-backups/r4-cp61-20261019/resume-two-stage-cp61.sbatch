#!/usr/bin/env bash
#SBATCH --job-name=prefix-two-stage-cp61-thread5
#SBATCH --partition=emergency_acd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
set -euo pipefail
base=/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4
export DYNRPO_PROGRESSIVE_PREFIX=cp50_prefix20_40_80_three_epoch_v1
export DYNRPO_PREFIX_TOKENS=20
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HOME="$base/cache/hf"
export XDG_CACHE_HOME="$base/cache"
export TMPDIR="$base/tmp"
cd "$base/repo"
exec 9>"$base/preparation/build-run.lock"
flock -n 9
exec "$base/env/bin/python" -m reproduction.hpc3_prefix5.two_stage_runtime --base "$base"

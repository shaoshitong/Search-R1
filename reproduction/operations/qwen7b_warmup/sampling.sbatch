#!/usr/bin/env bash
#SBATCH --job-name=q7b-native-bank
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/logs/sampling-%A_%a.log
set -euo pipefail
BASE=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019
srun --kill-on-bad-exit=1 bash "$BASE/repo/reproduction/operations/qwen7b_warmup/runtime.sh" \
  -u -m reproduction.operations.qwen7b_warmup.shard --config "$BASE/config.json" --shard "$SLURM_ARRAY_TASK_ID"

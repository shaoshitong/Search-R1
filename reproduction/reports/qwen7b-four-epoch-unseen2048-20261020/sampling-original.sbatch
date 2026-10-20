#!/usr/bin/env bash
#SBATCH --job-name=q7b-unseen2048
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/selection-2048-20261020/logs/sampling-%A_%a.log
set -euo pipefail
export QWEN7B_CODE_ROOT=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/repo-postprocess-96df5a23f23f
srun --kill-on-bad-exit=1 bash /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/repo-postprocess-96df5a23f23f/reproduction/operations/qwen7b_warmup/runtime.sh -u -m reproduction.operations.qwen7b_warmup.shard --config /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/selection-2048-20261020/config.json --shard "$SLURM_ARRAY_TASK_ID"

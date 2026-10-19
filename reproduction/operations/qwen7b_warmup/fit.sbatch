#!/usr/bin/env bash
#SBATCH --job-name=q7b-native-probe
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/logs/fit-%j.log
set -euo pipefail
BASE=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019
exec 9>"$BASE/probe/fit.lock"
flock -n 9
srun --kill-on-bad-exit=1 bash "$BASE/repo/reproduction/operations/qwen7b_warmup/runtime.sh" \
  -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m reproduction.operations.qwen7b_warmup.train --base "$BASE"

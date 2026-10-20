#!/usr/bin/env bash
#SBATCH --job-name=q7b-mixed-cp2-ft
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@240
#SBATCH --output=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-20261020/logs/fit-%j.log
set -euo pipefail
BASE=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-20261020
: "${QWEN7B_CODE_ROOT:?Use the explicitly published isolated code root}"
exec 9>"$BASE/probe/fit.lock"
flock -n 9
export PYTHONFAULTHANDLER=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" \
  -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m reproduction.operations.qwen7b_warmup.mixed_finetune --base "$BASE"

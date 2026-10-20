#!/usr/bin/env bash
#SBATCH --job-name=q7b-score4-unseen
#SBATCH --partition=acd_ue
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/selection-2048-20261020/logs/score-%j.log
set -euo pipefail
exec 9>/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/selection-2048-20261020/control/scoring.lock
flock -n 9
export QWEN7B_CODE_ROOT=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/repo-postprocess-96df5a23f23f
srun --kill-on-bad-exit=1 bash /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/repo-postprocess-96df5a23f23f/reproduction/operations/qwen7b_warmup/runtime.sh -u /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/control-tools/ad3a55aef923f3401d350fe7f3479996293e06c6/heldout_selection.py --mode score --base /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/selection-2048-20261020 --micro-groups 32

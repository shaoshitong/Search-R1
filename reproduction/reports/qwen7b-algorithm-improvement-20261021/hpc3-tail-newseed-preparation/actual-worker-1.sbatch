#!/usr/bin/env bash
#SBATCH --job-name=q7b-tail-newseed-w1
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail-newseedG5-20261021-v1/logs/worker-1-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/matched-g10-collector-code-20261021/code-2b293d6e6a0b
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail-newseedG5-20261021-v1/code-2590d0b27201/hpc3_tail_g5.py --worker 1 --config /data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail-newseedG5-20261021-v1/config.json --config-sha256 368ad5d8df9e0238bdee8033964b56f0f4a6426e606f9514114fac0dcb05d835

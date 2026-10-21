#!/usr/bin/env bash
#SBATCH --job-name=q7b-tail14592-L26
#SBATCH --account=mou176
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail14592-independent-L26-prepared-v1/logs/capture-%j.log
#SBATCH --chdir=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail14592-independent-L26-prepared-v1
set -euo pipefail
export QWEN7B_CODE_ROOT=/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-joint124194-audited243-L26-v1/code-db5b5ac04173
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec 9>/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail14592-independent-L26-prepared-v1/gpu-job.lock
flock -n 9
bash "/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-joint124194-audited243-L26-v1/code-db5b5ac04173/reproduction/operations/qwen7b_warmup/runtime.sh" -u "/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail14592-independent-L26-prepared-v1/code-757da0b096c5/capture_hpc3_tail14592_l26.py" --action capture --preparation-sha256 1d4fdc6cef690416c96d5dee8135549caf5793c42d9514c662d131fdd817035c
bash "/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-joint124194-audited243-L26-v1/code-db5b5ac04173/reproduction/operations/qwen7b_warmup/runtime.sh" -u "/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail14592-independent-L26-prepared-v1/code-757da0b096c5/capture_hpc3_tail14592_l26.py" --action merge

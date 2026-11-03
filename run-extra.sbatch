#!/bin/bash
#SBATCH --job-name=qwen3-8b-context-extra
#SBATCH --partition=debug
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:29:00
set -euo pipefail
cd "$TASK_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_CACHE_ROOT="$TASK_ROOT/cache/vllm"
export XDG_CACHE_HOME="$TASK_ROOT/cache"
export TRITON_CACHE_DIR="$TASK_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$TASK_ROOT/cache/torchinductor"
export HF_HOME="$TASK_ROOT/cache/huggingface"
# Unix IPC sun_path is limited to 107 bytes. Keep only ephemeral task-owned
# temporary files under a short unique shared-storage path; logs stay TASK_ROOT.
export TMPDIR
TMPDIR=$(mktemp -d "/data/user/mou176/vq3-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$TMPDIR"' EXIT
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$TASK_ROOT/cache/pycache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
hostname
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
findmnt -T "$TASK_ROOT" -n -o SOURCE,FSTYPE,TARGET
test -s "$TASK_ROOT/storage-sentinel.txt"
sha256sum "$TASK_ROOT/storage-sentinel.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
/share/anaconda3/envs/vllm017/bin/python "$TASK_ROOT/source/driver_extra.py"

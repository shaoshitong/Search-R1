#!/usr/bin/env bash
# Independent target prepared with final_restore_only prepare. Example allocation:
# sbatch --partition=emergency_acd --qos=formal-user --account=jxu957 --nodes=1 \
#   --ntasks=1 --gres=gpu:4 --cpus-per-task=48 --mem=512G --export=NONE \
#   final_restore_only.sh /data/user/jxu957/experiments/easy2560-corollout-joint4-finalrestore200-DATE
set -euo pipefail
umask 007
BASE="${1:?explicit independent final restore target required}"
export SEARCH_G1_BASE="$BASE" QPROBE_SOURCE="$BASE/repo" PYTHONPATH="$BASE/repo" SEARCH_G1_PROFILE=common8
export SEARCH_G1_TRAINING_WORLD_SIZE=4 SEARCH_G1_EXECUTION_PROFILE=native4_bf16_varlen_fp32_master
export PATH="$BASE/env/bin:$PATH" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache" TMPDIR="$BASE/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0 ECO_NO_SELF=1
unset PYTHONHOME RAY_ADDRESS SEARCH_G1_SLURM_SHARED_AUTHORIZATION
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
if [[ -f "$BASE/preparation/flash3-runtime/ready.json" ]]; then
  export LD_PRELOAD="$BASE/preparation/flash3-runtime/libcudart.so.12"
fi
cd "$BASE/repo"
exec python -B -m reproduction.easy2560_joint4.final_restore_only execute --base "$BASE"

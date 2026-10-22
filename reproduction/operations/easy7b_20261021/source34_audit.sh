#!/usr/bin/env bash
# CPU-only source audit. ACD requires one GPU reservation; it is never used.
# Caller owns the one-time sbatch claim and creates NEW_BASE/preparation before
# using it for --output/--error. No bootstrap or production submission occurs.
#SBATCH --partition=emergency_acd
#SBATCH --qos=formal-user
#SBATCH --account=jxu957
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=easy7b-source34-cpu-audit
#SBATCH --export=NONE
set -euo pipefail
umask 007
NEW_BASE="${1:?explicit new owned recovery base required}"
OPERATOR_SOURCE="${2:?explicit published operator source directory required}"
OLD_BASE='/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp12-20261022'
PYTHON='/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python'
mkdir -p -- "$NEW_BASE/preparation"
cd -- "$OPERATOR_SOURCE"
exec /usr/bin/env -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  "$PYTHON" -B -m search_g1.easy2560_restore \
  --base "$OLD_BASE" \
  --checkpoint "$OLD_BASE/checkpoints/search-g1-common8/actor/global_step_34" \
  --output "$NEW_BASE/preparation/source34-receipt.json"

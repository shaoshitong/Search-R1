#!/bin/bash
#SBATCH --job-name=cosearch200-final-audit
#SBATCH --account=mou176
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-final200-four-gpu-20261015/acceptance-%j.log
set -euo pipefail
TASK_BASE=/data/user/jxu957/experiments/cosearch-continue300-x203-20261008
EVAL_ROOT="$TASK_BASE/runtime/cosearch-final200-four-gpu-20261015"
export PYTHONPATH="$EVAL_ROOT/tools"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "$TASK_BASE/source/author-native/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11" -B - <<'PY'
import os
from common import OUT, TOOLS, PYTHON, env
assert (OUT/'complete.json').is_file(), 'Full final predictions have not completed'
os.execve(str(PYTHON),[str(PYTHON),'-B',str(TOOLS/'verify.py')],env())
PY

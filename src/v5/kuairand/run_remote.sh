#!/usr/bin/env bash
set -eu
analysis_root=$1
run_mode=${2:-full}
source_directory=$(dirname "$0")
python_binary=${V5_PYTHON:-/home/shaoshitong/anaconda3/bin/python}
data_directory=${V5_KUAIRAND_DATA:-/data/shaoshitong/ai_creator_game_20261104/v3/kuairand/raw/KuaiRand-Pure/data}
mkdir -p "$analysis_root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$run_mode" = smoke ]; then
    seeds=1
    permutations=10
else
    seeds=10
    permutations=200
fi
set +e
nice -n 10 "$python_binary" "$source_directory/cross_item_history.py" "$analysis_root" \
    --data "$data_directory" --seeds "$seeds" --permutations "$permutations" \
    > "$analysis_root/analysis.log" 2>&1
analysis_exit=$?
printf '%s\n' "$analysis_exit" > "$analysis_root/exit_code.txt"
exit "$analysis_exit"

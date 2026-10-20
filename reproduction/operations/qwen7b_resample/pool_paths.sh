#!/bin/bash
# Generalized copy of retained-fault-paths.sh, source SHA256:
# b2d62068695e156dcc90fc42e112ad24c676a1e7531fc26e2c0c6a01dbbfc1c1
# Fresh job-private mount namespace only; mount NAS10 before any aliases.
set -euo pipefail
expected="$1"; shift
case "$expected" in h100-5|h100-10) ;; *) exit 64 ;; esac
[[ $(hostname -s) == "$expected" ]]
[[ $(id -u) == 0 ]]
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]
[[ $(readlink /proc/self/ns/mnt) != $(readlink /proc/1/ns/mnt) ]]
mount --make-rprivate /
stash="/run/q7b-pool-nas10-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$stash"
if [[ $expected == h100-10 ]]; then
  # Preserve physical NAS10 before the retained /mnt/data aliases hide it.
  mount --bind /mnt/data "$stash"
else
  mount -t nfs4 -o vers=4.2,proto=tcp,hard,timeo=600,retrans=2 h100-10:/mnt/data "$stash"
fi
[[ -f "$stash/shaoshitong/qwen7b-resample-seed20261021-g5-20261020/config.json" ]]
mkdir -p /mnt/data /dataset/shaoshitong/Search-G1-common8
mount -t tmpfs -o size=16m,nodev,nosuid q7b-pool-retained-paths /mnt/data
nas=/mnt/data/sihe-share5/workspace/training_data_node_10
mkdir -p "$nas" /mnt/shared/workspace/training_data_node_10
mount --bind "$stash" "$nas"
mount --bind "$nas" /mnt/shared/workspace/training_data_node_10
retained=$nas/shaoshitong/search-g1-prefix5-h10011-retained-fixed-assets-20261018
runtime=$nas/shaoshitong/search-g1-prefix5-h10011-runtime-assets-20261018
[[ -f "$retained/cache2-RESULT.json" && -f "$runtime/COPY_COMPLETE" ]]
mkdir -p /mnt/data/shaoshitong
mount --bind "$retained/shaoshitong" /mnt/data/shaoshitong
mount --bind "$runtime/common8" /dataset/shaoshitong/Search-G1-common8
cache="$retained/corpus-cache-relocations/ea94181843f69e23119e92af730895d8a32a0fa95abf8bff438ad9a2882d5d14/cache"
target=/mnt/data/shaoshitong/search-g1-prefix20-online1000-20261016-ca1c60f/retrieval-cache
[[ -f "$cache/corpus.json" && -d "$target" ]]
mount --bind "$cache" "$target"
exec "$@"

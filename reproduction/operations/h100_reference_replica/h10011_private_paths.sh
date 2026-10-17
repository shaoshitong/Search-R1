#!/bin/bash
# Invoke with unshare --mount --propagation private. All bind mounts below
# exist only in this task's namespace; existing tasks retain their own paths.
set -euo pipefail
[[ $(hostname) == h100-11 ]]
[[ $(readlink /proc/self/ns/mnt) != $(readlink /proc/1/ns/mnt) ]]
shared=/mnt/shared/workspace/training_data_node_10
runtime=$shared/shaoshitong/search-g1-prefix5-h10011-runtime-assets-20261018
[[ -f $runtime/COPY_COMPLETE ]]
mkdir -p /mnt/data /dataset/shaoshitong/Search-G1-common8
mount -t tmpfs -o size=16m,nodev,nosuid prefix5-private-paths /mnt/data
mkdir -p /mnt/data/shaoshitong /mnt/data/sihe-share5/workspace/training_data_node_10
mount --bind /mnt/shared/workspace/training_data_node_8/shaoshitong /mnt/data/shaoshitong
mount --bind "$shared" /mnt/data/sihe-share5/workspace/training_data_node_10
mount --bind "$runtime/common8" /dataset/shaoshitong/Search-G1-common8
exec "$@"

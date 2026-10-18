#!/bin/bash
# Only a new mount namespace; expose retained immutable assets and own outputs.
set -euo pipefail
[[ $(hostname) == h100-10 ]]
[[ $(readlink /proc/self/ns/mnt) != $(readlink /proc/1/ns/mnt) ]]
stash=/run/search-g1-b-replica-retained18-root
mkdir -p "$stash"
mount --bind /mnt/data/shaoshitong "$stash"
mount -t tmpfs -o size=16m,nodev,nosuid b-replica-private-paths /mnt/data
shared=/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong
mkdir -p "$shared" /mnt/data/shaoshitong /dataset/shaoshitong/Search-G1-common8
mount --bind "$stash" "$shared"
fixed=$shared/search-g1-prefix5-h10011-retained-fixed-assets-20261018/shaoshitong
runtime=$shared/search-g1-prefix5-h10011-runtime-assets-20261018
[[ -f $runtime/COPY_COMPLETE && -d $fixed ]]
mount --bind "$fixed" /mnt/data/shaoshitong
mount --bind "$runtime/common8" /dataset/shaoshitong/Search-G1-common8
# Keep physical UUID leases shared with other tasks on H10018.
mkdir -p /mnt/data/shaoshitong/.search-g1-gpu-uuid-locks
mount --bind "$shared/.search-g1-gpu-uuid-locks" /mnt/data/shaoshitong/.search-g1-gpu-uuid-locks
exec "$@"

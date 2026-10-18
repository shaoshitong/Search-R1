#!/bin/bash
# Private namespace only. Payloads and outputs remain on retained H10018.
set -euo pipefail
node=$(hostname)
[[ $node == h100-8 || $node == h100-10 || $node == h100-5 ]]
[[ $(readlink /proc/self/ns/mnt) != $(readlink /proc/1/ns/mnt) ]]
stash=/run/search-g1-diverse-retained18-shared
mkdir -p "$stash"
if [[ $node == h100-10 ]]; then
  mount --bind /mnt/data/shaoshitong "$stash"
else
  mount --bind /mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong "$stash"
fi
mount -t tmpfs -o size=16m,nodev,nosuid diverse-private /mnt/data
shared=/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong
mkdir -p "$shared" /mnt/data/shaoshitong /dataset/shaoshitong/Search-G1-common8
mount --bind "$stash" "$shared"
fixed=$shared/search-g1-prefix5-h10011-retained-fixed-assets-20261018/shaoshitong
runtime=$shared/search-g1-prefix5-h10011-runtime-assets-20261018
[[ -f $runtime/COPY_COMPLETE && -d $fixed ]]
mount --bind "$fixed" /mnt/data/shaoshitong
mount --bind "$runtime/common8" /dataset/shaoshitong/Search-G1-common8
mkdir -p /mnt/data/shaoshitong/.search-g1-gpu-uuid-locks
mount --bind "$shared/.search-g1-gpu-uuid-locks" /mnt/data/shaoshitong/.search-g1-gpu-uuid-locks
exec "$@"

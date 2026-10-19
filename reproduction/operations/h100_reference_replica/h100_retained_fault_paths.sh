#!/bin/bash
# Fresh private namespace only; retained NAS10 assets, no source8 mounts.
set -euo pipefail
expected="$1"; shift
case "$expected" in h100-5|h100-11) ;; *) exit 64 ;; esac
[[ $(hostname) == "$expected" ]]
[[ $(readlink /proc/self/ns/mnt) != $(readlink /proc/1/ns/mnt) ]]
mkdir -p /mnt/data /dataset/shaoshitong/Search-G1-common8
mount -t tmpfs -o size=16m,nodev,nosuid prefix5-retained-paths /mnt/data
nas=/mnt/data/sihe-share5/workspace/training_data_node_10
mkdir -p "$nas"
mount -t nfs4 -o vers=4.2,proto=tcp,hard,timeo=600,retrans=2 h100-10:/mnt/data "$nas"
retained=$nas/shaoshitong/search-g1-prefix5-h10011-retained-fixed-assets-20261018
runtime=$nas/shaoshitong/search-g1-prefix5-h10011-runtime-assets-20261018
[[ -f "$retained/cache2-RESULT.json" && -f "$runtime/COPY_COMPLETE" ]]
mkdir -p /mnt/data/shaoshitong
mount --bind "$retained/shaoshitong" /mnt/data/shaoshitong
mount --bind "$runtime/common8" /dataset/shaoshitong/Search-G1-common8
cache="$retained/corpus-cache-relocations/ea94181843f69e23119e92af730895d8a32a0fa95abf8bff438ad9a2882d5d14/cache"
[[ -f "$cache/corpus.json" ]]
mount --bind "$cache" /mnt/data/shaoshitong/search-g1-prefix20-online1000-20261016-ca1c60f/retrieval-cache
exec "$@"

#!/bin/bash
# Use only in a fresh private mount namespace. This file does not stop, restart,
# or edit a trainer. A training handoff additionally requires released old leases
# and a complete joint-state resume; never replace mounts in a live namespace.
set -euo pipefail
[[ $(hostname) == h100-11 ]]
[[ $(readlink /proc/self/ns/mnt) != $(readlink /proc/1/ns/mnt) ]]
mkdir -p /mnt/data /dataset/shaoshitong/Search-G1-common8
mount -t tmpfs -o size=16m,nodev,nosuid prefix5-retained-paths /mnt/data
nas=/mnt/data/sihe-share5/workspace/training_data_node_10
mkdir -p "$nas"
mount -t nfs4 -o vers=4.2,proto=tcp,hard,timeo=600,retrans=2 h100-10:/mnt/data "$nas"
retained=$nas/shaoshitong/search-g1-prefix5-h10011-retained-fixed-assets-20261018
runtime=$nas/shaoshitong/search-g1-prefix5-h10011-runtime-assets-20261018
[[ -f "$retained/cache2-RESULT.json" && -f "$runtime/COPY_COMPLETE" ]]
/usr/bin/python3 - "$retained/cache2-RESULT.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['status']=='FIXED_INPUT_RETENTION_COMPLETE'
assert r['live_training_changed'] is False and r['model_bytes_via_Mac'] is False
PY
mkdir -p /mnt/data/shaoshitong
mount --bind "$retained/shaoshitong" /mnt/data/shaoshitong
mount --bind "$runtime/common8" /dataset/shaoshitong/Search-G1-common8
exec "$@"

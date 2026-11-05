#!/usr/bin/env bash
# Exact task-only remote replica; stream through the client, never raw Mac files.
set -euo pipefail
TASK_ROOT=/data/shaoshitong/ai_creator_game_20261104
TASK_REPLICA=$TASK_ROOT/v7/empirical_backup_from_hkust3
ssh -o BatchMode=yes -o ConnectTimeout=8 hkust2 "mkdir -p '$TASK_REPLICA/source_data'"
ssh -o BatchMode=yes -o ConnectTimeout=8 hkust3 "tar -C '$TASK_ROOT/v3/kuairec' -cf - item_daily_features.csv" |
  ssh -o BatchMode=yes -o ConnectTimeout=8 hkust2 "tar -C '$TASK_REPLICA/source_data' -xf -"
ssh -o BatchMode=yes -o ConnectTimeout=8 hkust3 "tar -C '$TASK_ROOT/v7' -cf - empirical openchannelstats_empirical source_discovery/raw source_discovery/public_pool source_discovery/audit" |
  ssh -o BatchMode=yes -o ConnectTimeout=8 hkust2 "tar -C '$TASK_REPLICA' -xf -"
# Use replica_manifest.py at both bases, then require identical file maps.
# The current run was verified with this comparison, recorded in V7_AUDIT.

# Closed-group archive on the existing same-cluster mount

This independent CPU helper copies only fully completed new ablation150 groups.
It does not change training, acquire training locks, initialize models, call GPUs
or HTTP, send signals, allocate nodes, or move assets through the Mac. The exact
destination is the already mounted NFS4 `h100-10:/mnt/data`:

`/mnt/shared/workspace/training_data_node_10/shaoshitong/recursive-ablation150-archive-20261014`

The source is limited to the seven named groups under
`/dataset/shaoshitong/recursive-ablation150-20261014`. A and foreign paths are
excluded. Each group must have all150 native completed-round records, real balanced
dev512 at5/50/100/150, exact current owner absent, exit0, parent wait0, and the
study's accepted owned-GPU release. This uses the existing published
`ablation150_storage_v1` and `ablation150_closed_storage_v2` closure readers.

## Storage readiness

The operator first supplies a SHA-bound readiness receipt from the actual existing
mount's write probe and second-node read of the same64-byte object:

```json
{
  "protocol": "ablation150_existing_same_cluster_storage_ready_v1",
  "ready": true,
  "study_root": "/dataset/shaoshitong/recursive-ablation150-20261014",
  "archive_root": "/mnt/shared/workspace/training_data_node_10/shaoshitong/recursive-ablation150-archive-20261014",
  "mount": {"mountpoint": "/mnt/shared/workspace/training_data_node_10", "fstype": "nfs4", "source": "h100-10:/mnt/data"},
  "probe": {"path": "actual absolute own probe path", "bytes": 64, "sha256": "actual64hex", "local_sha256": "same actual64hex", "server_sha256": "same actual64hex", "server_read_rc": 0, "write_rc": 0},
  "retention_until": null
}
```

Every invocation rechecks the real mount identity and the shared object's current
SHA. Unknown server retention remains unknown; a write probe does not invent a
retention date or promise permanent storage.

## Background maintenance

From the separately published repository root, after replacing the two receipt
placeholders and implementation commit with their actual values:

```sh
python -B -m release.recursive_bran_a.ablation150_archive_v1.archive maintain \
  --source-commit 233b0dc95de3f7b797d5b424d3b7dcaafca5328f \
  --implementation-commit ACTUAL_PUBLISHED_ARCHIVE_COMMIT \
  --storage-ready ACTUAL_READINESS_RECEIPT \
  --storage-ready-sha256 ACTUAL_READINESS_SHA \
  --interval 300 \
  --prune-superseded-after-low-space --reclaim-after-accepted
```

Default maintenance is copy-only. The two opt-ins first call the v2 low-space
superseded pre/rolling cleanup, then archive each completed group early, and only
later permit canonical native relocation when actual local free bytes remain
below the freshly measured next-key burst. `--once` runs one maintenance cycle.
Errors become attention receipts and never stop or modify the training pipeline.
The source's existing250–300GB per-group protected states therefore need not all
remain on the compute node. Copying itself is independent of GPU locks and cannot
be a training launch condition.

Copy targets are `archive_root/group/run_id/members/<original relative path>`.
All four canonical post-evaluation states (rank0–7 plus driver), four HF exports,
and exact original closure/config/manifests are copied. Each file has a full
source SHA before and after copying and a full destination SHA, with unchanged
source metadata. Matching `.partial` prefixes resume; wrong prefixes or existing
different final files refuse overwrite. File and parent fsync precede publication.
`inventory.json`, per-file accepted receipts, transfer progress, and
`complete-sha-verified.json` persist actual coverage and SHA mappings. Transfer
progress flushes every10 seconds or completed member; no waiting for a whole
group to diagnose transfer progress.

Ordinary300-second polling checks prior accepted full SHA with fresh unchanged
destination metadata. It explicitly does **not** claim fresh full-file SHA on
every poll. `verify` performs fresh full destination SHA, as does any actual
canonical reclamation. SHA verifies bytes, not tensor semantics.

## Canonical native reclamation and restoration

Canonical reclamation requires the accepted full archive plus the v2
`COMPLETE_SUPERSEDED_PAYLOADS_PRUNED` receipt and actual absence of **all** old
pre-evaluation/rolling `.pt` payloads. This matters because v2 still depends on
the canonical local files while pruning its own candidates. Until v2 completes,
the archive remains copy-only. The helper deletes at most the exact36 canonical
rank/driver `.pt` files; HF exports, original manifests, configs, logs, attempts,
and failure evidence remain local. It stops deletion when next-key space is
sufficient. Durable per-file intent/actual receipts allow recovery after unlink.

`receipts/canonical-archive-copy-*.json` links original paths to accepted archive
members. `receipts/canonical-archive-reclaim-v1/actual.json` records explicit
historical/pruned local status. Producer manifests stay byte-exact and retain
their original paths; they are **not** rewritten to pretend an absent payload is
a fresh local COMPLETE checkpoint. Later checkpoint audits must use the archive
receipt and its fresh destination SHA. A metadata-only aggregate EM watcher can
still use the retained native manifest and historical finite receipts.

To restore for a future authorized resume, first run `verify` with the exact
complete receipt/SHA. Copy the mapped `.pt` members back to their exact original
paths through independent fsynced `.partial` files while this group stays closed.
Keep the producer manifests unchanged; freshly hash every restored member against
its original saved digest before selecting it as a native resume checkpoint.
Native resume validators are not weakened to accept missing files.

Tests use tiny temporary files and a scaled headroom only:

```sh
python -B -m unittest release.recursive_bran_a.ablation150_archive_v1.test_archive
```

Limitations: archive copy/hash uses disk/network bandwidth in the background;
there is no guaranteed speed multiplier. An unrelated uncooperative filesystem
mutator is not serialized by training locks. Exact closed-owner/writer checks,
non-symlink paths, immutable manifests and before/after hashes constrain copying
and reclamation. Original incomplete or active states are never candidates.

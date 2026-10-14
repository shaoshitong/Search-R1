# Closed-group low-space pruning

This CPU-only helper is separate from the training source. It performs no GPU,
HTTP, tensor reload, signal, archival, model change or training-lock operation.
Dry-run is the default. Run from the published repository root:

```sh
python -B -m release.recursive_bran_a.ablation150_closed_storage_v2.execute \
  --once --source-commit 233b0dc95de3f7b797d5b424d3b7dcaafca5328f
```

`--watch --interval 60 --apply-after-low-space` enables future cleanup only when
fresh local `statvfs` free space is below the measured next-key burst:
`2 × max observed native-state bytes + max observed HF bytes + 16 GiB`.
The fixed CLI root is `/dataset/shaoshitong/recursive-ablation150-20261014`.
Only the seven named new groups are eligible; A and all other roots are excluded.
No qualifying group yields a small attention record, never a training stop.

A group must have the full 150 completed native records, actual balanced dev512
at 5/50/100/150, exact current entry PID absent, exact exit0, parent wait0, and the
study's accepted owned-GPU release. Its four canonical post-evaluation manifests,
full eight-rank/driver files and HF inventories must remain present and bound to
the completed records. The helper first streams every canonical native/HF file
through SHA256, compares recorded digests and before/after metadata, and rechecks
config/manifests/closure/no writers. A later retry reuses that full-SHA receipt only
with fresh, unchanged metadata and closure; it does not call old hashes fresh.

Only exact `rank-00.pt` through `rank-07.pt` and `driver.pt` in redundant
pre-evaluation states or obsolete rolling states of the closed group are unlinked.
Directories, manifests, HF/model files, logs, partial states and failure evidence
remain. The helper stops further deletion once next-key space is sufficient.

`receipts/closed-storage-v2/` keeps the immutable planned mapping, canonical SHA
proof, durable per-file intent and actual receipt, plus explicit tombstones mapping
the old payloads to their canonical post state. If execution dies after unlink but
before its actual receipt, a retry records only observed absence after its own
durable intent. Missing payloads without that intent refuse all further deletion.
Retained source manifests describe historical saved states; pruned payloads are
never presented as fresh COMPLETE checkpoints. This independent reader handles
these tombstones without modifying the v1 metadata inspector.

Limitations: SHA verifies byte identity, not tensor semantics; the burst is based
on the largest currently observed checkpoint sizes, not a future disk guarantee.
There are no training locks. Closed owner/writer checks and non-symlink directory
FD unlinking constrain deletion, but an uncooperative concurrent filesystem
mutator is not serialized. Canonical full-file hashing can consume disk bandwidth;
it happens only after real low space and once per eligible plan, with no extra GPU.

Tests use tiny temporary files only:

```sh
python -B -m unittest release.recursive_bran_a.ablation150_closed_storage_v2.test_execute
```

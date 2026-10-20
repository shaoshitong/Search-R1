# Qwen2.5-7B probe assets to H100 shared storage

No training is launched. The source peer checkout and jobs are unchanged.

## Destination

H100-10 local path: `/mnt/data/shaoshitong/qwen7b-probe-assets-20261020`.
Shared export: `/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/qwen7b-probe-assets-20261020` (per-node mount visibility must be observed, not assumed).

Official model **complete**: `official-qwen2.5-7b-instruct-a09a35458c70`, Qwen/Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`. All 11 selected files, 15,242,788,168 bytes, independently read from the destination and checked against official SHA identities. Download ran entirely on H100, using the retained node's existing HTTP egress. No model bytes crossed Mac.

Code **complete**: four public pinned Git snapshots downloaded on H100; every target file freshly SHA256 checked against independently generated Git-blob manifests. Sampling `f2d5ae7de92d198692abf5bf82143e309d91c271` (1568 files); warmup `96df5a23f23f6da0f24e79e67dcf0e3039f22de0` (1570); Mixed finetune `0eeb747d14ba11a1fc5ed1e3e4dec16f9073c435` (1610); migration `44944c26745213d7c2761037639f59be006b8b62` (1623). Receipt `code/complete.json`.

Probe and training data **in progress**, not accepted at H100 yet. Selected scope: 34 immutable archive groups, 27,054 source files, 219,394,421,216 bytes. Includes canonical input tables, 169,028-question G5 train bank, fixed512 G5 diagnostic, unseen2048 G5 diagnostic, original token/context/own-EM records, packed native features, configuration and epoch2 checkpoint. No new teacher rollout.

Source: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019` on HPC3. Actual data route: HPC3 → HKUST2 `/data/shaoshitong/qwen7b-h100-transfer-20261020` → private HF dataset `Sutoonq/Qwen25-7B-Probe-H100-Transfer-20261020` → H100-10. Mac carries only control, code manifests and small summaries.

Retained best warmup epoch2: `probe/epoch-0002.pt`, 3,744,947,872 bytes, SHA256 `893e2d71e9b45abb92a9573a6091a1a0fb0a5f7ffd3e4e1201d12061317095ad`. This is source identity, **not** proof of H100 acceptance yet. Native 7B feature layer26 and copied block27; hidden dimension3584, Qmax273, capture20, fit-prefix5. Do not substitute 3B architecture or retrain probe during migration.

## Controls and verification

`reproduction/operations/qwen7b_h100_assets/` contains source export, authenticated bridge, resumable receiver and code-only deployment controls. Credentials enter through encrypted stdin/process RAM only, never login caches, argv, manifests or Git. Controllers use once receipts and process birth identity; query those before any restart. Initial failing TLS/upload controllers were retired with owned-process evidence, retaining partial files. Current bridge has task-local transient TLS retries and reconciles previously accepted HF objects.

Each source payload is hashed during immutable packaging. Intermediate archives require source SHA, HKUST2 SHA and HF LFS SHA/size acceptance. Receiver validates archive SHA, extraction scope, payload SHA on write and fresh full-file readback. It creates relocated path mappings without rewriting original science manifests. Only actual `complete.json` status `MIGRATION_ALL_SELECTED_ASSETS_VERIFIED`, matching all 27,054 payloads plus official model and code receipts, closes migration. Current status remains incomplete.

Source and peer tasks are retained. No GPU job, Slurm request, VPN/global SSH change, reserved-block change or history deletion. Destination root writable physical capacity was checked; ordinary free differs from physical free and is not an exclusive allocation.

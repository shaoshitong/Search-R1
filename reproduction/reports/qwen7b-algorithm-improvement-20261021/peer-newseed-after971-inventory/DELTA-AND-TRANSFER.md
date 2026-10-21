# Finite donor increment after accepted971

The observation at epoch 1792588506 froze 138 newly sealed donor batches,
17,664 HotpotQA questions and 88,320 authentic seed20261021 trajectories.
All 971 previously accepted input/seal SHA bindings remained identical.
Qids and trajectory IDs are distinct from those accepted snapshots. One
question repeats a prior training text hash and one repeats a text hash within
this increment; both canonical training rows are retained. All 6,656 genuine
diagnostic/selection/final qids and text hashes were checked and excluded.

An independent sidecar CPU auditor used the byte-identical 3def055a auditor,
without changing the producer. Actual progress was 95 then 138 complete
batches, with fresh payload SHA, causal prefix and own-G5 identity checks.
The terminal receipt is 1,670 bytes, SHA256
`3c05011247e8fa691af012cc0baddfdfe0d6dd88892d058e025820c193d36f31`.
Its remote root is
`/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/source-proof-after971-51057f377942`.
This source qualification does not claim tokenizer strict EM; the target
ingester must recompute it.

The donor worker was checked at its actual host h100-5: PID3532221,
birth11163959, original run ID present. Its absence on h100-10 was a host
distinction, not a producer failure. No process was signalled or restarted.

The isolated snapshot is `audited138-after971-20261021-51057f37`. Its source
export completed 9 archives, 984,002,560 bytes and 1,961 members with fresh
archive/member SHA. Source complete: 462,498 bytes, SHA256
`6c7abbbdb3e6b2599ec5bff298335e566bdfbe6c4379214c8d247dab149acffc`.
Management uploader PID2975619/birth192330333 streams through RAM to the
private HF repository. It has its own namespace/locks; model data do not
transit or persist on the Mac. The receiver uses the qualified eight-range
transport with full archive and member SHA retained.

HPC3 tail generation deliberately uses the same sampling protocol run ID
and trajectory namespace but a different execution ID. Same qid/tid cannot
prove the same response payload. The donor five and HPC3 five must remain
whole, separately qualified execution groups. Their intersection/source
choice is a separate parent-owned step. Counts must not be added blindly.
Current 124,194-question training inputs and all active training remain
unchanged. This task starts no GPU work.

Published transfer entry:

```
python3 -m reproduction.operations.qwen7b_warmup.deploy_peer_frozen_delta_transfer \
  --control-root /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5 \
  --commit 5a957484afb1854d709b0cfe0531aa1608ad1a63 \
  --branch codex/peer-donor-after971-20261021 \
  --config reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-newseed-after971-inventory/migration-config.json \
  --action hf-status
```

Do not replay launch/start when a matching intent, lock or live birth exists.
The first audit control call timed out while the remote launch later
succeeded; the intent guard prevented a second writer. Read the existing
status first. Actual target completion and strict-EM acceptance are required
before this increment is reported as migrated training data.

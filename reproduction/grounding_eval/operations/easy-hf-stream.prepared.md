# Prepared Easy Selection iter200 preservation route

Status: **prepared and offline tested; never executed or uploaded**.

The plan binds thirteen required HF files, 13,604,381,694 bytes, to the
independently reread backup on retained H100 node18. No optimizer, environment,
credential, or original rollout is included. Source remains unchanged.

Run this module **on mgmt-host**, after deploying the reviewed source and public
plan. The following preparation command does not connect to HF or read the token:

```sh
python3 -m reproduction.grounding_eval.upload_h100_hf_stream \
  --plan reproduction/grounding_eval/operations/easy-hf-stream-plan.json \
  --task-dir /mnt/lustre/shaoshitong/grounding-eval-20261019/easy-hf-upload \
  --route-json /home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json \
  --token-file /mnt/lustre/shaoshitong/hf-checkpoint-upload-20261005/task-token
```

Adding `--execute` is a separate explicitly authorized step. A lock prevents
duplicate upload workers in this task directory. Before authorizing it, inspect
existing private HF paths for an identical export; an existing equal file is
verified and skipped, while a different file at the destination is rejected.

Data route is H10018 -> management RAM -> private HF, with no Mac data stream or
management disk staging. Full source SHA and stable stat identity are checked;
source reads are bounded and have deadlines. Each committed file is verified by
HF size plus LFS SHA256 or, for non-LFS metadata, Git blob ID. The final receipt
pins one repository revision. This receipt does not claim A800/HPC3 download or
grounding evaluation completion.

Progress distinguishes `source_read_position_bytes` from `verified_bytes`.
Source read positions may rewind during seek/retry and are not committed bytes.
Only verified per-file receipts contribute to `verified_bytes`.

The authorized synthetic 8MiB sample on 2026-10-19 verified source/sink SHA, with
1.00 MB/s including SSH setup and 2.87 MB/s during payload. This is not a large
checkpoint throughput guarantee. No concurrent multi-file upload has been tested.


## Authorized execution, 2026-10-19

The parent explicitly authorized execution after the original prepared-only review.
The single management uploader is PID 3973196, source commit
`ad09b983cdd6ed47b2640ba81b6ff628e6f4aa7c`. Its task directory is
`/mnt/lustre/shaoshitong/grounding-eval-20261019/easy-hf-upload`.
All thirteen original HF files, including original complete.json, are source-bound.
Three independent model-file streams are actively growing; source read positions
are not HF verified bytes. Only the pinned post-commit LFS SHA/Git blob receipt
may authorize the A800 fetch. No model bytes are staged on or passed through Mac.

The A800 destination is the dedicated task path
`/data/shaoshitong/grounding-eval-20261019/assets/easy-selection-iter200`.
Its separate migration receipt is `asset-migration-complete.json`, preserving
original `complete.json`. Fresh independent full-file readback is required before
normal inference. The retained18 source remains intact.

Easy historical runtime code and occurrence-only identity are deployed on A800 at
`runtime-easy-legacy-c05602bccc0c`; original full result rows were not transferred.
The normal-only launch uses pilot96 before full3610 and the actual raw-token
prefix loop. Adaptive execution is 128 -> 256 -> 512 on an admitted whole80GiB
GPU, bounded by12GiB real running free memory. A fresh-engine recovery may lower
execution cap while reusing only sealed, identity-verified batches.
The owned retriever is now19001, GPU6; former19000 is not used.
Only UUID/OFD/NVML-admitted GPU5 may be considered; foreign training owns0–4,
and GPU7 is foreign. CF waits for the parent's repaired strict protocol.

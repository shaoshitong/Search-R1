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

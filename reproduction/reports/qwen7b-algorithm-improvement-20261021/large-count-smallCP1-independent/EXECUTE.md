# Actual small CP1 audit entry

The source is pinned to8fce2e4683131d1d7c8de35c0e6a40a4fc693af6.
The original743978 allocation must still be running; no login-node tensor load.
Run from this repository, once CP1 exists:

```sh
python3 reproduction/operations/qwen7b_warmup/deploy_large_count_small_checkpoint.py \
  --control-root /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5 \
  --commit 8fce2e4683131d1d7c8de35c0e6a40a4fc693af6 --action audit
```

The helper verifies current owned job identity, actual CP1 stat and the64MiB
limit before an overlapping CPU2/gpu0 step. The auditor checks ZIP/meta tensor
scope before real CPU load. It never reads the old3.75GB E1 checkpoint.
`--action status` only checks CP1 existence/bytes and the independent receipt.
The source is already staged and its one synthetic CPU fixture passed; do not
restage under a new commit or repeat the fixture unnecessarily. Current READY
receipt explicitly records that real CP1 is absent and unaccepted.

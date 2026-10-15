# Local final200 alias publisher

`publish_nested_final200_aliases.py` is a separate, final200-only metadata entry.
Ordinary `reconcile_nested_full_current_aliases.py` and its five-field helper
remain limited to6..199 and retain their original bytes. The training source,
process, source checkout, native writer and cumulative cost producer are untouched.

Run only after the existing SHA-bound core/finalizer have published actual200:

```sh
python3 /absolute/checkout/reproduction/tree-eval-resume-20261013/local-control-backup/publish_nested_final200_aliases.py \
  --workspace /absolute/current/workspace \
  --metadata-source-commit FULL_40_HEX_PUBLISHED_COMMIT
```

The entry verifies its executing script and pinned dependencies against that local
Git commit. It makes no Git network call. The core SHA is
`676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1`; the
finalizer SHA is `61385504e4bb389cdba508c1b74f84fe6464e1d4f8ec11989e8cfe0c230dc541`.

Required evidence is the existing `FULL_OUTER200_PROGRESS_RECEIPT.json`,
`OUTER200_TRAINING_COMPLETE_FINAL_PHASE_ACTUAL_PROGRESS.json`, their bound native200
readback, two ordered polls/resources, CUM200, candidate200 index/part/registry and
rates. All declared immutable manifests and recursive complete-field cost parts
must close with exact SHA/bytes and both existing mirror bytes. Nested duplicate
JSON keys, missing parts, repeated/cyclic parts, different source/config/run,
Invocation, model/data, stale observations and partial publications are refused.
Remote source-file/PT descriptors and explicitly mutable derived snapshots remain
metadata observations; this entry never follows them or reads tensor contents.

Missing FULL/CUM/finalizer proof is PENDING with no publication. The core's actual200
metric-pending boundary is retained if an otherwise valid finalizer proof exists:
unobserved timing/variance/metric fields stay null. No metric199 or dev195 value is
promoted to current200. The normal current/latest dev is actual200 or null/pending.

Only the two STATUS aliases, two REPORT aliases, CURRENT native-followup and
observer JSON, and their two existing mirrors are refreshed. Fourteen old metadata
documents are closed in deterministic gzip backups before any alias replacement.
Numbered historical fields and the unfinished goal are preserved. Actual eight-rank
Adam counts are read from native/progress; direct child physical counts subtract
the actual cold-restored100/rank, and unknown all-history physical calls remain null.
Cold restore stays native5; saved native200 cursor201 is not training201. Exact
original Invocation is required; actual terminal MainPID0 is allowed by the pinned
validator and is reported separately from the original PID503255.

The phase comes from actual final-phase, terminal or explicitly authorized handoff
evidence. It never asserts final51713/HF/tokenizer acceptance, final CPU acceptance,
resource cleanup, a cold reload or goal completion. CUM200 is never rewritten; the
separate final candidate-cost entry remains separate.

Publication is a guarded local multi-file operation, not a global atomic filesystem
transaction. Every source and both mirrors are checked again before writing; all
fourteen targets and three receipt copies are read back twice. An identical rerun
revalidates the full input closure, closed backup and recomputed publication bytes
and makes zero writes. Missing/tampered outputs or partial receipt copies never
produce idempotent PASS. Write errors restore only recognized original/intended
bytes; unknown concurrent bytes require recovery from the preserved backup.

This release was exercised only with temporary synthetic metadata. Actual task
BASE200 invocations and actual200 success products at publication time: **0**.

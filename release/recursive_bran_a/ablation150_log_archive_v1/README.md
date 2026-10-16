# Closed log/tree/evaluation supplement

This CPU-only, copy-only helper fills the original canonical archive's log/tree
coverage gap. It leaves that archive, its daemon, all training directories and
original files unchanged. No training locks, model/tensor imports, GPU/HTTP calls,
signals, new evaluations or Mac heavy assets are involved.

It requires a SHA-bound accepted canonical group150 archive and reuses the
existing exact owner-absent, exit0, wait0, native150/four dev512, owned GPU release,
same-cluster mount and two-node readiness validators. Canonical payload checks
reuse prior full SHA plus fresh unchanged metadata; they do not stream weights.
The supplement copies full native `.log` files, closure records, exact accepted
512 question tree directories at5/50/100/150 (event/root/child/snapshot JSON),
accepted tree summary/provenance, and sparse all-leaf `trajectories.jsonl.gz` with
its unchanged manifest. `actor-inputs.pt` is explicitly excluded: the copied
trajectory manifest is historical evidence and does not assert that its tensor
member is part of this supplement. The native trainer retains aggregate dev EM
in its log/completion records, rather than separate per-question dev files; none
are fabricated or regenerated.

Destination is only the existing mount's independent
`.../recursive-ablation150-archive-20261014/supplements/GROUP/RUN/logs-trees-evaluation-v1/`.
Explicit inventory, source metadata/full SHA before and after, checked resumable
partials, full destination SHA, durable inventory and final per-member SHA
receipt determine completion. Progress is flushed at least every10 seconds of
active copying and at the last member; thousands of extra receipt files are
avoided. A changing source, unsafe path, interrupted copy
or missing accepted tree coverage holds only this supplement; no training action
is taken. Matching interrupted copies resume; differing files are not replaced.
The copy uses disk/network bandwidth; there is no speed or retention guarantee.

After publishing and deploying this package and its existing dependencies:

```sh
python -B -m release.recursive_bran_a.ablation150_log_archive_v1.archive copy \
  --source-commit 233b0dc95de3f7b797d5b424d3b7dcaafca5328f \
  --implementation-commit ACTUAL_PUBLISHED_COMMIT \
  --storage-ready ACTUAL_READINESS_JSON --storage-ready-sha256 ACTUAL_SHA \
  --canonical-complete ACTUAL_CANONICAL_COMPLETE_JSON \
  --canonical-complete-sha256 ACTUAL_CANONICAL_COMPLETE_SHA \
  --tree-summary-root /dataset/shaoshitong/recursive-ablation150-20261014/analysis/tree-summary-b8a28622
```

`verify` takes `--complete-receipt` and `--complete-receipt-sha256` instead of the
three copy-specific arguments and freshly hashes every supplement destination.
The other four identity/storage arguments remain required. An rc2/HOLD is an
incomplete background supplement, not a model or training failure.

For a background process with an explicit actual child wait receipt, invoke
`python -B -u -m release.recursive_bran_a.ablation150_log_archive_v1.run_copy`
with the same seven copy identity/input flags, plus `--deployment`, a new
`--control DEPLOYMENT/control/UNIQUE_NAME`, and `--runtime-manifest` with its
`--runtime-manifest-sha256`. The manifest is the existing deployer's
`{"commit": "...", "files": {"relative/runtime.py": "SHA", ...}}` document;
it must cover all nine production Python files, including this wrapper.
The wrapper rechecks those bytes, records owner/child-start/actual-exit only
inside that new control directory, and inherits stdout for immediate flushing.
An existing control directory is refused; no duplicate child or daemon is made.

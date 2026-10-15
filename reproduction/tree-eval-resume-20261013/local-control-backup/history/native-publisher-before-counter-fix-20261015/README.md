# Native progress publisher history before the counter fix

These two files are byte-for-byte copies from branch
`codex/nested-native5-controls-20261014` at commit
`74f33861d2d8f7331da235ffaae7a560b4efd0a3`. They preserve the exact old
publisher/finalizer pair used by already published immutable FULL receipts
and proofs. This directory is historical evidence, not a new execution entry.

- `sync_nested_native_progress.py` SHA256:
  `020bec1d2f1f06aa2149ad5cc1d1c72f41567c3b8cd10de10193b9a7d1a964c6`
- `finalize_nested_progress.py` SHA256:
  `90a68d6449fbdf50d0dfdbdf01b1574df5b7b13be1b584f1a8503e842fbf018e`

The current publisher distinguishes the validated cold native5 load from the
latest completed optimizer outer and explicitly refreshes the two legacy
metric-pending aliases. The current finalizer changes only its publisher
dependency hash. All existing immutable FULL/proof files and their original
hash bindings remain unchanged, including outer122. The new pair is for the
next newly verified actual outer, beginning with outer123; it must not be used
to rewrite old receipts or proofs for validation.

The restore, unit, metric, rank, native, step-advance, identity, monotonic and
final-phase checks remain unchanged. Validation for this local-only patch is
limited to AST parsing and exact source-byte comparisons of all unchanged
functions/classes and the finalizer dependency-only change. Neither publisher
main nor finalizer is executed by this repair; training, models, configuration,
evaluation and the original trainer are outside its scope.

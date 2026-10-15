# Local current aliases after an existing FULL publication

`reconcile_nested_full_current_aliases.py` is an optional local data/report step
after the existing publisher, cost assembly and two-observation finalizer finish.
It never collects evidence or changes training. It accepts only actual complete
outer6..199 of source `778b2b43dad90fc0c1c5d95d77c2326286011b73`, unit
`sst-tree-nested-branch-fresh200-778b2b43.service`, PID503255 and Invocation
`a313bda919834d24b4c5d491698d39ce`. Pending metrics/dev/cost, partial publication,
another experiment, final200 and stale observations refuse before publication.

Dependencies stay byte-pinned and are not modified:

- `sync_nested_native_progress.py`: SHA256 `676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1`.
- `finalize_nested_progress.py`: SHA256 `61385504e4bb389cdba508c1b74f84fe6464e1d4f8ec11989e8cfe0c230dc541`.

Example after the root owner selects already completed actual evidence:

```sh
python3 reconcile_nested_full_current_aliases.py \
  --outer 126 \
  --poll outputs/tree-nested-branch-handoff-20261013/NATIVE5_CONTINUE_POLL_ACTUAL.json \
  --resource outputs/tree-nested-branch-handoff-20261013/NATIVE5_CONTINUE_RESOURCE_ACTUAL.json \
  --proof outputs/tree-nested-branch-handoff-20261013/OUTER126_FULL_NEXT127_ACTUAL_PROGRESS.json \
  --workspace /absolute/local/workspace
```

The example poll/resource placeholders must be replaced with the owner's actual
latest same-N observations. Creating or publishing this script does not invoke it.

The script reads SHA-bound local FULLN/native/cost/curve/candidateN evidence,
reuses the published unit/restore/metric/ranks/native validators and reconstructs
the proof's second actual progress check. It verifies immutable cost field parts
and the exact FULL hash. The latest input must still report completeN and actual
nextN+1; it cannot overwrite a newer current boundary or observation epoch.

Actual Adam and skip counts remain separate from processed opportunities.
Cold restore stays5/100 each; latest native writerN is a different boundary.
Current timing/variance belongs to FULLN. Current dev is null on non-dev outers,
while latest real completed dev and its immutable curve remain identified.
ActiveN+1 sampling completion, partial writes and variance remain unknown. Old
current rollout pointers and overwritten aliases are retained as historical.
CandidateN is a separately bound supplement for the following cumulative join;
the script never appends or double-counts it into CUMN. CPU seconds come only
from actual resource `CPUUsageNSec / 1e9`; final service CPU remains null.

On explicit invocation, small prior JSON/Markdown snapshots are saved as closed
gzip members with `mtime=0` before
replacing root/nested STATUS and REPORT, CURRENT_NATIVE_RECOVERY_FOLLOWUP,
CURRENT_OBSERVER_CONNECTION_STATE and both existing local mirrors. The shared
publisher lock and re-read hashes reject concurrent writes. An immutable
CURRENT_ALIAS_RECONCILIATION_OUTERN receipt binds the evidence. This step does
not accept the remaining200/final51713/HF-tokenizer/cleanup goal.
The snapshot manifest records original and stored gzip sizes/SHA256; reading and
decompressing each stored member must reproduce the exact original bytes.

Implementation was prepared from the already published FULL125 evidence and
ROOT_FULL125_PUBLICATION_READBACK_1323_1324 data scopes. No training job, new test,
review, canonical publication or script invocation is part of code publication.

# old-context-fp32-async

This isolated evaluator implements the latest requested evaluation recipe:
pre-d00 context/content semantics and FP32 model inference, while retaining
1f75 asynchronous resident phases and independent trajectory progression.
Training never imports this package. It is not the old synchronous evaluator,
and does not claim bitwise equivalence to historical synchronous metrics.

`prepare` binds a real complete checkpoint receipt (0/5/50/100/150/200), the
canonical validation512 or final51713 input manifest, a published evaluator
source map, and isolated services/resources. `pipeline` owns only its declared
services and processes. `runner` evaluates greedily without constructing Adam
or invoking training fit/restoration. `journal` resumes only actual durable
per-occurrence scalar measurements, preserving repeated canonical questions.
There is no recovery from log averages or count-only progress.

The controller preserves the old separate limits: every next request must
have <=4096 prompt-plus-response tokens, while each response append has an
independent <=4096 response-only bound. Initial prompt is at most2048;
observations are tokenized and cut at500 with no closing-wrapper repair;
ordinal4 is the extra final generation with search disabled. No context-based
early final or answer reservation applies. Overlength old requests fail and
retain their evidence; they never silently switch to the new reserve policy.

FP32 is enforced in the rollout-only HF loader, FSDP parameter casting and
vLLM, and checked against actual loaded parameter dtypes. Asynchronous phases
retain current raw sampling/provenance checks. Any phase error is propagated
through owned cleanup. CPU policy/metadata fixtures and compilation do not
prove actual CUDA execution or quality. Those remain pending until a real
isolated evaluation finishes with the exact canonical question count and
actual worker/runtime/exit evidence.

Remote CLI entry points:

- `python -m reproduction.compatible_async_evaluation.prepare --help`
- `python -m reproduction.compatible_async_evaluation.pipeline --base REMOTE_BASE`
- `python -m reproduction.compatible_async_evaluation.runner --binding BINDING --config CONFIG --source-base SOURCE_BASE --journal JOURNAL`

Preparation does not submit a job. Inspect an existing execution's intent and
receipts before retrying; never overwrite/replay an occupied namespace.

Terminal evaluation resume (remote CPU preparation only):

```
python -m reproduction.compatible_async_evaluation.resume \
  --prior-base OLD_REMOTE_EVALUATION --base NEW_EMPTY_REMOTE_EVALUATION \
  --control-publication-proof NEW_CONTROL_ROOT/publication-proof.json \
  --control-commit FULL_PUBLISHED_SHA --sharing-bindings REMOTE_SHARING_JSON
```

This requires two actual terminal/accounting observations, an empty native-job
cgroup, no original PID births/sessions/GPU contexts, and a partial contiguous
journal of complete 64-question batches with its prior actual FP32 worker proof.
Checkpoint, data/order, duplicate occurrence ordinals, generation, recipe and
scientific code are unchanged. Only execution/control files may differ. Both
control commits and the original file hashes are retained in the new intent.
Only scalar journal files are copied remotely; original artifacts remain intact.
New process ledger, Ray namespace and owned services are created upon a later
explicit Slurm launch. No job is submitted by preparation.

Before accounting is purged, `--prior-base OLD --base UNUSED --capture-terminal
REMOTE_PROOF.json` can preserve two actual terminal/release observations. Later
pass `--terminal-proof REMOTE_PROOF.json --terminal-proof-sha256 INDEPENDENT_SHA`.
Missing SSH/accounting alone never proves exit: the pinned artifact must match
that exact prior native job/manifest/ledger, and current queue/process/cgroup/GPU
release checks still run. Command failures remain errors.

Any occupied or partly prepared target is refused. Use `--base NEW --inspect` to
read its once intent/receipt; inspect unknown outcomes before choosing another
fresh namespace. Pipeline refuses launch until the matching resume receipt and
copied prefix hashes exist. A complete journal is a finished-result verification
case, not a resume. No actual GPU resume has yet been performed or validated.

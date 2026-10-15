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

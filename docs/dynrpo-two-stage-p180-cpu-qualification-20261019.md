# R4 P180: three-epoch fitting and probe CPU qualification

Job733898 remains on ACD1-6 with four shared H100 GPUs. The immutable runtime
is2000d46a530c64d78be7802bf11175a68b65effe. No training source, allocation,
loss, scorer or optimizer was changed for this qualification.

## Actual completed180

Student180 completed20 real Adam writes per rank, zero skips, reaching3600
and scheduler180. The source sampling snapshot was actor0179. The P10 pool
contained2560 distinct questions and12800 original G5 trajectories. Its
outcome-based512-question selection retained277 strict Mixed questions and
235 nonmixed fills:54.1015625% Mixed. Selected K=0..5 counts were
110/76/49/67/85/125. The2560 pool is larger than the ordinary1024 complete
pool; this result is not evidence of a benefit from the subsequently fitted
probe.

The Prefix80 probe continued2200->2250->2300->2350, three epochs and150
actual Adam writes per rank. All three epoch receipts and the final joint
were observed. The separate actual181-use acceptance remains pending and
will independently check save-before-fit ordering, all update ledgers and
the exact new head used for next-outer scoring and actor continuity.

Joint180 SHA:
`f28010b9090ad2a7f49965332a4ee901fabc7db54541a1c39eafdca81a2b936b`.
Joint-to-joint time was1017.637s. Successful phases summed963.127s; the
probe fit loop took44.753s and its full phase96.859s. Phase timings exclude
some storage and orchestration overhead, so they are not interchangeable
with the joint-to-joint measurement.

## Same-data fitting curve

These accuracies classify strict Mixed using q>=0.5; they are not formal
question-answering EM. Every row evaluates the same2560 training questions
and512 independent diagnostic questions, generated using frozen actor0179.

| Epoch | Adam/rank | Train loss | Diagnostic loss | Train accuracy | Diagnostic accuracy |
|---|---:|---:|---:|---:|---:|
|Before|2200|1.367245|1.948716|67.6563%|48.8281%|
|1|2250|1.054514|1.944538|77.0313%|49.4141%|
|2|2300|0.854681|1.941316|84.6094%|50.9766%|
|3|2350|0.751216|1.944627|85.7813%|50.1953%|

Training balanced accuracy rose58.3925%->87.5871%; diagnostic balanced
accuracy changed55.9819%->54.2328%. Training fitting improved; these
diagnostics do not establish improved generalization. Eligible training
questions were historically exposed during EMA5 warmup, while diagnostic
questions were excluded. No threshold tuning or strict-scoring relaxation
was performed.

Metrics SHA:
`fec59b9068d34e7acc94fad0d2fb4e0eea12e4c4e93402ad4a7f32b003bdac01`.
Local scalar artifact:two-stage-periodic-000180-fit-curve.json.

## Independent probe payload qualification

The existing66f3dd86bc37 operator passed the full probe SHA and actual CPU
loading of its model, buffers, Adam and all four rank RNG states. It checked
three epoch boundaries and the source joint, and retained an immutable
remote hardlink. CUDA remained uninitialized; peak RSS was1549045760 bytes.
This qualifies the probe payload; it does not qualify full actor CPU loading,
GPU forward or a stopped-writer recovery rehearsal. All tensors stayed remote.

Head SHA:
`490d44693c83bf830a2303055a8cbb035081db40114e1f85b5765cbb46cc05b9`.
Head bytes:933674012.
Remote proof:preparation/live-probe-000180-v1-full-sha-cpu.json.
Proof SHA:
`1f70ad8d45483034b373e6680a857d0cf7726416246bd3596e6a86a7894ebd2c`.
The exact3243 proof bytes were mirrored locally and SHA-verified.

## Remaining acceptance

Actual181 has entered prefix generation. Scoring with the2350-step head and
all four actor ranks' restore/write continuity must still be observed before
claiming actual next-outer use accepted. P190, the200 milestone and fixed512
QA, final51713, checkpoint retention and owned-resource closure also remain
required. The full goal remains active.

The all-history throughput observation exceeded its bounded timeout; no
training failure was inferred. A bounded171..180 scalar query succeeded and
was merged with the already verified62..170 rows. An initial overly broad
joint-metadata mirror exceeded the small-output limit; the corrected bounded
scalar mirror succeeded. Neither observation failure changed training.

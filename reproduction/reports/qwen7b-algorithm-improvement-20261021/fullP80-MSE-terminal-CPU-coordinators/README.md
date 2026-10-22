FullP80 MSE terminal CPU coordinators

One independent durable worker per actual job:750189=seed9137/plan2026102227;750190=seed19137/plan2026102302. Root alone starts them with explicit --seed. Source-stage does not start workers or CPU readers.

Running training: only accounting, complete/controller/progress metadata; no checkpoint or model packages, no shared CPU lock. A real COMPLETED0:0 and exact10epoch/13210update complete receipt are both required. Then the worker calls the already immutable693a81df CPU entry with --epoch10. The existing validator owns COMMON_LOCK around every full checkpoint SHA/load/restore, so do not add a second outer flock on the same file. Both seed workers may wait concurrently, but their CPU loads serialize through that existing lock.

Existing exact CPU acceptance is reused only after matching actualjob/epoch/13210steps/binding/seed/physical fixture+P20/source/complete refs. Terminal training or validation failure leaves failure.json and is not blindly restarted. Unknown existing same-job worker/validator identities stop duplicate launching.

No Slurm mutation, hot changes, training cancellation or final4096 read.

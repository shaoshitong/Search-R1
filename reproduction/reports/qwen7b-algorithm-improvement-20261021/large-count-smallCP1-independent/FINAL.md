# Actual independent small CP1 acceptance

The real CP1 audit passed inside Slurm allocation 743978 using a CPU-only overlapping step: 2 CPU, GPU count 0, no CUDA initialization. The fixed srun entry sets the empty CUDA environment **inside** the step; the first launcher guard failure remains preserved and the verifier itself was not weakened.

The 8,281,973-byte CP1 and 4,175,798-byte CP0 were freshly SHA-checked and bounded-ZIP checked before CPU loading. The new 512,518-parameter head raw/EMA/Adam/RNG four components, Adam step 1, source old-four-rank archive at step 10464, cursor/order, frozen cache/config, and producer first-update exact replay agree. No legacy model or Adam payload was opened; no actor, legacy, or audit optimizer update occurred. Stored CUDA RNG bytes were hashed; no CUDA-kernel RNG restoration is claimed.

The actual receipt is `independent-CP1-acceptance.json`, 3,268 bytes, SHA256 `31c429c8998b474329100350ef200b3d17bb014e782f66ee97510e2383eb9915`. All real parameters and optimizer/RNG states remain remote. The accompanying local JSON contains hashes and acceptance metadata only. This is real CP1 acceptance, separate from the earlier synthetic fixture qualification.

# Additional matched layer pilots

L15/P20, L18/P20, L24/P20; all copy the same official Qwen2.5-7B block at zero-based index 15, 18, or 24 and use the immediately preceding block output (14, 17, or 23). The actor is frozen and retains only blocks before the copied trainable block.

The new runtime is byte-for-byte original 5a3cfa runtime except CLI layer choices expand from [21,27] to [15,18,21,24,27]. Original L21 and L27 computation is unchanged; active immutable entries and source are untouched.

All use exactly the same existing 8192-question token fixture, three epoch G5 plans and seeds, fixed train 2048 and independent 2048, 192 optimizer updates, LR5e-6, EMA0.99, 2GPU, 12CPU, 128GiB, global128/local64/micro8/accum8. Initial normal acd_u, 30min for P20 backfill. No final4096 access, no new data generation, no Slurm submission by stage controller.

Root executes submit-depth-cells-once.py remotely once; it reconciles per-cell receipt/intent/queue and validates held identity before releasing. Read-depth-cells.py is read-only.

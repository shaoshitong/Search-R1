# Additional matched layer pilots

L15/P20, L18/P20, L24/P20; all copy the same official Qwen2.5-7B block at zero-based index 15, 18, or 24 and use the immediately preceding block output (14, 17, or 23). The actor is frozen and retains only blocks before the copied trainable block.

The new runtime is byte-for-byte original 5a3cfa runtime except CLI layer choices expand from [21,27] to [15,18,21,24,27]. Original L21 and L27 computation is unchanged; active immutable entries and source are untouched.

All use exactly the same existing 8192-question token fixture, three epoch G5 plans and seeds, fixed train 2048 and independent 2048, 192 optimizer updates, LR5e-6, EMA0.99, 2GPU, 12CPU, 128GiB, global128/local64/micro8/accum8. Initial normal acd_u, 30min for P20 backfill. No final4096 access, no new data generation, no Slurm submission by stage controller.

Root executes submit-depth-cells-once.py remotely once; it reconciles per-cell receipt/intent/queue and validates held identity before releasing. Read-depth-cells.py is read-only.

## Actual matched training distribution

Fresh reads of all three original, immutable epoch plans show exactly **8192 unique questions, 3489 Mixed and 4703 nonMixed** in each epoch. This is not an actual 50/50 sample. There are only 3489 G10Mixed-eligible questions in this fixture, so the original `min(N//2, eligible)` assignment makes all 3489 eligible questions positive and chooses a genuine G5 with both correct and wrong rollouts for each. Remaining questions are homogeneous G10 and therefore homogeneous G5. No G10Mixed question is assigned a homogeneous G5 negative.

The single group-level Mixed BCE uses global class weights **1.173975351103468 for positives and 0.8709334467361258 for negatives**, making total class loss contribution balanced, despite the 3489/4703 sample counts. This is the exact existing L21/L27 objective and data distribution for a matched layer comparison. It is not the natural random-G5 control and does not repair the missing hard-negative distribution.

The target is `Y = 1[0 < sum_j strict_EM_j < 5]`; there is one direct group logit and one weighted Mixed BCE, with no trajectory BCE and no probability-product-derived Mixed loss. No plan or active runtime was changed by these observations. See `actual-training-plan-counts.json` for the three source SHA values and actual counts.

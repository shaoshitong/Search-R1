# Fresh second-seed full169028 Z replica — stage only

Authorized seed pair: init_seed19137 and train_plan_seed2026102302. Diagnostic seed2026102228, fixedtrain2048, independent2048 G5 views, accepted169028ownG10 bank and officialQwen revision stay fixed. The copied official layer21 and packed head architecture, Prefix20, pure z_mse_anchor objective (anchor1, epsilon1e-6), LR5e-6, global128, per-rank64/micro8/accum8, EMA0.9995 and10full epochs/13210updates are unchanged.

New isolated runtime `online_packed_variance_seeded.py` is copied from published7ac274ac with explicit seed plumbing, head SHA receipts and an explicit fresh/exact-resume boundary. Initialization occurs at the same point before actor loading as in the original run. Only the new runtime changes; running7ac source is untouched. The secondseed entry uses `--start-mode fresh`, rejects an existing latest.json and does not name a resume checkpoint.

Both seed values, diagnostic seed and actual initial_head_sha256 are written to controller and every checkpoint. Exact resume, if separately authorized later, requires those same fields plus the original full raw/EMA/Adam/RNG state. The source fixture is read only; no new rollout, token bank or hidden cache is generated.

CPU tests demonstrate that explicit trainplan seed changes both the actual question order and ownM/trajectory draws while remaining reproducible, and that the fresh head's initialization SHA changes between seeds9137/19137. This head component test does not load the official actor. The actual production initial-head SHA will be measured after actor loading when root later starts the job. Three CPU tests passed; GPU training has not started.

Output: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1`. Existing full fixture: `.../online-packed-full169028-M3to10-P20-20261022-v1/fixture`. Stage entry is2GPU, ordinary acd_u/normal,8hours; root owns submission and any priority change.

The old799e75f3 CPU validator is intentionally bound to the original three-arm path and seed2026102227; it is not an acceptance receipt for this new replica. Any later terminal acceptance must bind this replica's explicit seeds and new source/root. No final4096 access occurred.

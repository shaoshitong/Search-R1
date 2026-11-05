# V5: history mechanisms and feasible creator games

The Chinese main report is `docs/V5_RESEARCH_REPORT_ZH.md`. Separate reports,
protocols, acceptance receipts and aggregates are under `docs/V5_*` and
`docs/V5_AUDIT/`. All raw inputs and row-level predictions stay on HKUST3.
No GPU, media download, login or gated dataset is needed.

Use an existing Python 3.12 environment with numpy, pandas, scipy and pyarrow.
The completed runs used `/home/shaoshitong/anaconda3/bin/python` and the dataset
root `/data/shaoshitong/ai_creator_game_20261104`. Fetch missing inputs using the
V3 collectors, preserving official/pinned checksums. Do not run on the Mac by
default: raw and intermediate paths must point to remote storage.

## KuaiRand

Deploy `kuairand/` files together to `v5/kuairand/code/`. The runner captures the
actual exit code and limits numerical libraries to one CPU thread. The protocol
uses ten fixed splits, both directions, three origins and 200 matched identity
negative controls. A smoke run is functional only, not a substitute for this run.

```bash
bash src/v5/kuairand/run_remote.sh /data/shaoshitong/ai_creator_game_20261104/v5/kuairand/reproduced full
python src/v5/kuairand/summarize_diagnostics.py \
  /data/shaoshitong/ai_creator_game_20261104/v5/kuairand/reproduced \
  src/v5/kuairand/cross_item_history.py
```

The final accepted output is `v5/kuairand/final_scope/`. An earlier output in
`full/` used the shorthand “non-AI”; this was corrected to historical 2022 data
without observed AI adoption, then rerun. All numerical model/comparison/null
sections were exactly equal, recorded in the metadata-equivalence audit.

## KuaiRec

Deploy `kuairec/` files together to `v5/kuairec/code/`. The two separately frozen
protocols cover cross-old-work transfer and first-publication-day prediction.
Detailed commands are in `kuairec/README.md`. Runner arguments are the common
research root, not the `v5/kuairec` output subdirectory:

```bash
python src/v5/kuairec/run_remote.py /data/shaoshitong/ai_creator_game_20261104
python src/v5/kuairec/run_new_upload.py /data/shaoshitong/ai_creator_game_20261104
python src/v5/kuairec/verify_acceptance.py /data/shaoshitong/ai_creator_game_20261104
```

Only observed upload-calendar-day rows form the supplementary target pool.
Missing rows are never zero-filled. “Unseen” means not in the training window,
not newly registered. Q/F are past-observed catalogue counts, not complete
production. Target feedback is excluded. Attempts and true exits are preserved;
the first supplementary attempt had an integer-power dtype bug, fixed before
completed evaluation without changing the protocol.

## Integer game

The fixed 176-condition grid uses positive integer content and at least one work
per account. An analytical dominance bound covers all potentially optimal actions
within each given parameter condition; this is not a theorem over all parameters.

```bash
python src/v5/theory/weighted_integer_audit.py \
  --protocol src/v5/theory/protocol.json \
  --output-dir /data/shaoshitong/ai_creator_game_20261104/v5/theory/reproduced
python src/v5/theory/summarize_audit.py \
  /data/shaoshitong/ai_creator_game_20261104/v5/theory/reproduced/weighted_integer_results.json \
  --output /data/shaoshitong/ai_creator_game_20261104/v5/theory/reproduced/comparisons.json
```

The audit independently checks integer allocations and reported unilateral best
responses. Preserve the no-pure-equilibrium condition and strict boundary reversals.
Do not round continuous accounts or reuse an unconstrained allocation when a new
account would receive less than one work.

## Interpretation and publication

V5 does not identify true owners, AI cost shocks, a causal history effect, actual
account cold start, policy exponents or real-world Nash behavior. Frozen old-work
memory, cross-work association and new-work group calibration are different
prediction tasks. All code and protocols are published on the existing backup
branch; raw individual logs and prediction arrays are excluded. Final source SHA,
independent fetch and archive checks are in `CODE_PUBLICATION_V5.json` delivered
alongside the reports.

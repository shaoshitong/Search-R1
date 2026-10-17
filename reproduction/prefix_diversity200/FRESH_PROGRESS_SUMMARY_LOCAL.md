# Local progress summary

Run after the existing collectors have saved a consistent local snapshot:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m reproduction.prefix_diversity200.fresh_progress_summary_local \
  --root /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2 \
  --classification progress --note 'Brief annotation for this observed turn'
```

Use `--classification verified_wait` when the main task has verified an unchanged or in-flight state. Classification/note are supplied annotations, not new observations.

The only complete boundary is `LATEST_ACTUAL.latest_joint`; saved next-outer actor/probe and raw STATUS counters are separately named under `in_flight`. Same-run joint6 handoff facts from the original base remain valid. Facts, paired losses and cost rows are checked against run/outer/rank clocks and joint identities. Cost totals require the original local `COST_CURRENT_PROGRESS.json` rows; absent evidence is pending, never an invented zero. Five-outer bins and the actual latest five rounds are separate.

Updates: GOAL/REQUIREMENT derived fields, optimization cost, paired-loss components, mixed-selection trend and separate `CURRENT_PROGRESS_SUMMARY.json/.md`; STATUS receives only `derived_progress_summary`. Other STATUS fields, goal state, configuration, source evidence and full REPORT/history remain unchanged. Derived artifacts and merged STATUS are synchronized to the two established mirrors. Resource-cost snapshots retain their own observation/coverage scope.

Required local inputs are CURRENT_CONFIG and LATEST_ACTUAL; missing facts, cost, pairs or resource evidence remain pending. All validation and input-stability checks happen before writes. Run/config/fact identity errors abort without changing outputs. This standard-library tool never reads remote paths, imports Torch, calls a controller, starts QA/training, or declares final200/full51713/cleanup complete. Unit fixtures use temporary roots and explicit temporary mirrors; no real ROOT summary has been run for publication.

Schema bindings: config is checked against the exact bytes SHA in LATEST_ACTUAL; the latest joint SHA must equal that outer's scalar fact SHA. Direct facts require all eight rank identities and twenty successful writes. Cost rows require the matching complete-joint receipt SHA, including original-base outers1–6. Paired loss requires matching post-update probe, selection and own-EM SHA; a missing expected binding stays pending, a conflicting binding aborts. Historical failed timing attempts are retained; `all_latest_phase_exits_zero` reports their actual codes independently of successful joint writes.

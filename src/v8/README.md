# Reader-oriented research report

The report was rewritten from the completed research records, rather than adding
experiments or appending another technical supplement.

- `docs/AI_CREATOR_RESEARCH_SUMMARY_ZH.md`: reviewed, editable final text.
- `docs/AI_CREATOR_RESEARCH_SUMMARY_ZH.pdf`: compiled Chinese report.
- `docs/V8_DPSK_EDITORIAL_BLUEPRINT.md`: local DPSK's full editorial assessment.
- `docs/V8_DPSK_RESEARCH_REPORT_DRAFT.md`: local DPSK's completed manuscript.
- `docs/V8_AUDIT/editorial_changes.json`: corrections and cuts applied in review.
- `docs/V8_AUDIT/authoring_acceptance.json`: actual local-harness completion and
  document provenance. No model thinking trace is published.

## Actual local DPSK workflow

The installed launcher was called directly, without UI automation:

`/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh`

The headless defaults disable native account/credential services and initially
returned missing-key/sign-in errors. The task-specific patch enables the
existing native authorization, credentials and account services for this one
process and selects `deepseek-account / deepseek-flash`. It contains no secrets,
does not overwrite authentication files, and does not synchronize credentials.

The editorial and manuscript runs used separate sessions and distinct output
files. Both actually exited 0. Their prompts, receipts, input manifest and the
finished documents are preserved; the initial launch failures are not reported
as successful writing. Raw event logs, including model thinking, are not part of
the published source package.

```bash
python src/v8/run_dpsk.py src/v8/dpsk_editorial_prompt.txt new_editorial_run
python src/v8/run_dpsk.py src/v8/dpsk_report_prompt.txt new_manuscript_run
```

These commands require the user's existing installed/logged-in Harness. They
start new writing sessions and should use new run names.

## Review and rebuild

`review_draft.py` applies the documented numerical and clarity review to the
archived DPSK draft. It preserves the original. The report removes version
history, operational details, repeated definitions, arbitrary parameter-grid
counts and long proofs, while keeping adverse results that limit the claims.
It corrects test-window overlap, probability denominators, unsupported
normalization explanations, an incorrect cost-only opening claim, an overloaded
symbol and the analytic example's missing alpha.

```bash
python src/v8/review_draft.py
python src/v8/plot_report.py
python src/v8/render_report.py docs/AI_CREATOR_RESEARCH_SUMMARY_ZH.md \
  docs/AI_CREATOR_RESEARCH_SUMMARY_ZH.pdf
```

Rebuilding uses existing numpy/matplotlib, Pandoc, Typst, Chinese fonts and
Poppler. The two vector figures read three retained aggregate JSON files under
`docs/V7_AUDIT`; no raw observations or new model fitting are used. Visual QA
renders every page. Earlier reports, datasets, calculations and negative
results remain archived.

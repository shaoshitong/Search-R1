# V7: persistent works, account accumulation, and forward-looking identity choices

This extension keeps earlier versions unchanged. It contains a finite-horizon
game, a derivation package, actual time-held-out exposure prediction, a new
public-data collection, and a Chinese compiled research supplement.

## Read the results

- `docs/AI_CREATOR_GAME_STATE_MODEL_ZH_V7.pdf`: eight-page Chinese supplement.
- `docs/V7_DERIVATION_PACKAGE.md`: assumptions, global fixed-state threshold,
  exact two-period example, and scope of dynamic equilibrium.
- `docs/V7_EMPIRICAL_PROTOCOL.md`: frozen KuaiRec prediction protocol.
- `docs/V7_DATA_AND_LITERATURE.md`: primary sources, public-source access, and
  which theoretical claims each data source can and cannot identify.
- `docs/V7_AUDIT/`: small aggregate results, hashes, actual exits, and acceptance.

These paths are relative to the repository root. No raw observations, video
identifiers, account-level contribution arrays, credentials, or environments are
distributed in this source package.

## Theory reproduction

Requires Python 3.12, numpy, and scipy. On HKUST3 the completed run used
`/home/shaoshitong/anaconda3/bin/python`, one CPU thread and nice=10.

```bash
OPENBLAS_NUM_THREADS=1 python src/v7/theory/dynamic_game.py \
  --protocol src/v7/theory/protocol.json --output /remote/v7/theory/main
OPENBLAS_NUM_THREADS=1 python src/v7/theory/verify_dynamic.py /remote/v7/theory/main
OPENBLAS_NUM_THREADS=1 python src/v7/theory/dynamic_game.py \
  --protocol src/v7/theory/protocol_alpha1_supplement.json \
  --output /remote/v7/theory/linear
OPENBLAS_NUM_THREADS=1 python src/v7/theory/verify_dynamic.py /remote/v7/theory/linear
```

The main grid has 48 conditions; the separately disclosed, result-motivated
linear grid has 24. Account and production limits are explicit economic
constraints. Complete nonempty-support LP enumeration handles mixed equilibria
when pure equilibria are absent. Continuation equilibrium selection is a
specified convention; the output is not a uniqueness theorem.

The initial threshold audit used a needlessly tight production-search bound.
The corrected safe bound is `m+R/c`; original run details are preserved remotely
under `theory/results`, and the final main results are under
`theory/results_boundfix`. All 48 dynamic equilibria and their comparisons were
unchanged; the final independent audit passed. The correction is not a change to
the frozen game grid.

## Data reproduction

KuaiRec source SHA256:
`45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b`.
Raw data stays on HKUST3 under
`/data/shaoshitong/ai_creator_game_20261104/v3/kuairec/`.

```bash
OPENBLAS_NUM_THREADS=1 python src/v7/empirical/analyze_states.py \
  --source /remote/raw/item_daily_features.csv \
  --protocol src/v7/empirical/protocol.json --output /remote/v7/empirical/results
```

The score aggregates exactly to `h**beta * T**alpha`; this is a bridge between
the empirical and theoretical scoring families. Exposure-memory proxies do not
identify physical inventory, non-transferable account capital, causal ranking
elasticities, AI costs, or real-owner identity splitting. The state-feature
one-day forecast is a genuine future-observation task, not an attempt to validate
a state update by its own definition.

Public Open Channel Stats collection and independent file verification are in
`src/v7/discovery/`; the separate empirical pilot is in `src/v7/empirical/`.
Only public manifest-listed channels were requested. Missing impressions remain
missing; views and impressions are different metrics. Seven public entries,
six with observations, and four eligible forecasting channels have different
denominators and must not be conflated.

## Figures and PDF

The completed Mac build used existing matplotlib 3.10.8, numpy, Typst 0.14.2,
Hiragino Sans GB/Songti SC fonts, and Poppler for visual QA. No local environment
was created.

```bash
python src/v7/report/plot_results.py
typst compile --root . src/v7/report/report.typ docs/AI_CREATOR_GAME_STATE_MODEL_ZH_V7.pdf
```

Only aggregate JSON drives the four vector figures. The PDF is a research
supplement; policies, state dynamics, and identity behaviour are not claimed to
have all been causally validated.

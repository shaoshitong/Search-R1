# AI creator exposure game: course research project

This source package studies a platform choosing a within-account quantity elasticity and creators choosing production and identities. It combines analytic equilibria with descriptive Civitai metadata and a hypothetical allocation replay. It makes no claim to a new general theory or a causal estimate of a deployed recommender.

## Reproduce on HKUST

The source snapshot is public and non-gated, pinned to Hugging Face dataset `thefcraft/civitai-stable-diffusion-337k`, commit `758d4d72c87c5177c349230f38f2e8e8e3b6a23c`. The downloader uses a public mirror and checks its LFS SHA256. Its metadata card does not declare a license; raw records are not redistributed in this repository.

Use an existing Python environment with the packages in `requirements.txt`. In the completed run, `/home/shaoshitong/anaconda3/bin/python` on HKUST3 supplied pandas 2.2.2 and pyarrow 22.0.0. No training or GPU is required.

```bash
python src/collect_civitai.py /data/shaoshitong/ai_creator_game_20261104
python src/summarize_civitai.py /data/shaoshitong/ai_creator_game_20261104
python src/analyze_civitai.py \
  /data/shaoshitong/ai_creator_game_20261104/raw/train-00000-of-00001-ace5b28cebba25a7.parquet \
  /data/shaoshitong/ai_creator_game_20261104/results
```

The completed snapshot has 327,138 rows before duplicate image IDs are removed. Counts from a historical snapshot are not complete production histories. SFW means the source boolean only; no images are fetched or displayed. Public usernames are accounts, not identified economic owners. Prompt reuse does not identify plagiarism, content quality, or common ownership. Engagement capture time is unknown.

## Files

- `src/collect_civitai.py`: pinned metadata download, actual byte-progress log and SHA256 check.
- `src/summarize_civitai.py`: image-ID deduplication, source diagnostics, fixed-seed samples.
- `src/analyze_civitai.py`: aggregate descriptive analysis, allocation replay and figures.
- `docs/`: model assumptions, proofs, research plan and source bibliography.

Data and large intermediate files remain on the remote host. The repository contains source and necessary documentation only. AI tools assisted with literature discovery and model review; all quantitative claims must be tied to the corresponding output, and all theoretical claims to the stated assumptions.

## V2: measurement robustness and exact integer games

`docs/CURRENT_THEORY_ZH_V2.md` explains the current model and its identification limits in Chinese. `docs/V2_DATA_AND_VALIDATION.md` reports the expanded Civitai snapshot, independent DiffusionDB metadata, a separate 2026 model-version background source, and exact-integer mechanism checks. Small source receipts and aggregate results are in `docs/V2_AUDIT/`; all raw data remain on HKUST3.

Run the V2 data commands in `src/v2/data/README.md` from a directory containing the three V2 data scripts, or substitute their full paths. Reproduce the account-label audit with `python src/v2/data/audit_account_labels.py /data/shaoshitong/ai_creator_game_20261104/v2`. It uses the V1 raw snapshot in the parent directory.

```bash
python src/v2/finite_market.py --output /data/shaoshitong/ai_creator_game_20261104/v2/theory/finite_market_results.json
python src/v2/check_strictness.py --output /data/shaoshitong/ai_creator_game_20261104/v2/theory/reproduced_strictness.json
```

These finite-game calculations are not human behavioral experiments. In particular, the hypothetical two-account gain near 41.4% under square-root scoring follows largely from the scoring formula, not from a fitted real-world effect. The expanded public metadata do not identify actual impressions, economic ownership, cost shocks, or a deployed recommendation exponent.

## V3: real logged exposure and held-out prediction

`src/v3/README.md` gives the acquisition and analysis commands. V3 adds the official
KuaiRand-Pure archive, the verified KuaiRec daily-feature CSV, and stable account IDs
from the existing Civitai model-version snapshot. The Chinese V3 report and catalog
are in `docs/`, with small numerical/source audits in `docs/V3_AUDIT/`.

KuaiRand offers modest cross-entropy support for a concave conditional quantity
score in one tab; total variation does not improve. KuaiRec does not replicate the
concavity result. Historical exposure shares predict best in both. These are
non-AI historical recommendation datasets, so no AI cost shock, economic-owner
identity choice, or observed Nash equilibrium is identified.

## V4: interpreting persistence and refining the game

`docs/THEORY_REFINEMENT_HISTORY_V4.md` separates stable creator heterogeneity,
account capital and old-item persistence. It audits the historical-share baseline's
high-dimensional account information and explains why account fixed effects absorb
the static-pool quantity exponent. It derives an optional weighted-account contest
with cold-start disadvantage, including a heterogeneous-type/cost one-period
equilibrium and conditional AI cost comparative statics. These are analytical model
results, not new empirical estimates. Dynamic inventory, reputation and identity
entry remain unsolved extensions. V3 data outputs and archived source remain frozen.

## V5: mechanism diagnostics and feasible integer games

`src/v5/README.md` supplies the reproduction entry points. V5 tests cross-work
history transfer at rolling origins, evaluates genuine newly uploaded works using
only past catalogue/history features, and audits 176 integer creator-game
conditions with minimum-content constraints. Shared history gains are weak and
concentrated; first-day quantity features mainly correct seen/unseen group mass.
Cold-start capital and AI cost effects remain unidentified. The Chinese main
report, separate analyses, negative results and AI-based review are in `docs/V5_*`.

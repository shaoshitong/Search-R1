# V3: observed exposure validation

All raw files and numerical intermediate arrays stay on HKUST3, under
`/data/shaoshitong/ai_creator_game_20261104/v3/`. No GPU or new environment is needed.
Use an existing Python 3.12 environment with pandas, pyarrow, numpy and scipy.

```bash
python src/v3/exposure/collect_kuairand_ranges.py /data/shaoshitong/ai_creator_game_20261104/v3/kuairand
python src/v3/data/acquire_item_daily_replica.py /data/shaoshitong/ai_creator_game_20261104/v3/kuairec
python src/v3/exposure/analyze_kuairand.py /data/shaoshitong/ai_creator_game_20261104/v3/kuairand
python src/v3/exposure/analyze_kuairec.py /data/shaoshitong/ai_creator_game_20261104/v3/kuairec
```

The official KuaiRand archive MD5 is checked after supported HTTP Range transfers.
If a connection ends early, rerun the range collector to resume missing suffixes.
`collect_kuairand.py` preserves the original single-stream method and its optional
local-transfer mode; single streams failed in the completed run. Do not interpret
a partial download as a completed dataset.

KuaiRec's public, pinned single-CSV replica is checked against its LFS SHA256 and
the official ZIP member's byte length and CRC32. The full official ZIP MD5 was not
verified; its failed attempt is kept in the audit. No gated sources are accessed.

KuaiRand's primary protocol uses Apr 22–26 training, Apr 27–28 validation and Apr
29–May 8 testing. The fixed pool is uploaded before Apr 22. Models are fitted on
training only; monthly future statistic features are excluded. Only tab 1 has
sufficient same-tab random exposure. Viewer `user_id` is never treated as creator
identity. For post hoc age-bound sensitivity, run the same analysis with
`--age-cap 0`, `--age-cap 4`, `--age-cap 8`, each with a distinct `--result-name`.
Report all caps and retain the primary age cap 2; do not select by test loss.

KuaiRec uses a retrospective balanced 63-day common-item panel; future row presence
defines the sample but future outcome values are not used for fitting. Missing
item-day rows are not imputed to zero. Only 46 authors have multiple selected items.

Model loss is per-exposure cross entropy; uncertainty is exploratory resampling of
test days conditional on fitted models and the selected pool. It does not identify
causal recommendation exponents, AI cost effects, true owners or Nash equilibria.
Quantity models omit persistent heterogeneity and lose to historical-share baselines.

The public Civitai probe is diagnostic only: HTTP000/TLS/connection failures did
not yield live snapshots. Account-ID auditing uses the existing 2026 SQLite source,
not credentials or inferred economic ownership. See the Chinese V3 report and data
catalog in `docs/` for exact scope and source attribution.

SeaArt's eight metadata-only Parquets are collected by
`data/collect_seaart_metadata.py ROOT/seaart` and summarized by
`data/summarize_seaart_metadata.py ROOT/seaart`. Each file is pinned and checked
against LFS SHA256. The source has no author IDs, no verified recommendation
impressions, no per-item AI/HGC labels and no declared license. Like counters often
exceed view counters; no like/view probability, game fit or media download is made.

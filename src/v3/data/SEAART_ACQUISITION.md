# SeaArt metadata-only acquisition

Source: https://huggingface.co/datasets/lehduong/seaart-hq

Pinned commit: `4321a586faf084ddfa6ab415caae4073af13aa67`.
The public repository API reports `gated=false`. All eight Parquets are metadata
tables containing media URLs, not embedded image bytes. No URL in a row is fetched.
The card does not declare a license. The collector pins the API's byte counts and
LFS SHA256 values, downloads on HKUST3 through a public mirror, and validates every
file. Raw files stay on remote storage; local deliverables contain aggregate audits.

```sh
/home/shaoshitong/anaconda3/bin/python collect_seaart_metadata.py /data/shaoshitong/ai_creator_game_20261104/v3/seaart
/home/shaoshitong/anaconda3/bin/python summarize_seaart_metadata.py /data/shaoshitong/ai_creator_game_20261104/v3/seaart
```

The summary reads only IDs, timestamp-like fields, counters, and source codes.
It does not read prompts/titles into the analysis table, download media, fit a
regression, or calculate like/view probabilities. It reports raw rows and a
deterministic first-occurrence item-ID deduplication separately. `num_of_view` is
not verified impressions. No author-ID field, economic owner map, explicit AI/HGC
label, or documented NSFW-code mapping was found. The millisecond interpretation
of `created_at` does not establish whether it denotes upload or capture time.

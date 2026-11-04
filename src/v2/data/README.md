# V2 data acquisition and measurement robustness

All raw and row-level data are on `hkust3` under
`/data/shaoshitong/ai_creator_game_20261104/v2/`. V1 files were not changed.
The Mac contains only these scripts and small aggregate audit JSON files.
No images, model weights, credentials or gated files were downloaded.

## Acquired sources

1. `poloclub/diffusiondb`, commit
   `fb620fbe49fa4420e0734bd9c0df11f51176b61f`: public/non-gated,
   CC0-1.0, 194,548,652-byte `metadata.parquet`. This is the independent
   historical Stable Diffusion Discord gallery, not a Civitai snapshot.
2. `hanruijiang/civitai-stable-diffusion-2.5m`, commit
   `d0bbe735e7092473328196adb677534056997879`: public/non-gated,
   Apache-2.0 declared by the card, 767,369,614-byte JSONL gzip containing
   image IDs, author usernames, creation dates, prompts, NSFW labels and stats.
   The source is inspired by the 337k collection, so overlap is assessed by ID.
3. `Shiba-2-shiba/civitai-model-version-trends`, commit
   `fb10c1c476d8201c496650d77c0aa33d46087248`: public/non-gated,
   no declared license, 234,590,860-byte SQLite gzip. Model-version supply,
   observed through 2026-11-03, is a different unit from posted image content.

All three files were downloaded directly to HKUST3 through `hf-mirror.com`
and matched the public repository's LFS SHA256 and byte count. Download
receipts include repeated byte-growth observations and final exit status.
The first Civitai-2.5m attempt returned HTTP429; a later normal download with
`?download=true` and bounded retry succeeded and passed the same LFS hash.

Civiverse-6M was checked and excluded: its public API reports `gated=manual`.
ImageGem remained excluded as gated. Civistash was public but offered large
media-mixed tar archives without available converted metadata parquet, so it
was not acquired.

## Commands on HKUST3

```sh
/home/shaoshitong/anaconda3/bin/python code/download_public_metadata.py diffusiondb_2m /data/shaoshitong/ai_creator_game_20261104/v2
/home/shaoshitong/anaconda3/bin/python code/download_public_metadata.py civitai_2_5m /data/shaoshitong/ai_creator_game_20261104/v2
/home/shaoshitong/anaconda3/bin/python code/download_public_metadata.py civitai_model_trends_2026 /data/shaoshitong/ai_creator_game_20261104/v2
/home/shaoshitong/anaconda3/bin/python code/summarize_v2.py diffusiondb /data/shaoshitong/ai_creator_game_20261104/v2
/home/shaoshitong/anaconda3/bin/python code/summarize_v2.py civitai /data/shaoshitong/ai_creator_game_20261104/v2
/home/shaoshitong/anaconda3/bin/python code/summarize_v2.py model_trends /data/shaoshitong/ai_creator_game_20261104/v2
```

## Measurement definitions

- A normalized prompt is lowercased, whitespace-collapsed and trimmed. This
  is textual equality, not semantic similarity or a content-quality score.
- Duplicate excess is `(nonempty records - unique values) / nonempty records`.
  Within-account duplication uses unique `(account, normalized prompt)` pairs.
- Account-volume quantiles and the top-10% count share exclude missing and
  `deleted_account` identifiers. The denominator is known-account records.
- DiffusionDB is described as image entries. Approximate request units merge
  `(user_name, timestamp, normalized prompt)` only when all three are present;
  incomplete records stay separate. These are not authenticated request IDs.
- DiffusionDB threshold variants require both predicted image and prompt NSFW
  scores to be at most 0.1 or 0.2. They are not verified SFW annotations.
- Civitai analysis removes repeated image IDs before profiling. Cross-snapshot
  comparisons distinguish joint-nonempty prompt agreement, known-username
  agreement, and unchanged/increased/decreased reaction totals. Reaction totals
  consistently sum likes, hearts, laughs and cries across the two snapshots.

## What these data can establish

They provide external robustness of redundancy and account-volume descriptions,
sensitivity to generated-image versus approximate-request units, and changes in
engagement measurements for identical Civitai image IDs across source snapshots.

They do not observe economic-person account ownership, impressions/attention
shares, production cost or an exogenous cost shock, or the actual recommendation
exponent. Thus neither more records nor similar prompts across usernames verify
real account splitting, the model's equilibrium, or its causal comparative statics.
Source snapshot names are not precise per-record engagement capture timestamps.

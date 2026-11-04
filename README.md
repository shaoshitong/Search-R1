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

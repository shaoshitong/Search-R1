#!/usr/bin/env python3
"""Summarize metadata and draw reproducible SFW samples; never fetch images."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_stats(value):
    if isinstance(value, dict):
        return value
    return ast.literal_eval(value) if isinstance(value, str) and value else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    path = args.root / "raw/train-00000-of-00001-ace5b28cebba25a7.parquet"
    df = pd.read_parquet(path)
    print(json.dumps({"stage": "loaded", "rows": len(df), "columns": list(df)}), flush=True)
    source_rows = len(df)
    duplicate_image_ids = int(df["id"].duplicated().sum())
    df = df.drop_duplicates("id", keep="first").copy()
    dates = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
    sfw = df.loc[df["nsfw"].eq(False)].copy()
    stats = df["stats"].map(parse_stats)
    reactions = stats.map(lambda value: sum(value.get(k, 0) or 0 for k in
                                            ["likeCount", "heartCount", "laughCount", "cryCount"]))
    comments = stats.map(lambda value: value.get("commentCount", 0) or 0)
    prompt = df["prompt"].fillna("").astype(str)
    normalized_prompt = prompt.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    nonempty = normalized_prompt.ne("")
    sfw_prompt = normalized_prompt.loc[sfw.index]
    sfw_nonempty = sfw_prompt.ne("")
    author_counts = sfw.dropna(subset=["username"]).groupby("username").size()
    random_sample = sfw.sample(n=min(10000, len(sfw)), random_state=42)
    eligible = author_counts.loc[author_counts.ge(5)].index.to_numpy()
    rng = np.random.default_rng(42)
    chosen = rng.choice(eligible, size=min(1000, len(eligible)), replace=False)
    panel = sfw.loc[sfw["username"].isin(chosen)].sample(frac=1, random_state=42)
    panel = panel.groupby("username", sort=False, group_keys=False).head(10)
    sample_receipts = []
    for name, sample in [("random_sfw_10000.parquet", random_sample),
                         ("author_stratified_sfw.parquet", panel)]:
        output = args.root / "derived" / name
        output.parent.mkdir(exist_ok=True)
        sample.to_parquet(output, index=False)
        sample_receipts.append({"path": str(output), "rows": len(sample),
                                "authors": int(sample["username"].nunique()),
                                "bytes": output.stat().st_size, "sha256": sha256(output)})
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256(path), "source_rows": source_rows, "analysis_rows": len(df),
        "schema": str(pq.read_schema(path)), "columns": list(df),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "unique_image_ids": int(df["id"].nunique()),
        "duplicate_image_ids": duplicate_image_ids,
        "deduplication": "Retain first occurrence of each image id before all analytical summaries and sampling.",
        "authors": int(df["username"].nunique()),
        "author_missing": int(df["username"].isna().sum()),
        "date_min": str(dates.min()), "date_max": str(dates.max()),
        "invalid_dates": int(dates.isna().sum()),
        "date_month_counts": dates.dt.strftime("%Y-%m").value_counts().sort_index().to_dict(),
        "nsfw_counts": {str(k): int(v) for k, v in df["nsfw"].value_counts(dropna=False).items()},
        "sfw_rows": len(sfw), "sfw_authors": int(sfw["username"].nunique()),
        "sfw_authors_at_least_5_images": len(eligible),
        "prompt_nonempty": int(nonempty.sum()),
        "prompt_duplicate_excess_fraction": float(normalized_prompt.loc[nonempty].duplicated().mean()),
        "sfw_prompt_nonempty": int(sfw_nonempty.sum()),
        "sfw_prompt_duplicate_excess_fraction": float(sfw_prompt.loc[sfw_nonempty].duplicated().mean()),
        "missing_fractions": {k: float(v) for k, v in df.isna().mean().items()},
        "reactions_quantiles": {str(k): float(v) for k,v in reactions.quantile([0,.25,.5,.75,.9,.99,1]).items()},
        "comments_quantiles": {str(k): float(v) for k,v in comments.quantile([0,.5,.9,.99,1]).items()},
        "samples": sample_receipts,
        "sample_seed": 42,
        "sample_method": "Uniform SFW image sample; separately choose up to 1000 authors with >=5 SFW images uniformly, then at most 10 random images each.",
        "software": {"pandas": pd.__version__, "pyarrow": pyarrow.__version__},
        "limitations": [
            "Historical snapshot, not a contemporary 2026 panel.",
            "Original API sampling frame and capture time are not independently validated.",
            "NSFW is the source boolean; it is not a validated safety annotation.",
            "Creation time is observable; engagement capture time is not.",
            "Prompt repetition is a redundancy proxy, not a content-quality score.",
            "Within-snapshot creator counts are not complete creator production histories.",
            "No recommendation impressions, author effort, or creator earnings are observed.",
            "No license declared in dataset-card metadata; no redistribution of raw data.",
        ],
    }
    audit = args.root / "audit/metadata_summary.json"
    audit.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

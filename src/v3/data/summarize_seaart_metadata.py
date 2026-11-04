#!/usr/bin/env python3
"""Audit SeaArt counters and identifiers; do not interpret views as impressions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from collect_seaart_metadata import COMMIT, FILES, REPO

COUNTERS = ["num_of_view", "num_of_like", "num_of_collection", "num_of_task"]
AUDIT_COLUMNS = ["id", "created_at", "nsfw", "rating"] + COUNTERS


def counters(frame):
    result = {}
    for column in COUNTERS:
        values = pd.to_numeric(frame[column], errors="coerce")
        result[column] = {
            "missing": int(frame[column].isna().sum()),
            "invalid_numeric": int((frame[column].notna() & values.isna()).sum()),
            "zero": int(values.eq(0).sum()), "negative": int(values.lt(0).sum()),
            "positive": int(values.gt(0).sum()),
            "min": float(values.min()), "max": float(values.max()),
            "sum": float(values.sum()),
        }
    views = pd.to_numeric(frame["num_of_view"], errors="coerce")
    likes = pd.to_numeric(frame["num_of_like"], errors="coerce")
    result["paired_counter_checks"] = {
        "both_observed": int((views.notna() & likes.notna()).sum()),
        "likes_greater_than_views": int(likes.gt(views).sum()),
        "positive_likes_with_zero_views": int((likes.gt(0) & views.eq(0)).sum()),
    }
    return result


def date_summary(frame):
    raw = frame["created_at"]
    dates = pd.to_datetime(raw, unit="ms", utc=True, errors="coerce")
    return {
        "field": "created_at", "interpretation": "Unix milliseconds interpreted in UTC; source semantics undocumented",
        "minimum": dates.min().isoformat(), "maximum": dates.max().isoformat(),
        "missing": int(raw.isna().sum()),
        "invalid_under_unix_ms_interpretation": int((raw.notna() & dates.isna()).sum()),
        "distinct_timestamps": int(raw.nunique()),
        "distinct_utc_dates": int(dates.dt.date.nunique()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    frames, file_summaries = [], []
    all_fields = set()
    for name, expected_size, expected_sha in FILES:
        parquet = pq.ParquetFile(args.root / "raw" / name)
        fields = parquet.schema_arrow.names
        all_fields.update(fields)
        frame = parquet.read(columns=AUDIT_COLUMNS).to_pandas()
        frames.append(frame)
        file_summaries.append({"filename": name, "rows": len(frame),
                               "fields": fields, "dtypes": {f.name: str(f.type) for f in parquet.schema_arrow}})
        print(json.dumps({"stage": "read_metadata", "filename": name, "rows": len(frame)}), flush=True)
    frame = pd.concat(frames, ignore_index=True)
    known = frame["id"].notna() & frame["id"].astype(str).str.len().gt(0)
    unique = frame.loc[known].drop_duplicates("id", keep="first")
    duplicated = frame.loc[known & frame["id"].duplicated(keep=False)]
    metric_disagreements = (
        duplicated.groupby("id")[COUNTERS + ["created_at", "nsfw"]].nunique(dropna=False).gt(1).any(axis=1)
    )
    author_fields = sorted(c for c in all_fields if any(t in c.lower() for t in ["author", "creator", "owner", "user", "account"]))
    summary = {
        "repository": REPO, "commit": COMMIT,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(args.root / "raw"),
        "record_unit": "source SeaArt item metadata row; single archival collection",
        "files": file_summaries, "raw_rows": len(frame), "missing_or_empty_id_rows": int((~known).sum()),
        "unique_known_item_ids": int(frame.loc[known, "id"].nunique()),
        "duplicate_known_id_excess_rows": int(frame.loc[known, "id"].duplicated().sum()),
        "duplicate_id_groups_with_counter_time_or_nsfw_disagreement": int(metric_disagreements.sum()),
        "date_summary_all_rows": date_summary(frame),
        "counters_all_rows": counters(frame),
        "deduplicated_known_ids_keep_first": {"rows": len(unique), "date_summary": date_summary(unique), "counters": counters(unique)},
        "nsfw_code_counts_all_rows": {str(k): int(v) for k, v in frame["nsfw"].value_counts(dropna=False).items()},
        "rating_code_counts_all_rows": {str(k): int(v) for k, v in frame["rating"].value_counts(dropna=False).items()},
        "available_author_or_account_fields": author_fields,
        "stable_author_id_available": False,
        "explicit_item_ai_human_label_available": False,
        "license_declared_by_source": None,
        "media_downloaded": False, "like_view_probability_computed": False, "regression_fitted": False,
        "limitations": [
            "num_of_view is a source browsing counter, not verified recommendation impressions.",
            "No creator account ID, economic owner map, production cost, policy shock, or repeated capture time is provided.",
            "created_at is parsed as Unix milliseconds; upload-versus-collection semantics are not documented.",
            "Source NSFW codes have no documented mapping here; no SFW label or filtering is asserted.",
            "AI origin is platform context; no explicit per-item AI-versus-human annotation was found.",
            "Deduplication keeps the first row in deterministic split order, not a proven latest observation.",
            "Views and likes may have different counting scopes or update timing; no like/view probability is estimated.",
            "The source provides no dataset card license; only small aggregate audits are distributed.",
        ],
    }
    audit_root = args.root / "audit"
    audit_root.mkdir(exist_ok=True)
    (audit_root / "seaart_metadata_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"completed": True, "rows": len(frame), "unique_known_item_ids": summary["unique_known_item_ids"],
                      "likes_greater_than_views": summary["counters_all_rows"]["paired_counter_checks"]["likes_greater_than_views"],
                      "stable_author_id_available": False}), flush=True)


if __name__ == "__main__":
    main()

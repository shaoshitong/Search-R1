#!/usr/bin/env python3
"""Describe the observed KuaiRec item-day table, without fitting a policy model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


def quantiles(series):
    return {str(k): float(v) for k,v in series.quantile([0,.25,.5,.75,.9,.95,.99,1]).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    path = args.root / "item_daily_features.csv"
    df = pd.read_csv(path)
    print(json.dumps({"stage": "loaded", "rows": len(df), "columns": list(df)}), flush=True)
    known = df["author_id"].notna() & df["author_id"].ge(0)
    dates = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    uploads = pd.to_datetime(df["upload_dt"], errors="coerce")
    pairs = df[["video_id", "date"]].drop_duplicates()
    author_items = df.loc[known, ["author_id", "video_id"]].drop_duplicates().groupby("author_id").size()
    public = df.loc[known & df["visible_status"].eq("public")]
    public_author_items = public[["author_id", "video_id"]].drop_duplicates().groupby("author_id").size()
    public_daily_inventory = public.groupby(["date", "author_id"])["video_id"].nunique()
    public_multi = public_daily_inventory.loc[public_daily_inventory.ge(2)]
    daily = df.groupby("date").agg(rows=("video_id", "size"), items=("video_id", "nunique"),
                                   authors=("author_id", "nunique"), show_cnt=("show_cnt", "sum"),
                                   zero_exposure_rows=("show_cnt", lambda x: x.eq(0).sum()),
                                   missing_exposure_rows=("show_cnt", lambda x: x.isna().sum()))
    daily.to_csv(args.root / "daily_aggregate.csv")
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(path), "record_unit": "observed video × statistics date",
        "raw_rows": len(df), "columns": list(df), "column_count": len(df.columns),
        "dtypes": {k:str(v) for k,v in df.dtypes.items()},
        "unique_video_date_pairs": len(pairs),
        "duplicate_video_date_rows": int(df.duplicated(["video_id", "date"]).sum()),
        "items": int(df["video_id"].nunique()),
        "known_author_ids": int(df.loc[known, "author_id"].nunique()),
        "missing_or_negative_author_rows": int((~known).sum()),
        "items_with_multiple_known_author_ids": int(df.loc[known].groupby("video_id")["author_id"].nunique().gt(1).sum()),
        "date_min": str(dates.min()), "date_max": str(dates.max()), "distinct_dates": int(df["date"].nunique()),
        "upload_date_min": str(uploads.min()), "upload_date_max": str(uploads.max()),
        "missing_upload_date_rows": int(uploads.isna().sum()),
        "visible_status_counts": {str(k):int(v) for k,v in df["visible_status"].value_counts(dropna=False).items()},
        "video_type_counts": {str(k):int(v) for k,v in df["video_type"].value_counts(dropna=False).items()},
        "authors_with_at_least_two_observed_items": int(author_items.ge(2).sum()),
        "observed_items_per_author_quantiles": quantiles(author_items),
        "authors_with_at_least_two_public_label_items": int(public_author_items.ge(2).sum()),
        "public_label_author_date_groups": len(public_daily_inventory),
        "public_label_author_date_groups_with_at_least_two_items": len(public_multi),
        "authors_ever_with_at_least_two_public_label_items_same_date": int(public_multi.index.get_level_values("author_id").nunique()),
        "public_label_items_per_author_date_quantiles": quantiles(public_daily_inventory),
        "observed_dates_per_item_quantiles": quantiles(pairs.groupby("video_id").size()),
        "show_cnt_missing_rows": int(df["show_cnt"].isna().sum()),
        "show_cnt_zero_rows": int(df["show_cnt"].eq(0).sum()),
        "show_cnt_negative_rows": int(df["show_cnt"].lt(0).sum()),
        "show_cnt_positive_rows": int(df["show_cnt"].gt(0).sum()),
        "show_cnt_total_over_observed_rows": int(df["show_cnt"].sum()),
        "show_cnt_quantiles": quantiles(df["show_cnt"]),
        "show_user_num_missing_rows": int(df["show_user_num"].isna().sum()),
        "show_user_num_zero_rows": int(df["show_user_num"].eq(0).sum()),
        "unobserved_pairs_in_item_by_observed_date_rectangle": int(df["video_id"].nunique()*df["date"].nunique()-len(pairs)),
        "missing_fractions": {k:float(v) for k,v in df.isna().mean().items()},
        "daily_aggregate_path": str(args.root / "daily_aggregate.csv"),
        "limitations": [
            "Video × day observations cover selected KuaiRec items, not the complete platform inventory.",
            "show_cnt is the documented daily number of shows; it is not creator revenue or economic-person attention share.",
            "Do not fill unobserved item-date pairs with zero: upload/availability and source coverage can differ.",
            "public-label inventory is an observed-status proxy; the source describes visible_status as the video's current APP state.",
            "author_id is a creator-account identifier, not a cross-account owner map.",
            "No AI-content flag, exogenous production-cost shock, or known recommendation exponent is observed.",
            "The 2026 package release date is not the date of the 2020 observations.",
        ],
    }
    (args.root / "item_daily_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

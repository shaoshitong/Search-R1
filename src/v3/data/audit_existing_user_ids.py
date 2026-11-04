#!/usr/bin/env python3
"""Recover public stable account IDs already present in the 2026 snapshot."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("research_root", type=Path)
    args = parser.parse_args()
    database = args.research_root / "v2/civitai_model_trends_2026/civitai.sqlite"
    output_root = args.research_root / "v3/civitai_existing_user_ids"
    output_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True)
    rows = []
    for model_id, label, first_seen, last_seen, raw in connection.execute(
            "select model_id,creator_username,first_seen_at,last_seen_at,raw_json from models"):
        data = json.loads(raw)
        rows.append({"model_id": model_id, "userId": data.get("userId"), "username": label,
                     "first_seen_at": first_seen, "last_seen_at": last_seen})
    frame = pd.DataFrame(rows)
    frame["userId"] = frame["userId"].astype("Int64")
    known = frame["userId"].notna() & frame["userId"].ge(0)
    mapping = frame.loc[known & frame["username"].notna(), ["userId", "username"]].drop_duplicates()
    target = output_root / "model_account_ids.parquet"
    frame.to_parquet(target, index=False)
    versions = pd.read_sql_query("select version_id,model_id,published_at from model_versions", connection)
    versions = versions.merge(frame[["model_id", "userId"]], on="model_id", how="left")
    version_target = output_root / "version_account_ids.parquet"
    versions.to_parquet(version_target, index=False)
    report = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot_commit": "fb10c1c476d8201c496650d77c0aa33d46087248",
        "models": len(frame), "models_with_known_userId": int(known.sum()),
        "distinct_known_userIds": int(frame.loc[known, "userId"].nunique()),
        "distinct_nonmissing_usernames": int(frame["username"].nunique()),
        "userIds_with_multiple_labels": int(mapping.groupby("userId").size().gt(1).sum()),
        "labels_with_multiple_userIds": int(mapping.groupby("username").size().gt(1).sum()),
        "versions": len(versions),
        "versions_with_known_userId": int((versions["userId"].notna() & versions["userId"].ge(0)).sum()),
        "files": [{"path": str(p), "bytes": p.stat().st_size} for p in [target, version_target]],
        "interpretation": "userId is a platform account identifier. It is not an economic-person identifier and does not link multiple accounts to an owner. These are model/version creators, not all posted-image creators.",
    }
    (output_root / "account_id_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    connection.close()


if __name__ == "__main__":
    main()

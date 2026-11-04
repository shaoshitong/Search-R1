#!/usr/bin/env python3
"""Metadata robustness summaries. These are not tests of economic identity."""
import argparse
import ast
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import shutil
import sqlite3

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def normalize(series):
    return series.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def known_account(series):
    return series.notna() & ~series.isin(["", "deleted_account"])


def profile(frame, user, prompt, date, label):
    normalized = normalize(frame[prompt])
    known = known_account(frame[user])
    nonempty = normalized.ne("")
    counts = frame.loc[known].groupby(user, sort=False).size()
    within = pd.DataFrame({"user": frame.loc[known & nonempty, user],
                           "prompt": normalized.loc[known & nonempty]})
    dates = pd.to_datetime(frame[date], utc=True, errors="coerce")
    return {
        "label": label, "records": len(frame), "accounts": len(counts),
        "missing_or_deleted_account_records": int((~known).sum()),
        "missing_prompt_records": int((~nonempty).sum()),
        "missing_timestamp_records": int(dates.isna().sum()),
        "date_min": str(dates.min()), "date_max": str(dates.max()),
        "per_account_record_quantiles": {str(k): float(v) for k, v in
             counts.quantile([0, .25, .5, .75, .9, .95, .99, 1]).items()},
        "top_10pct_accounts_record_share": float(counts.nlargest(math.ceil(len(counts) / 10)).sum() / counts.sum()) if len(counts) else None,
        "top_share_denominator": "records with known account identifier",
        "nonempty_prompt_records": int(nonempty.sum()),
        "global_prompt_duplicate_excess_fraction": float(normalized.loc[nonempty].duplicated().mean()) if nonempty.any() else None,
        "within_account_nonempty_prompt_records": len(within),
        "within_account_prompt_duplicate_excess_fraction": float(within.duplicated().mean()) if len(within) else None,
    }


def diffusiondb(root):
    path = root / "diffusiondb_2m/metadata.parquet"
    columns = ["image_name", "prompt", "user_name", "timestamp", "image_nsfw", "prompt_nsfw"]
    df = pd.read_parquet(path, columns=columns)
    print(json.dumps({"stage": "diffusiondb_loaded", "rows": len(df)}), flush=True)
    summary = {"source": "poloclub/diffusiondb", "record_unit": "generated image entry",
               "raw_rows": len(df), "unique_image_names": int(df["image_name"].nunique()),
               "schema": str(pq.read_schema(path)), "license": "CC0-1.0", "profiles": []}
    for threshold in [None, .1, .2]:
        subset = df if threshold is None else df.loc[df["image_nsfw"].le(threshold) & df["prompt_nsfw"].le(threshold)]
        label = "all" if threshold is None else f"both_predicted_nsfw_scores_le_{threshold}"
        summary["profiles"].append(profile(subset, "user_name", "prompt", "timestamp", label + "/image_units"))
        work = subset.copy()
        work["_normalized_prompt"] = normalize(work["prompt"])
        complete = known_account(work["user_name"]) & work["timestamp"].notna() & work["_normalized_prompt"].ne("")
        approximate = pd.concat([
            work.loc[complete].drop_duplicates(["user_name", "timestamp", "_normalized_prompt"]),
            work.loc[~complete],
        ])
        summary["profiles"].append(profile(approximate, "user_name", "prompt", "timestamp", label + "/approximate_request_units"))
    summary["limitations"] = [
        "Approximate requests merge equal (account hash, timestamp, normalized prompt); these are not confirmed requests.",
        "Timestamp can be inaccurate for duplicate images, as documented by the source.",
        "Account hash is an account identifier, not economic-person identity or cross-account ownership.",
        "Historical Discord prompt-gallery sample; no recommendation share, creator earnings, production cost or policy alpha.",
        "Low predicted NSFW thresholds are model-based filters, not verified SFW labels.",
    ]
    return summary


def convert_civitai(root):
    folder = root / "civitai_2_5m"
    source = folder / "civitai-2023-11-14-sorted.jsonl.gz"
    target = folder / "metadata_slim.parquet"
    schema = pa.schema([("id", pa.int64()), ("username", pa.string()),
                        ("createdAt", pa.string()), ("nsfw", pa.bool_()),
                        ("nsfwLevel", pa.string()), ("prompt", pa.string()),
                        ("reactions", pa.int64()), ("stats_json", pa.string())])
    rows = []
    count = 0
    with pq.ParquetWriter(target, schema, compression="zstd") as writer, gzip.open(source, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            stats = record.get("stats") or {}
            rows.append({"id": record["id"], "username": record.get("username"),
                         "createdAt": record.get("createdAt"), "nsfw": record.get("nsfw"),
                         "nsfwLevel": str(record["nsfwLevel"]) if record.get("nsfwLevel") is not None else None,
                         "prompt": record.get("prompt") or (record.get("meta") or {}).get("prompt"),
                         "reactions": sum(stats.get(k, 0) or 0 for k in ["likeCount", "heartCount", "laughCount", "cryCount"]),
                         "stats_json": json.dumps(stats, sort_keys=True, separators=(",", ":"))})
            if len(rows) == 50000:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                count += len(rows)
                print(json.dumps({"stage": "civitai_conversion", "rows": count}), flush=True)
                rows = []
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    return target


def civitai(root):
    path = convert_civitai(root)
    frame = pd.read_parquet(path)
    summary = {"source": "hanruijiang/civitai-stable-diffusion-2.5m", "record_unit": "posted image entry",
               "raw_rows": len(frame), "unique_image_ids": int(frame["id"].nunique()),
               "duplicate_image_ids": int(frame["id"].duplicated().sum()), "schema": str(pq.read_schema(path)),
               "license": "Apache-2.0 declared in dataset card", "profiles": []}
    frame = frame.drop_duplicates("id", keep="first")
    summary["profiles"].append(profile(frame, "username", "prompt", "createdAt", "all/unique_image_units"))
    sfw = frame.loc[frame["nsfw"].eq(False)]
    summary["profiles"].append(profile(sfw, "username", "prompt", "createdAt", "source_nsfw_false/unique_image_units"))
    summary["nsfw_counts"] = {str(k): int(v) for k, v in frame["nsfw"].value_counts(dropna=False).items()}
    summary["nsfw_level_counts"] = {str(k): int(v) for k, v in frame["nsfwLevel"].value_counts(dropna=False).items()}
    earlier_path = root.parent / "raw/train-00000-of-00001-ace5b28cebba25a7.parquet"
    old = pd.read_parquet(earlier_path, columns=["id", "username", "createdAt", "prompt", "stats"]).drop_duplicates("id")
    joined = old.merge(frame, on="id", suffixes=("_v1", "_v2"))
    old_reactions = joined["stats"].map(lambda value: sum(ast.literal_eval(value).get(k, 0) or 0 for k in ["likeCount", "heartCount", "laughCount", "cryCount"]))
    delta = joined["reactions"] - old_reactions
    old_prompt = normalize(joined["prompt_v1"])
    new_prompt = normalize(joined["prompt_v2"])
    both_prompts = old_prompt.ne("") & new_prompt.ne("")
    both_users = known_account(joined["username_v1"]) & known_account(joined["username_v2"])
    new_only = frame.loc[~frame["id"].isin(old["id"])]
    new_dates = pd.to_datetime(new_only["createdAt"], utc=True, errors="coerce")
    summary["v1_overlap"] = {
        "images": len(joined), "v1_unique_images": len(old),
        "fraction_of_v1": len(joined) / len(old),
        "joint_nonempty_prompts": int(both_prompts.sum()),
        "normalized_prompt_agreement_both_nonempty": float(old_prompt.loc[both_prompts].eq(new_prompt.loc[both_prompts]).mean()),
        "normalized_prompt_agreement_including_empty": float(old_prompt.eq(new_prompt).mean()),
        "joint_known_accounts": int(both_users.sum()),
        "username_agreement_both_known": float(joined.loc[both_users, "username_v1"].eq(joined.loc[both_users, "username_v2"]).mean()),
        "reactions_unchanged": int(delta.eq(0).sum()), "reactions_increased": int(delta.gt(0).sum()),
        "reactions_decreased": int(delta.lt(0).sum()),
        "reaction_delta_quantiles": {str(k): float(v) for k, v in delta.quantile([0,.25,.5,.75,.9,.99,1]).items()},
        "new_image_ids": len(new_only), "new_id_date_min": str(new_dates.min()), "new_id_date_max": str(new_dates.max()),
    }
    summary["limitations"] = [
        "Filename labels a 2023-11-14 source snapshot; exact per-record engagement capture times are not provided.",
        "Cross-snapshot reaction changes are not daily engagement increments or a causal policy panel.",
        "Same platform and overlapping image IDs; not an independent platform sample.",
        "Account username is not economic identity; account ownership, impressions, costs and ranking alpha remain unobserved.",
        "Original API coverage and selection are not independently established.",
    ]
    return summary


def model_trends(root):
    folder = root / "civitai_model_trends_2026"
    path = folder / "civitai.sqlite"
    if not path.exists():
        with gzip.open(folder / "civitai.sqlite.gz", "rb") as source, path.open("wb") as target:
            shutil.copyfileobj(source, target, 16 * 1024 * 1024)
    con = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
    queries = {
        "models": "select count(*),count(distinct creator_username),sum(creator_username is null),min(first_seen_at),max(last_seen_at) from models",
        "versions": "select count(*),count(distinct model_id),min(created_at),max(created_at),min(published_at),max(published_at),sum(created_at is null),sum(published_at is null) from model_versions",
        "run_status": "select status,execution_status,count(*) from collection_runs group by status,execution_status",
        "coverage_status": "select coverage_status,count(*) from collection_run_types group by coverage_status",
        "daily_counts": "select count(*),min(observed_date),max(observed_date),min(collected_at),max(collected_at) from daily_counts",
        "active_total_history": "select count(*),count(distinct observed_at),min(observed_at),max(observed_at) from active_total_history",
        "model_types": "select model_type,count(*) from models group by model_type",
    }
    frame = pd.read_sql_query("select v.version_id,v.published_at,m.creator_username,m.nsfw from model_versions v left join models m on m.model_id=v.model_id", con)
    counts = frame.loc[known_account(frame["creator_username"])].groupby("creator_username").size()
    example = json.loads(con.execute("select raw_json from model_versions where raw_json is not null limit 1").fetchone()[0])
    summary = {
        "source": "Shiba-2-shiba/civitai-model-version-trends", "record_unit": "model version, not generated image",
        "license": "not declared", "uncompressed_bytes": path.stat().st_size,
        "schema": con.execute("select name,sql from sqlite_master where type='table'").fetchall(),
        "query_results": {key: con.execute(query).fetchall() for key, query in queries.items()},
        "version_json_fields": sorted(example),
        "version_stats_fields": sorted((example.get("stats") or {}).keys()),
        "per_account_version_quantiles": {str(k): float(v) for k,v in counts.quantile([0,.25,.5,.75,.9,.95,.99,1]).items()},
        "top_10pct_accounts_version_share": float(counts.nlargest(math.ceil(len(counts)/10)).sum()/counts.sum()),
        "limitations": [
            "LoRA and Checkpoint supply are different product units from posted image content.",
            "All retained per-type coverage rows are partial; absent coverage is not a verified zero.",
            "created_at is missing; published_at is the usable publication field.",
            "First-seen/last-seen fields do not provide historical per-version engagement snapshots.",
            "No economic-person account mapping, recommendation share, cost shock or known ranking alpha.",
        ],
    }
    con.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["diffusiondb", "civitai", "model_trends"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = {"diffusiondb": diffusiondb, "civitai": civitai, "model_trends": model_trends}[args.source](args.root)
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (args.root / "audit").mkdir(exist_ok=True)
    output = args.root / "audit" / (args.source + "_summary.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

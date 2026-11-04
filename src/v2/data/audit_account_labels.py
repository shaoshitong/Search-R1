#!/usr/bin/env python3
"""Audit changes in observed account labels; do not infer common ownership."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
from summarize_v2 import known_account, profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    old = pd.read_parquet(args.root.parent / "raw/train-00000-of-00001-ace5b28cebba25a7.parquet",
                          columns=["id", "username", "createdAt", "prompt", "nsfw"]).drop_duplicates("id")
    new = pd.read_parquet(args.root / "civitai_2_5m/metadata_slim.parquet", columns=["id", "username"])
    joined = old[["id", "username"]].merge(new, on="id", suffixes=("_v1", "_v2"))
    known = known_account(joined["username_v1"]) & known_account(joined["username_v2"])
    paired = joined.loc[known]
    exact = paired["username_v1"].eq(paired["username_v2"])
    folded = paired["username_v1"].str.casefold().eq(paired["username_v2"].str.casefold())
    changed = paired.loc[~exact]
    transitions = changed.groupby(["username_v1", "username_v2"]).size()
    mapping = changed[["username_v1", "username_v2"]].drop_duplicates()
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "overlap_images_with_both_account_labels": len(paired),
        "exact_agreement_fraction": float(exact.mean()),
        "casefold_agreement_fraction": float(folded.mean()),
        "changed_label_image_records": len(changed),
        "changed_label_distinct_old_labels": int(changed["username_v1"].nunique()),
        "changed_label_distinct_new_labels": int(changed["username_v2"].nunique()),
        "changed_label_distinct_pairs": len(mapping),
        "old_labels_mapping_to_multiple_new_labels": int(mapping.groupby("username_v1").size().gt(1).sum()),
        "new_labels_receiving_multiple_old_labels": int(mapping.groupby("username_v2").size().gt(1).sum()),
        "ten_largest_label_transitions_fraction_of_changed_images": float(transitions.nlargest(10).sum()/transitions.sum()),
        "v1_profiles_same_definitions": [
            profile(old, "username", "prompt", "createdAt", "v1/all_unique_image_units"),
            profile(old.loc[old["nsfw"].eq(False)], "username", "prompt", "createdAt", "v1/source_nsfw_false_unique_image_units"),
        ],
        "interpretation": "Observed usernames change across snapshots for identical image IDs. This can reflect renaming or source/attribution changes; it is not evidence of account splitting or common ownership. No users are linked into economic-person identities.",
    }
    path = args.root / "audit/overlap_account_label_audit.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

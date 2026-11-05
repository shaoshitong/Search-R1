"""Additional coverage audit only; does not refit or select any V5 model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
frame = pd.read_csv(args.source, usecols=["video_id", "date", "author_id", "upload_dt"])
frame["day"] = pd.to_datetime(frame.date.astype(str), format="%Y%m%d")
meta = frame.drop_duplicates("video_id").set_index("video_id")[["author_id", "upload_dt"]]
meta["upload"] = pd.to_datetime(meta.upload_dt)
start, end = frame.day.min(), frame.day.max()
meta = meta.loc[meta.upload.between(start, end)].copy()
grouped = frame.loc[frame.video_id.isin(meta.index)].groupby("video_id").day.agg(["min", "max", "count"])
meta = meta.join(grouped)
expected = (end-meta.upload).dt.days+1
interior = (meta["max"]-meta["min"]).dt.days+1-meta["count"]
missing = expected-meta["count"]
coverage = meta["count"]/expected
result = {
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "new_catalogue_items": len(meta),
    "calendar_item_days_from_upload_to_source_end": int(expected.sum()),
    "observed_new_item_days": int(meta["count"].sum()),
    "unobserved_post_upload_item_days": int(missing.sum()),
    "items_with_complete_post_upload_observation": int(missing.eq(0).sum()),
    "items_with_some_unobserved_post_upload_days": int(missing.gt(0).sum()),
    "items_last_seen_before_source_end": int(meta["max"].lt(end).sum()),
    "items_with_interior_date_gaps": int(interior.gt(0).sum()),
    "interior_unobserved_item_days": int(interior.sum()),
    "post_upload_coverage_ratio_quantiles": {str(q): float(v) for q,v in coverage.quantile([0,.25,.5,.75,.9,.99,1]).items()},
    "interpretation": "These are missing source observations, not zero exposures or verified withdrawal. The extra audit changes no V5 split, window, score, calibration grid or fitted result."
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result))

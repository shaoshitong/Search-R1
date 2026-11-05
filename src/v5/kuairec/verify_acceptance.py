"""Small completion audit; no refitting and no missing-day imputation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("research_root", type=Path)
args = parser.parse_args()
root = args.research_root / "v5/kuairec"
source = args.research_root / "v3/kuairec/item_daily_features.csv"
primary = json.loads((root / "results/cross_work_transfer.json").read_text())
supplement = json.loads((root / "new_upload_results/new_upload_first_day.json").read_text())
primary_exit = json.loads((root / "audit/run_01_exit.json").read_text())
supplement_exit = json.loads((root / "audit/new_upload_run_02_exit.json").read_text())
assert primary_exit["actual_process_exit_code"] == supplement_exit["actual_process_exit_code"] == 0
assert digest(source) == primary["source_sha256"] == supplement["source_sha256"]
assert digest(root / "code/protocol.json") == primary["protocol_sha256"] == primary_exit["protocol_sha256"]
assert digest(root / "code/new_upload_protocol.json") == supplement["protocol_sha256"] == supplement_exit["protocol_sha256"]
assert digest(root / "code/analyze_transfer.py") == primary_exit["source_code_sha256"] == supplement_exit["helper_source_sha256"]
assert digest(root / "code/analyze_new_uploads.py") == supplement_exit["source_code_sha256"]
assert len(primary["origins"]) == len(supplement["origins"]) == 4
assert len(set(d for o in supplement["origins"] for d in o["test_dates"])) == 28
frame = pd.read_csv(source, usecols=["video_id", "author_id", "date", "upload_dt", "show_cnt"])
first = frame.loc[frame.date.astype(str).eq(frame.upload_dt.str.replace("-", ""))]
counts = []
for origin in supplement["origins"]:
    train = frame.loc[frame.date.isin(origin["training_dates"])]
    test = first.loc[first.date.isin(origin["test_dates"])].copy()
    test["seen"] = test.author_id.isin(set(train.author_id))
    assert len(test) == origin["test_first_day_works"]
    counts.append({"origin": origin["origin"], "test_works": len(test),
                   "unique_accounts": int(test.author_id.nunique()),
                   "observed_zero_exposure_works": int(test.show_cnt.eq(0).sum()),
                   "daily_candidate_count_range": [int(test.groupby("date").size().min()), int(test.groupby("date").size().max())],
                   "seen_accounts": int(test.loc[test.seen, "author_id"].nunique()),
                   "unseen_accounts": int(test.loc[~test.seen, "author_id"].nunique())})
manifest = {}
for subdir in ["code", "new_upload_results", "audit"]:
    for path in sorted((root / subdir).glob("*")):
        if path.is_file() and path.name != "acceptance.json":
            manifest[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": digest(path)}
report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "accepted": True,
          "primary_real_exit": primary_exit["actual_process_exit_code"],
          "supplement_real_exit": supplement_exit["actual_process_exit_code"],
          "source_protocol_and_code_hashes_match_execution": True,
          "both_have_four_completed_origins": True, "supplement_unique_test_dates": 28,
          "primary_result_sha256": digest(root / "results/cross_work_transfer.json"),
          "supplement_result_sha256": digest(root / "new_upload_results/new_upload_first_day.json"),
          "supplement_first_day_counts": counts, "remote_artifacts": manifest,
          "row_level_predictions_remain_remote": True,
          "interpretation": "Execution and provenance acceptance only; no claim of causal or game-equilibrium identification."}
(root / "audit/acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "remote_artifacts"}, indent=2))

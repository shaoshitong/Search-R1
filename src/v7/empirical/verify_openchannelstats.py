"""Independently reconstruct the saved small impressions pilot."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


p=argparse.ArgumentParser();p.add_argument("research_root",type=Path);args=p.parse_args()
root=args.research_root/"v7/openchannelstats_empirical";r=root/"results";a=root/"audit"
summary=json.loads((r/"summary.json").read_text());receipt=json.loads((a/"run_01_exit.json").read_text())
cfg=json.loads((root/"code/openchannelstats_protocol.json").read_text())
assert receipt["actual_process_exit_code"]==0
assert sha(root/"code/openchannelstats_protocol.json")==receipt["protocol_sha256"]==summary["protocol_sha256"]
assert sha(root/"code/analyze_openchannelstats.py")==receipt["code_sha256"]==summary["source_code_sha256"]
assert datetime.fromisoformat(cfg["frozen_utc"])<datetime.fromisoformat(receipt["started_utc"])
assert all(sha(Path(f))==info["sha256"] for f,info in summary["input_files"].items())
pred=pd.read_parquet(r/"test_predictions.parquet");counts=pred.groupby(["model","channel","date"]).video_id.transform("size")
valid=pred.loc[counts.ge(2)].copy()
sums=pred.groupby(["model","channel","date"]).probability.sum();assert np.allclose(sums,1)
reconstructed={}
for model,g in valid.groupby("model"):
    value=float(-(g.impressions*np.log(g.probability)).sum()/g.impressions.sum())
    expected=next(x["test_log_loss"] for x in summary["pooled_within_channel_scores"] if x["model"]==model)
    assert abs(value-expected)<1e-12;reconstructed[model]=value
rows=[]
for channel,g in valid.groupby("channel"):
    exposure=float(g.loc[g.model.eq("age_only"),"impressions"].sum())
    old=next(x for x in summary["channels"] if x["channel"]==channel)
    m={x["model"]:x["test_log_loss"] for x in old["models"]}
    rows.append({"channel":channel,"exposure_weight":exposure,"delta_memory_minus_age":m["age_times_memory"]-m["age_only"]})
total=sum(x["exposure_weight"] for x in rows)
rows=sorted(rows,key=lambda x:x["exposure_weight"],reverse=True)
leave=[]
for excluded in rows:
    kept=[x for x in rows if x["channel"]!=excluded["channel"]]
    leave.append({"excluded_channel":excluded["channel"],"fixed_prediction_delta":sum(x["exposure_weight"]*x["delta_memory_minus_age"] for x in kept)/sum(x["exposure_weight"] for x in kept)})
report={"accepted":True,"checked_utc":datetime.now(timezone.utc).isoformat(),"actual_process_exit_code":0,
        "protocol_before_fitting":True,"source_and_execution_hashes_match":True,"source_csv_files_verified":len(summary["input_files"]),
        "probability_normalization_max_error":float(np.abs(sums-1).max()),"pooled_losses_reconstructed":reconstructed,
        "test_video_days_per_model":int(len(pred)/pred.model.nunique()),"test_impressions_weight":total,
        "channels_by_weight":rows,"largest_channel_weight_share":rows[0]["exposure_weight"]/total,
        "leave_one_channel_scoring_only":leave,
        "results_manifest":{p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in r.iterdir() if p.is_file()},
        "interpretation":"Fixed prediction reconstruction and leave-one-channel scoring only; no refit or causal effect."}
(a/"acceptance.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))

"""Reconstruct saved predictions and check execution; does not refit models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


parser=argparse.ArgumentParser();parser.add_argument("research_root",type=Path);args=parser.parse_args()
root=args.research_root/"v7/empirical";results=root/"results";audit=root/"audit"
summary=json.loads((results/"summary.json").read_text());receipt=json.loads((audit/"run_01_exit.json").read_text());cfg=summary["protocol"]
assert receipt["actual_process_exit_code"]==0
assert sha(args.research_root/"v3/kuairec/item_daily_features.csv")==summary["source_sha256"]
assert sha(root/"code/protocol.json")==summary["protocol_sha256"]==receipt["protocol_sha256"]
assert sha(root/"code/analyze_states.py")==summary["source_code_sha256"]==receipt["code_sha256"]
assert datetime.fromisoformat(cfg["frozen_utc"])<datetime.fromisoformat(receipt["started_utc"])
pred=pd.read_parquet(results/"test_predictions.parquet")
assert not pred.duplicated(["model","video_id","date"]).any()
assert pred.groupby(["model","origin","source_side","date","author_id"]).shared.nunique().max()==1
assert np.allclose(pred.loc[pred.new,"memory"],1)
assert np.allclose(pred.loc[~pred.source_available.astype(bool),"shared"],1)
prob_sums=pred.groupby(["model","origin","source_side","date"]).probability.sum()
assert np.allclose(prob_sums,1,atol=1e-12)
checks=[]
for model,part in pred.groupby("model"):
    account_p=part.groupby(["origin","source_side","date","author_id"]).probability.transform("sum")
    volume=part.show_cnt.sum()
    video=float(-(part.show_cnt*np.log(part.probability)).sum()/volume)
    account=float(-(part.show_cnt*np.log(account_p)).sum()/volume)
    saved=next(x for x in summary["pooled_models"] if x["model"]==model and x["subset"]=="all")
    assert abs(video-saved["video_joint_log_loss"])<1e-10
    assert abs(account-saved["account_joint_log_loss"])<1e-10
    new=part.loc[part.new]
    mass=new.groupby(["origin","source_side","date"]).probability.transform("sum")
    new_loss=float(-(new.show_cnt*np.log(new.probability/mass)).sum()/new.show_cnt.sum())
    saved_new=next(x for x in summary["pooled_models"] if x["model"]==model and x["subset"]=="new")
    assert abs(new_loss-saved_new["video_conditional_log_loss"])<1e-10
    checks.append({"model":model,"video_log_loss":video,"account_log_loss":account,"new_conditional_log_loss":new_loss})
one=pred.loc[pred.model.eq("work_only")]
assert len(one)==190568 and int(one.new.sum())==4315 and one.date.nunique()==28
daily=pd.read_csv(results/"daily_metrics.csv")
rng=np.random.default_rng(cfg["bootstrap_seed"]);minimum_daily_draw_weight=float("inf")
for model,reference in [("dual_state","work_only"),("work_only","no_state"),("account_only","no_state"),("dual_state","account_only")]:
    for subset in ["all","new","new_seen","new_unseen","old","source_available","source_unavailable"]:
        g=daily.loc[daily.model.eq(model)&daily.subset.eq(subset)].groupby(["origin","date"])[["exposures","conditional_exposures"]].sum()
        for weight in ["exposures","conditional_exposures"]:
            volume=g[weight].to_numpy()
            if volume.sum()==0:continue
            draws=rng.integers(0,len(volume),size=(cfg["bootstrap_repetitions"],len(volume)))
            den=volume[draws].sum(axis=1)
            assert (den>0).all();minimum_daily_draw_weight=min(minimum_daily_draw_weight,float(den.min()))
            origin_volume=g[weight].groupby(level=0).sum().to_numpy()
            draws_o=rng.integers(0,len(origin_volume),size=(cfg["bootstrap_repetitions"],len(origin_volume)))
            assert (origin_volume[draws_o].sum(axis=1)>0).all()
contributions=pd.read_parquet(results/"account_contributions.parquet").groupby("ai")[["delta_nll","exposure"]].sum()
draws=rng.integers(0,len(contributions),size=(cfg["bootstrap_repetitions"],len(contributions)))
den=contributions.exposure.to_numpy()[draws].sum(axis=1)
assert (den>0).all()
state={"accepted":True,"checked_utc":datetime.now(timezone.utc).isoformat(),"actual_process_exit_code":0,
       "source_protocol_code_hashes_match":True,"protocol_frozen_before_fitting":True,
       "rows_per_model":len(one),"new_works_per_model":int(one.new.sum()),"unique_test_dates":28,
       "public_source_state_is_common_within_target_account_day":True,"new_work_memory_equals_prior":True,
       "probabilities_sum_to_one_max_error":float(np.abs(prob_sums-1).max()),"reconstructed_metrics":checks,
       "bootstrap_weights":{"daily_and_origin_all_valid":True,"minimum_daily_draw_weight":minimum_daily_draw_weight,
                            "account_valid_draws":int((den>0).sum()),"account_total_draws":len(den),"minimum_account_draw_weight":float(den.min()),
                            "zero_weight_accounts_retained":int(contributions.exposure.eq(0).sum())},
       "results_manifest":{p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in results.iterdir() if p.is_file()},
       "scope":"Reconstruction and provenance verification, no model refitting or new statistical claim."}
(audit/"acceptance.json").write_text(json.dumps(state,indent=2)+"\n")
print(json.dumps({k:v for k,v in state.items() if k!="results_manifest"},indent=2))

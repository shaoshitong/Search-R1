"""Post-result fixed-prediction decomposition, no model refit or protocol change."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser=argparse.ArgumentParser();parser.add_argument("research_root",type=Path);args=parser.parse_args()
root=args.research_root/"v7/empirical";path=root/"results/test_predictions.parquet"
all_rows=pd.read_parquet(path)
rows=all_rows.loc[all_rows.new & all_rows.model.isin(["no_state","account_only"])].copy()
pieces=[];contributions=[]
for (model,origin,side,date),day in rows.groupby(["model","origin","source_side","date"],sort=True):
    total=float(day.show_cnt.sum()); day=day.copy();day["pnew"]=day.probability/day.probability.sum()
    for available in [True,False]:
        group=day.loc[day.source_available.astype(bool).eq(available)]
        exposure=float(group.show_cnt.sum());mass=float(group.pnew.sum())
        within=float(-(group.show_cnt*np.log(np.maximum(group.pnew/mass,1e-300))).sum()) if len(group) else 0.
        between=float(-exposure*np.log(max(mass,1e-300)))
        joint=float(-(group.show_cnt*np.log(np.maximum(group.pnew,1e-300))).sum())
        assert abs(joint-within-between)<1e-5
        pieces.append({"model":model,"origin":origin,"source_side":int(side),"date":int(date),"group":"available" if available else "unavailable",
                       "works":len(group),"exposures":exposure,"new_pool_exposures":total,"within_nll":within,"between_nll":between,
                       "joint_in_new_pool_nll":joint,"predicted_mass_weighted":mass*total})
    for _,r in day.iterrows():
        contributions.append({"model":model,"author_id":int(r.author_id),"exposure":float(r.show_cnt),"nll":float(-r.show_cnt*np.log(r.pnew))})
df=pd.DataFrame(pieces);result=[]
for (model,group),g in df.groupby(["model","group"],sort=True):
    volume=g.exposures.sum();total=g.new_pool_exposures.sum()
    result.append({"model":model,"group":group,"works":int(g.works.sum()),"exposures":float(volume),
                   "actual_new_exposure_mass":float(volume/total),"predicted_new_exposure_mass":float(g.predicted_mass_weighted.sum()/total),
                   "joint_in_new_pool_log_loss":float(g.joint_in_new_pool_nll.sum()/volume),
                   "within_group_conditional_log_loss":float(g.within_nll.sum()/volume),
                   "singleton_group_day_cells":int(g.works.eq(1).sum()),"empty_group_day_cells":int(g.works.eq(0).sum())})
totals=df.groupby("model")[["exposures","within_nll","between_nll","joint_in_new_pool_nll"]].sum()
volume=float(totals.loc["no_state","exposures"])
decomposition={k:float((totals.loc["account_only",k]-totals.loc["no_state",k])/volume) for k in ["within_nll","between_nll","joint_in_new_pool_nll"]}
assert abs(decomposition["joint_in_new_pool_nll"]-decomposition["within_nll"]-decomposition["between_nll"])<1e-12
rng=np.random.default_rng(20261108);intervals=[]
for label in ["available","unavailable","all"]:
    for metric in ["within_nll","joint_in_new_pool_nll","between_nll"]:
        l=df.loc[df.model.eq("account_only") & (df.group.eq(label) if label!="all" else True)].groupby("date")[[metric,"exposures"]].sum()
        r=df.loc[df.model.eq("no_state") & (df.group.eq(label) if label!="all" else True)].groupby("date")[[metric,"exposures"]].sum()
        delta=(l[metric]-r[metric]).to_numpy();weights=l.exposures.to_numpy()
        draws=rng.integers(0,len(l),size=(1000,len(l)));den=weights[draws].sum(axis=1);valid=den>0
        boot=delta[draws].sum(axis=1)[valid]/den[valid]
        intervals.append({"group":label,"metric":metric,"delta":float(delta.sum()/weights.sum()),
                          "paired_whole_date_bootstrap_95":np.quantile(boot,[.025,.975]).tolist(),"valid_draws":int(valid.sum()),"draws":1000})
c=pd.DataFrame(contributions).groupby(["model","author_id"])[["exposure","nll"]].sum()
l=c.loc["account_only"];r=c.loc["no_state"];account=pd.DataFrame({"delta":l.nll-r.nll,"exposure":l.exposure}).sort_values("exposure",ascending=False)
influence={"accounts":len(account),"improved":int(account.delta.lt(0).sum()),"worse":int(account.delta.gt(0).sum()),
           "top10_exposure_fraction":float(account.exposure.head(10).sum()/account.exposure.sum()),
           "drop_top_scoring_only_delta":{str(k):float(account.iloc[k:].delta.sum()/account.iloc[k:].exposure.sum()) for k in [1,5,10]}}
report={"completed_utc":datetime.now(timezone.utc).isoformat(),"post_hoc_fixed_prediction_diagnostic":True,"model_refit":False,
        "source_prediction_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"groups":result,
        "account_only_minus_no_state_new_pool_decomposition":decomposition,"date_intervals":intervals,"influence":influence,
        "scope":"Joint here means conditional on the new-work pool, then split by source-history availability; differs from main all-candidate new joint contribution. Group mass weighted by actual new-pool volume, not a future input.",
        "limitations":["Source availability is selected past catalogue coverage, not registration or owner capital.","Fixed prediction scores, no causal decomposition or refitting uncertainty.","Two hash directions reuse the same source, dates paired before bootstrap."]}
(root/"audit/new_work_group_diagnostic.json").write_text(json.dumps(report,indent=2)+"\n")
df.to_csv(root/"results/new_work_group_diagnostic_daily.csv",index=False)
print(json.dumps(report,indent=2))

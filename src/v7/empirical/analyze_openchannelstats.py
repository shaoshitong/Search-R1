"""Small independent opt-in YouTube impressions pilot; no channel-capital fit."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def probs(part,score):
    codes,_=pd.factorize(part.date,sort=True)
    return score/np.bincount(codes,weights=score)[codes]


parser=argparse.ArgumentParser();parser.add_argument("--research-root",type=Path,required=True);parser.add_argument("--protocol",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args();cfg=json.loads(args.protocol.read_text());args.output.mkdir(parents=True,exist_ok=True)
poolpath=args.research_root/cfg["source_pool_audit_relative"]
assert sha(poolpath)==cfg["source_pool_audit_sha256"]
pool=json.loads(poolpath.read_text());reports=[];daily_records=[];predictions=[];input_manifest={}
def progress(stage,**kw):
    row={"stage":stage,"utc":datetime.now(timezone.utc).isoformat(),**kw}
    with (args.output/"progress.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)
progress("source_manifest_verified",channels=len(pool["channels"]))
for channel in pool["channels"]:
    handle=channel["public_handle"];base=Path(channel["root"])/"raw/openchannelstats"/handle
    for name in ["videos.csv","video_daily.csv","channel_daily.csv"]:
        path=base/name;actual=sha(path);expected=(base/(name+".sha256")).read_text().split()[0]
        assert actual==expected
        input_manifest[str(path)]={"bytes":path.stat().st_size,"sha256":actual}
    meta=pd.read_csv(base/"videos.csv",comment="#",usecols=["video_id","published_at"])
    raw=pd.read_csv(base/"video_daily.csv",comment="#")
    report={"channel":handle,"raw_video_days":len(raw),"metadata_videos":len(meta)}
    if len(raw)==0:
        report.update(status="empty_public_entry",eligible=False);reports.append(report);progress("channel_coverage_only",channel=handle,status=report["status"]);continue
    raw["day"]=pd.to_datetime(raw.date);raw["cutoff"]=pd.to_datetime(raw.recorded_through)
    meta["publish_day"]=pd.to_datetime(meta.published_at)
    merged=raw.merge(meta[["video_id","publish_day"]],on="video_id",how="left",validate="many_to_one")
    keep=merged.impressions.notna() & merged.impressions.ge(0) & merged.days_since_publish.ge(0) & merged.day.ge(merged.publish_day) & merged.day.le(merged.cutoff)
    d=merged.loc[keep].sort_values(["day","video_id"]).reset_index(drop=True)
    assert not d.duplicated(["video_id","date"]).any()
    report.update(missing_impression_rows=int(merged.impressions.isna().sum()),negative_impression_rows=int(merged.impressions.lt(0).sum()),prepublish_rows=int((merged.days_since_publish.lt(0)|merged.day.lt(merged.publish_day)).sum()),after_recorded_through_rows=int(merged.day.gt(merged.cutoff).sum()),valid_video_days=len(d),recorded_through_values=sorted(raw.recorded_through.unique().tolist()))
    if not len(d):report.update(status="no_valid_impression_rows",eligible=False);reports.append(report);continue
    end=d.day.max();test_start=end-pd.Timedelta(days=cfg["test_calendar_days"]-1);cal_start=test_start-pd.Timedelta(days=cfg["calibration_calendar_days"]);train_start=cal_start-pd.Timedelta(days=cfg["train_calendar_days"])
    d["age"]=(d.day-d.publish_day).dt.days;d["age_group"]=np.digitize(d.age,cfg["age_bins_days"])
    train=d.loc[d.day.ge(train_start)&d.day.lt(cal_start)];cal=d.loc[d.day.ge(cal_start)&d.day.lt(test_start)];test=d.loc[d.day.ge(test_start)]
    cal_multi=int(cal.groupby("date").size().ge(2).sum());test_multi=int(test.groupby("date").size().ge(2).sum())
    report.update(valid_date_range=[str(d.day.min().date()),str(end.date())],training_window=[str(train_start.date()),str((cal_start-pd.Timedelta(days=1)).date())],calibration_window=[str(cal_start.date()),str((test_start-pd.Timedelta(days=1)).date())],test_window=[str(test_start.date()),str(end.date())],training_observed_dates=int(train.day.nunique()),calibration_multivideo_dates=cal_multi,test_multivideo_dates=test_multi,test_rows=len(test),test_videos=int(test.video_id.nunique()))
    eligible=train.day.nunique()>=cfg["minimum_training_observed_dates"] and cal_multi>=cfg["minimum_calibration_multivideo_dates"] and test_multi>=cfg["minimum_test_multivideo_dates"] and train.impressions.sum()>0 and cal.impressions.sum()>0 and test.impressions.sum()>0
    if not eligible:
        report.update(status="insufficient_prespecified_window",eligible=False);reports.append(report);progress("channel_coverage_only",channel=handle,status=report["status"],test_rows=len(test));continue
    mu=float(train.impressions.mean());g=train.groupby("age_group").impressions.agg(["sum","size"])
    means=(g["sum"]+cfg["age_prior_video_days"]*mu)/(g["size"]+cfg["age_prior_video_days"])
    d["base"]=d.age_group.map(means).fillna(mu)
    vi=pd.factorize(d.video_id,sort=True)[0];nv=int(vi.max()+1);actual=np.zeros(nv);expected=np.zeros(nv);memory=np.empty(len(d));previous=None;rho=2**(-1/cfg["memory_half_life_days"]);prior=cfg["memory_prior_video_days"]*mu
    for day,positions in d.groupby("day",sort=True).indices.items():
        if previous is not None:
            decay=rho**((day-previous).days);actual*=decay;expected*=decay
        ids=vi[positions];memory[positions]=(actual[ids]+prior)/(expected[ids]+prior)
        np.add.at(actual,ids,d.iloc[positions].impressions.to_numpy(float));np.add.at(expected,ids,d.iloc[positions].base.to_numpy(float));previous=day
    d["memory"]=memory
    d=d.sort_values(["video_id","day"])
    d["previous_day"]=d.groupby("video_id").day.shift();d["previous_impressions"]=d.groupby("video_id").impressions.shift()
    d["lag_score"]=np.where((d.day-d.previous_day).dt.days.eq(1),d.previous_impressions+1,d.base)
    d=d.sort_values(["day","video_id"]).reset_index(drop=True)
    cal=d.loc[d.day.ge(cal_start)&d.day.lt(test_start)];test=d.loc[d.day.ge(test_start)].copy()
    cm=cal.groupby("date").video_id.transform("size").ge(2).to_numpy();cy=cal.impressions.to_numpy(float)
    losses=[]
    for gamma in cfg["gamma_grid"]:
        pp=probs(cal,cal.base.to_numpy()*cal.memory.to_numpy()**gamma)
        losses.append(float(-(cy[cm]*np.log(np.maximum(pp[cm],1e-300))).sum()/cy[cm].sum()))
    gamma=cfg["gamma_grid"][int(np.argmin(losses))]
    scores={"uniform":np.ones(len(test)),"age_only":test.base.to_numpy(),"age_times_memory":test.base.to_numpy()*test.memory.to_numpy()**gamma,"previous_day_with_prior":test.lag_score.to_numpy()}
    metrics=[]
    for model,score in scores.items():
        pred=probs(test,score);temp=test[["video_id","date","impressions","age"]].copy();temp["probability"]=pred;temp["model"]=model;temp["channel"]=handle
        predictions.append(temp)
        for date,positions in temp.groupby("date",sort=True).indices.items():
            part=temp.iloc[positions];volume=float(part.impressions.sum());count=len(part);nll=float(-(part.impressions*np.log(np.maximum(part.probability,1e-300))).sum())
            daily_records.append({"channel":handle,"model":model,"date":date,"rows":count,"volume":volume if count>=2 else 0.,"nll":nll if count>=2 else 0.})
        own=pd.DataFrame([x for x in daily_records if x["channel"]==handle and x["model"]==model]);volume=own.volume.sum()
        metrics.append({"model":model,"test_log_loss":float(own.nll.sum()/volume),"test_impressions_weight":float(volume)})
    channel_daily=pd.read_csv(base/"channel_daily.csv",comment="#");channel_daily["date"]=pd.to_datetime(channel_daily.date).dt.strftime("%Y-%m-%d")
    old=test.loc[test.age.gt(0)].groupby("date").impressions.agg(["sum","size"]).rename(columns={"sum":"old_impressions","size":"old_rows"}).reset_index()
    old=old.merge(channel_daily[["date","videos_published_that_day"]],on="date",how="left",validate="one_to_one")
    associations=[]
    for label,mask in [("publication_day",old.videos_published_that_day.gt(0)),("no_publication_day",old.videos_published_that_day.eq(0)),("unknown_publication",old.videos_published_that_day.isna())]:
        part=old.loc[mask];n=int(part.old_rows.sum())
        associations.append({"day_type":label,"observed_dates":len(part),"old_video_days":n,"old_impressions":float(part.old_impressions.sum()),"impressions_per_observed_old_video_day":float(part.old_impressions.sum()/n) if n else None})
    report.update(status="analyzed",eligible=True,selected_gamma=gamma,calibration_memory_losses=losses,models=metrics,old_publication_association=associations)
    reports.append(report);progress("channel_analysis_complete",channel=handle,test_rows=len(test),selected_gamma=gamma,models=metrics)
daily=pd.DataFrame(daily_records);daily.to_csv(args.output/"daily_metrics.csv",index=False)
pd.concat(predictions,ignore_index=True).to_parquet(args.output/"test_predictions.parquet",index=False)
pooled=[{"model":m,"test_log_loss":float(g.nll.sum()/g.volume.sum()),"impressions_weight":float(g.volume.sum())} for m,g in daily.groupby("model")]
l=daily.loc[daily.model.eq("age_times_memory")].groupby("date")[["nll","volume"]].sum();r=daily.loc[daily.model.eq("age_only")].groupby("date")[["nll","volume"]].sum()
delta=(l.nll-r.nll).to_numpy();weights=l.volume.to_numpy();rng=np.random.default_rng(cfg["bootstrap_seed"]);draws=rng.integers(0,len(l),size=(cfg["bootstrap_repetitions"],len(l)));den=weights[draws].sum(axis=1);valid=den>0;boot=delta[draws].sum(axis=1)[valid]/den[valid]
summary={"completed_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":sha(args.protocol),"source_pool_audit_sha256":sha(poolpath),"source_code_sha256":sha(Path(__file__)),"channels":reports,"eligible_channels":sum(x["eligible"] for x in reports),"pooled_within_channel_scores":pooled,"age_memory_minus_age":{"delta":float(delta.sum()/weights.sum()),"paired_date_bootstrap_95":np.quantile(boot,[.025,.975]).tolist(),"valid_draws":int(valid.sum()),"draws":len(draws)},"input_files":input_manifest,"limitations":["Public opt-in tiny channel sample, not random or complete platform market.","No estimation of common channel H beta: it cancels in within-channel daily shares.","Lifetime/peak/day1 metadata not used as predictive input.","Missing impressions remain missing; conditional observed pools and reporting cutoff respected.","Publication-day associations are descriptive, not causal cannibalization or AI-cost responses."]}
(args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
progress("complete",eligible_channels=summary["eligible_channels"],pooled=pooled,comparison=summary["age_memory_minus_age"])

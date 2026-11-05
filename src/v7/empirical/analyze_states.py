"""V7 observed-memory prediction; metadata only, CPU, no causal state claim."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def construct_features(frame, training_dates, config):
    training = frame.loc[frame.date.isin(training_dates)]
    mu = float(training.show_cnt.mean())
    group = training.groupby("base_group").show_cnt.agg(["sum", "size"])
    means = (group["sum"] + config["baseline_prior_item_days"]*mu)/(group["size"]+config["baseline_prior_item_days"])
    expected = frame.base_group.map(means).fillna(mu).to_numpy(float)
    nv, na = frame.vi.max()+1, frame.ai.max()+1
    item_meta = frame.drop_duplicates("vi").sort_values("vi")
    va = item_meta.ai.to_numpy(int)
    upload = item_meta.upload_day.to_numpy(dtype="datetime64[D]")
    item_y, item_e = np.zeros(nv), np.zeros(nv)
    source_y, source_e = np.zeros((2,na)), np.zeros((2,na))
    known, q = np.zeros(nv,dtype=bool), np.zeros(na)
    fields = {k:np.empty(len(frame)) for k in ["memory", "shared", "all_history", "q", "f", "mean_memory", "own_available", "source_available"]}
    prior = config["prior_item_days"]*mu
    rho = 2**(-1/config["memory_half_life_days"])
    previous = None
    for date, positions in frame.groupby("date",sort=True).indices.items():
        part = frame.iloc[positions]
        day = part.day.iloc[0].to_datetime64().astype("datetime64[D]")
        if previous is not None:
            decay = rho**int((day-previous)/np.timedelta64(1,"D"))
            item_y *= decay; item_e *= decay
            source_y *= decay; source_e *= decay
        vi, ai = part.vi.to_numpy(int), part.ai.to_numpy(int)
        side = part.side.to_numpy(int)
        memories = (item_y+prior)/(item_e+prior)
        source_h = (source_y+prior)/(source_e+prior)
        all_h = (source_y.sum(axis=0)+prior)/(source_e.sum(axis=0)+prior)
        mmean = np.bincount(va[known],weights=memories[known],minlength=na)/np.maximum(q,1)
        mmean[q==0] = 1
        recent = known & (upload >= day-np.timedelta64(14,"D")) & (upload<day)
        f = np.bincount(va[recent],minlength=na)
        fields["memory"][positions] = memories[vi]
        fields["shared"][positions] = source_h[1-side,ai]
        fields["all_history"][positions] = all_h[ai]
        fields["q"][positions], fields["f"][positions] = q[ai], f[ai]
        fields["mean_memory"][positions] = mmean[ai]
        fields["own_available"][positions] = item_e[vi]>0
        fields["source_available"][positions] = source_e[1-side,ai]>0
        newly_seen = np.unique(vi[~known[vi]])
        q += np.bincount(va[newly_seen],minlength=na)
        known[newly_seen] = True
        np.add.at(item_y,vi,part.show_cnt.to_numpy(float))
        np.add.at(item_e,vi,expected[positions])
        for source_side in [0,1]:
            mask = side==source_side
            source_y[source_side] += np.bincount(ai[mask],weights=part.show_cnt.to_numpy(float)[mask],minlength=na)
            source_e[source_side] += np.bincount(ai[mask],weights=expected[positions][mask],minlength=na)
        previous = day
    result = frame.copy()
    for key,value in fields.items(): result[key] = value
    result["base"] = expected/mu
    result["new"] = result.day.eq(result.upload_day)
    result["train_seen"] = result.ai.isin(set(training.ai))
    assert np.allclose(result.loc[result.new,"memory"],1)
    assert result.groupby(["date","side","ai"]).shared.nunique().max()==1
    return result, {"training_global_mean_exposure":mu,"prior_exposure_equivalent":prior,
                    "duration_age_groups_observed":len(means),"rho_observed":rho}


def probability(part, setting):
    alpha, gamma, beta = setting
    z = part.base.to_numpy()*part.memory.to_numpy()**gamma
    group, _ = pd.factorize(pd.MultiIndex.from_frame(part[["date","ai"]]),sort=True)
    day, _ = pd.factorize(part.date,sort=True)
    total = np.bincount(group,weights=z)
    score = part.shared.to_numpy()**beta * total[group]**(alpha-1)*z
    p = score/np.bincount(day,weights=score)[day]
    group_score = np.bincount(group,weights=score)
    first = np.unique(group,return_index=True)[1]
    expected = part.shared.to_numpy()[first]**beta * total**alpha
    error = float(np.max(np.abs(group_score-expected)/np.maximum(expected,1e-12)))
    return p, group, error


def records(part, p, group):
    work = part.copy()
    work["p"] = p
    account_p = np.bincount(group,weights=p)[group]
    work["account_nll"] = -work.show_cnt.to_numpy()*np.log(np.maximum(account_p,1e-300))
    output = []
    for date, day in work.groupby("date",sort=True):
        masks = {"all":np.ones(len(day),dtype=bool), "old":~day.new.to_numpy(),
                 "new":day.new.to_numpy(), "new_seen":(day.new & day.train_seen).to_numpy(),
                 "new_unseen":(day.new & ~day.train_seen).to_numpy(),
                 "source_available":day.source_available.to_numpy(bool),
                 "source_unavailable":~day.source_available.to_numpy(bool)}
        for label, mask in masks.items():
            d = day.loc[mask]
            count, volume, mass = len(d), float(d.show_cnt.sum()), float(d.p.sum())
            joint = float(-(d.show_cnt*np.log(np.maximum(d.p,1e-300))).sum())
            conditional = float(-(d.show_cnt*np.log(np.maximum(d.p/mass,1e-300))).sum()) if count>=2 else 0.
            output.append({"date":int(date),"subset":label,"candidates":count,"exposures":volume,
                           "joint_nll":joint,"conditional_nll":conditional,
                           "conditional_exposures":volume if count>=2 else 0.,
                           "predicted_mass_times_day_exposures":mass*float(day.show_cnt.sum()),
                           "day_exposures":float(day.show_cnt.sum()),
                           "account_nll":float(d.account_nll.sum()),"observed_zero_rows":int(d.show_cnt.eq(0).sum())})
    return output


def summarize(frame):
    reports=[]
    for (model,subset),g in frame.groupby(["model","subset"],sort=False):
        volume=float(g.exposures.sum()); cv=float(g.conditional_exposures.sum())
        reports.append({"model":model,"subset":subset,"video_joint_log_loss":float(g.joint_nll.sum()/volume) if volume else None,
                        "video_conditional_log_loss":float(g.conditional_nll.sum()/cv) if cv else None,
                        "account_joint_log_loss":float(g.account_nll.sum()/volume) if volume else None,
                        "exposure_weight":volume,"conditional_exposure_weight":cv,"row_evaluations":int(g.candidates.sum()),
                        "observed_zero_rows":int(g.observed_zero_rows.sum()),
                        "singleton_or_empty_day_cells":int(g.candidates.lt(2).sum())})
    return reports


def ridge_forecasts(table, train_dates, cal_dates, test_dates, expanded_columns, config):
    train=table.loc[table.date.isin(train_dates)]
    cal=table.loc[table.date.isin(cal_dates)]
    test=table.loc[table.date.isin(test_dates)]
    result=[]
    def metrics(pred):
        delta=pred-test.log_y.to_numpy()
        return {"mse_log1p_exposure":float(np.mean(delta**2)),"mae_log1p_exposure":float(np.abs(delta).mean()),"test_pairs":len(test)}
    result.append({"model":"carry_forward",**metrics(test.prev_log_y.to_numpy())})
    result.append({"model":"training_mean",**metrics(np.full(len(test),train.log_y.mean()))})
    for model,columns in [("ar1",["prev_log_y"]),("ar1_plus_observed_states_supply",expanded_columns)]:
        mean=train[columns].mean().to_numpy(); scale=train[columns].std(ddof=0).replace(0,1).to_numpy()
        def design(data): return np.column_stack([np.ones(len(data)),(data[columns].to_numpy()-mean)/scale])
        xt,xc,xs=design(train),design(cal),design(test)
        fits=[]
        for penalty in config["ridge_penalty_grid"]:
            reg=np.eye(xt.shape[1])*penalty*len(train);reg[0,0]=0
            coef=np.linalg.solve(xt.T@xt+reg,xt.T@train.log_y.to_numpy())
            loss=float(np.mean((xc@coef-cal.log_y.to_numpy())**2))
            fits.append((loss,penalty,coef))
        loss,penalty,coef=min(fits,key=lambda x:x[0])
        pred=np.maximum(0,xs@coef)
        result.append({"model":model,"selected_penalty":penalty,"calibration_mse":loss,"columns":columns,**metrics(pred)})
    return {"training_pairs":len(train),"calibration_pairs":len(cal),"test_pairs":len(test),"models":result,
            "scope":"Next-observed-calendar-day exposure prediction, not verification of structural state transition."}


def transition_tables(features):
    item=features.sort_values(["vi","day"]).copy()
    for name in ["show_cnt","day"]: item["previous_"+name]=item.groupby("vi")[name].shift()
    item=item.loc[(item.day-item.previous_day).dt.days.eq(1)].copy()
    item["log_y"]=np.log1p(item.show_cnt);item["prev_log_y"]=np.log1p(item.previous_show_cnt)
    for old,new in [("memory","log_memory"),("shared","log_shared"),("base","log_base")]:item[new]=np.log(item[old])
    item["log_q"]=np.log1p(item.q);item["log_f"]=np.log1p(item.f)
    account=features.groupby(["date","day","ai"],sort=True).agg(show_cnt=("show_cnt","sum"),mean_memory=("mean_memory","first"),all_history=("all_history","first"),q=("q","first"),f=("f","first")).reset_index().sort_values(["ai","day"])
    account["previous_show_cnt"]=account.groupby("ai").show_cnt.shift()
    account["previous_day"]=account.groupby("ai").day.shift()
    account=account.loc[(account.day-account.previous_day).dt.days.eq(1)].copy()
    account["log_y"]=np.log1p(account.show_cnt);account["prev_log_y"]=np.log1p(account.previous_show_cnt)
    account["log_memory"]=np.log(account.mean_memory);account["log_shared"]=np.log(account.all_history)
    account["log_q"]=np.log1p(account.q);account["log_f"]=np.log1p(account.f)
    return item,account


def comparisons(daily, contributions, config):
    result=[];rng=np.random.default_rng(config["bootstrap_seed"])
    for model,reference in [("dual_state","work_only"),("work_only","no_state"),("account_only","no_state"),("dual_state","account_only")]:
        for subset in ["all","new","new_seen","new_unseen","old","source_available","source_unavailable"]:
            l=daily.loc[daily.model.eq(model)&daily.subset.eq(subset)].groupby(["origin","date"])[["joint_nll","conditional_nll","exposures","conditional_exposures"]].sum()
            r=daily.loc[daily.model.eq(reference)&daily.subset.eq(subset)].groupby(["origin","date"])[["joint_nll","conditional_nll","exposures","conditional_exposures"]].sum()
            for label,nll,weight in [("joint","joint_nll","exposures"),("conditional","conditional_nll","conditional_exposures")]:
                delta=(l[nll]-r[nll]).to_numpy();volume=l[weight].to_numpy()
                if volume.sum()==0:continue
                draws=rng.integers(0,len(delta),size=(config["bootstrap_repetitions"],len(delta)))
                denom=volume[draws].sum(axis=1);boot=delta[draws].sum(axis=1)/np.maximum(denom,1e-300)
                od=(l[nll]-r[nll]).groupby(level=0).sum().to_numpy();ov=l[weight].groupby(level=0).sum().to_numpy()
                draws_o=rng.integers(0,len(od),size=(config["bootstrap_repetitions"],len(od)))
                ob=od[draws_o].sum(axis=1)/np.maximum(ov[draws_o].sum(axis=1),1e-300)
                result.append({"model":model,"reference":reference,"subset":subset,"evaluation":label,
                               "delta":float(delta.sum()/volume.sum()),"paired_date_bootstrap_95":np.quantile(boot,[.025,.975]).tolist(),
                               "origin_block_bootstrap_95":np.quantile(ob,[.025,.975]).tolist()})
    c=contributions.groupby("ai")[["delta_nll","exposure"]].sum()
    d,v=c.delta_nll.to_numpy(),c.exposure.to_numpy()
    draws=rng.integers(0,len(c),size=(config["bootstrap_repetitions"],len(c)))
    boot=d[draws].sum(axis=1)/np.maximum(v[draws].sum(axis=1),1e-300)
    ranked=c.sort_values("exposure",ascending=False)
    influence={"accounts_observed":len(c),"accounts_with_positive_exposure":int(c.exposure.gt(0).sum()),
               "accounts_with_improvement":int(c.delta_nll.lt(0).sum()),"accounts_with_worsening":int(c.delta_nll.gt(0).sum()),
               "top10_exposure_fraction":float(ranked.exposure.head(10).sum()/c.exposure.sum()),
               "pooled_delta":float(c.delta_nll.sum()/c.exposure.sum()),
               "fixed_prediction_account_bootstrap_95":np.quantile(boot,[.025,.975]).tolist(),
               "drop_top_scoring_contribution_deltas":{str(k):float(ranked.iloc[k:].delta_nll.sum()/ranked.iloc[k:].exposure.sum()) for k in [1,5,10]},
               "interpretation":"Post-hoc fixed contributions, no refit or probability renormalization; not a market counterfactual."}
    return result,influence


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--protocol",type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.protocol.read_text());args.output.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat();clock=time.monotonic()
    def progress(stage,**kw):
        point={"stage":stage,"utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":round(time.monotonic()-clock,3),**kw}
        with (args.output/"progress.jsonl").open("a") as f:f.write(json.dumps(point)+"\n")
        print(json.dumps(point),flush=True)
    assert sha(args.source)==cfg["source_sha256"]
    frame=pd.read_csv(args.source,usecols=["video_id","author_id","date","upload_dt","video_duration","show_cnt"])
    frame=frame.sort_values(["date","video_id"]).reset_index(drop=True)
    assert not frame.duplicated(["video_id","date"]).any() and frame.show_cnt.notna().all()
    assert frame.groupby("video_id").author_id.nunique().max()==1
    frame["day"]=pd.to_datetime(frame.date.astype(str),format="%Y%m%d");frame["upload_day"]=pd.to_datetime(frame.upload_dt)
    frame["vi"]=pd.factorize(frame.video_id,sort=True)[0];frame["ai"]=pd.factorize(frame.author_id,sort=True)[0]
    split={v:int(hashlib.sha256((cfg["split_salt"]+str(int(v))).encode()).hexdigest()[:16],16)%2 for v in frame.video_id.unique()}
    frame["side"]=frame.video_id.map(split)
    age=(frame.day-frame.upload_day).dt.days
    dur=np.digitize(frame.video_duration.fillna(-1),cfg["duration_bins_ms"]);dur[frame.video_duration.isna()]=len(cfg["duration_bins_ms"])+1
    frame["base_group"]=np.digitize(age,cfg["age_bins_days"])*(len(cfg["duration_bins_ms"])+2)+dur
    dates=np.sort(frame.date.unique());assert len(dates)==63
    grids={"no_state":list(itertools.product(cfg["alpha_grid"],[0.],[0.])),
           "work_only":list(itertools.product(cfg["alpha_grid"],cfg["work_gamma_grid"],[0.])),
           "account_only":list(itertools.product(cfg["alpha_grid"],[0.],cfg["account_beta_grid"])),
           "dual_state":list(itertools.product(cfg["alpha_grid"],cfg["work_gamma_grid"],cfg["account_beta_grid"]))}
    profile={"rows":len(frame),"videos":int(frame.video_id.nunique()),"accounts":int(frame.author_id.nunique()),
             "date_range":[int(dates[0]),int(dates[-1])],"zero_observed_exposure_rows":int(frame.show_cnt.eq(0).sum()),
             "first_day_rows":int(frame.day.eq(frame.upload_day).sum()),"source_bytes":args.source.stat().st_size,
             "hash_sides_video_counts":{str(k):int(g.video_id.nunique()) for k,g in frame.groupby("side")}}
    progress("source_verified",**profile)
    all_daily=[];origin_reports=[];all_contrib=[];all_predictions=[];transitions=[];bridge_error=0.
    for number,start in enumerate(cfg["origin_start_indices"],1):
        D,C,T=cfg["training_days"],cfg["calibration_days"],cfg["test_days"]
        train,cal,test=dates[start:start+D],dates[start+D:start+D+C],dates[start+D+C:start+D+C+T]
        features,base_audit=construct_features(frame,train,cfg)
        origin="R"+str(number)
        progress("past_only_states_ready",origin=origin,cal_rows=int(features.date.isin(cal).sum()),test_rows=int(features.date.isin(test).sum()))
        for source_side in cfg["source_directions"]:
            target_side=1-source_side
            calibration=features.loc[features.date.isin(cal)&features.side.eq(target_side)].reset_index(drop=True)
            target=features.loc[features.date.isin(test)&features.side.eq(target_side)].reset_index(drop=True)
            report={"origin":origin,"source_side":source_side,"target_side":target_side,"training_dates":train.tolist(),"calibration_dates":cal.tolist(),"test_dates":test.tolist(),"baseline":base_audit,
                    "test_rows":len(target),"test_accounts":int(target.ai.nunique()),"test_exposure":float(target.show_cnt.sum()),
                    "source_history_row_fraction":float(target.source_available.mean()),
                    "source_history_exposure_fraction":float(target.loc[target.source_available.astype(bool),"show_cnt"].sum()/target.show_cnt.sum()),
                    "new_works":int(target.new.sum()),"new_train_seen_works":int((target.new&target.train_seen).sum()),
                    "q_quantiles":target.q.quantile([0,.5,.9,1]).to_dict(),"f_quantiles":target.f.quantile([0,.5,.9,1]).to_dict(),"models":[]}
            model_prediction={}
            for model,settings in grids.items():
                losses=[]
                for setting in settings:
                    p,_,error=probability(calibration,setting);bridge_error=max(bridge_error,error)
                    losses.append(float(-(calibration.show_cnt.to_numpy()*np.log(np.maximum(p,1e-300))).sum()/calibration.show_cnt.sum()))
                selected=int(np.argmin(losses));setting=settings[selected]
                p,group,error=probability(target,setting);bridge_error=max(bridge_error,error)
                model_prediction[model]=p
                daily=records(target,p,group)
                for item in daily:item.update(origin=origin,source_side=source_side,model=model)
                all_daily.extend(daily)
                report["models"].append({"model":model,"grid_size":len(settings),"selected_alpha_gamma_beta":list(setting),"calibration_video_log_loss":losses[selected],"test":summarize(pd.DataFrame(daily))[0:]})
                pred=target[["video_id","author_id","date","new","train_seen","source_available","memory","shared","q","f","show_cnt"]].copy()
                pred["origin"]=origin;pred["source_side"]=source_side;pred["model"]=model;pred["probability"]=p
                all_predictions.append(pred)
            contribution=target[["ai","date","show_cnt"]].copy()
            contribution["origin"]=origin;contribution["source_side"]=source_side
            contribution["delta_nll"]=-target.show_cnt.to_numpy()*np.log(np.maximum(model_prediction["dual_state"],1e-300))+target.show_cnt.to_numpy()*np.log(np.maximum(model_prediction["work_only"],1e-300))
            contribution=contribution.rename(columns={"show_cnt":"exposure"});all_contrib.append(contribution)
            origin_reports.append(report)
            progress("direction_complete",origin=origin,source_side=source_side,new_works=report["new_works"],selected={m["model"]:m["selected_alpha_gamma_beta"] for m in report["models"]})
        item_pairs,account_pairs=transition_tables(features)
        tr={"origin":origin}
        tr["item"]=ridge_forecasts(item_pairs,train,cal,test,["prev_log_y","log_memory","log_shared","log_q","log_f","log_base"],cfg)
        tr["account"]=ridge_forecasts(account_pairs,train,cal,test,["prev_log_y","log_memory","log_shared","log_q","log_f"],cfg)
        transitions.append(tr)
        (args.output/"origins_progress.json").write_text(json.dumps(origin_reports,indent=2)+"\n")
        progress("origin_complete",origin=origin,item_transition_pairs=tr["item"]["test_pairs"],account_transition_pairs=tr["account"]["test_pairs"])
    daily=pd.DataFrame(all_daily);contributions=pd.concat(all_contrib,ignore_index=True)
    daily.to_csv(args.output/"daily_metrics.csv",index=False)
    contributions.to_parquet(args.output/"account_contributions.parquet",index=False)
    pd.concat(all_predictions,ignore_index=True).to_parquet(args.output/"test_predictions.parquet",index=False)
    compare,influence=comparisons(daily,contributions,cfg)
    aggregate={"study_id":cfg["study_id"],"started_utc":started,"completed_utc":datetime.now(timezone.utc).isoformat(),
               "protocol_sha256":sha(args.protocol),"source_sha256":sha(args.source),"source_code_sha256":sha(Path(__file__)),
               "protocol":cfg,"source_profile":profile,"origins_and_directions":origin_reports,"pooled_models":summarize(daily),
               "comparisons":compare,"influence_dual_minus_work":influence,"observed_one_step_transition_predictions":transitions,
               "maximum_relative_account_score_bridge_error":bridge_error,
               "unique_test_dates":int(daily.date.nunique()),"source_and_target_work_sets_disjoint":True,
               "no_future_survival_filter":True,"missing_days_not_declared_true_zero":True,
               "interpretation":"Conditional observed-pool predictions, not causal capital, structural stock transitions or AI-cost identification."}
    assert bridge_error<1e-10
    (args.output/"summary.json").write_text(json.dumps(aggregate,indent=2)+"\n")
    manifest={p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in args.output.iterdir() if p.is_file() and p.name!="artifact_manifest.json"}
    (args.output/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    progress("complete",origins=4,directions=8,unique_test_dates=aggregate["unique_test_dates"],bridge_error=bridge_error,main_influence=influence)


if __name__=="__main__":main()

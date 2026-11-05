"""Supplement: first-day exposure prediction without target-work feedback."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from analyze_transfer import sha256


def duration_bucket(values, bins):
    buckets = pd.Series(np.digitize(values.fillna(-1).to_numpy(), bins), index=values.index).astype(str)
    return buckets.mask(values.isna(), "missing")


def probabilities(scores, dates):
    codes, days = pd.factorize(dates, sort=True)
    scores = np.maximum(np.asarray(scores, dtype=float), 1e-15)
    denominator = np.zeros((scores.shape[0], len(days)))
    for index in range(len(days)):
        denominator[:, index] = scores[:, codes == index].sum(axis=1)
    return scores / denominator[:, codes]


def candidate_scores(features, settings, model, mu):
    result = []
    for setting in settings:
        score = np.ones(len(features)) if model == "uniform_videos" else features.duration_base.to_numpy().copy()
        if "history_prior_item_days" in setting and setting["history_prior_item_days"] != "no_history":
            prior = setting["history_prior_item_days"]
            history = (features.old_exposures.to_numpy()+prior*mu)/(features.old_item_days.to_numpy()+prior)/mu
            score *= history
        score *= (1+features.Q.to_numpy(dtype=float)) ** setting.get("beta_Q", 0)
        score *= (1+features.F.to_numpy(dtype=float)) ** setting.get("beta_F", 0)
        result.append(score)
    return np.asarray(result)


def evaluate(rows, p):
    records = []
    for date, positions in rows.groupby("date", sort=True).indices.items():
        r = rows.iloc[positions]
        y = r.show_cnt.to_numpy(dtype=float)
        pred = p[positions]
        total = float(y.sum())
        valid = len(r) >= 2
        record = {"date": int(date), "candidates": len(r), "all_exposures": total,
                  "joint_exposures": total if valid else 0.,
                  "joint_nll": float(-(y*np.log(np.maximum(pred,1e-15))).sum()) if valid else 0.}
        for label, flag in [("seen", True), ("unseen", False)]:
            mask = r.seen_in_training.to_numpy() == flag
            count = int(mask.sum())
            exposure = float(y[mask].sum())
            mass = float(pred[mask].sum())
            record.update({label+"_candidates": count,
                           label+"_joint_exposures": exposure if valid else 0.,
                           label+"_joint_nll": float(-(y[mask]*np.log(np.maximum(pred[mask],1e-15))).sum()) if valid else 0.,
                           label+"_predicted_mass_times_day_volume": mass*total if valid else 0.,
                           label+"_conditional_exposures": exposure if count >= 2 else 0.,
                           label+"_conditional_nll": float(-(y[mask]*np.log(np.maximum(pred[mask]/mass,1e-15))).sum()) if count >= 2 else 0.,
                           label+"_singleton_exposures": exposure if count == 1 else 0.})
        records.append(record)
    return records


def summarize_records(records):
    frame = pd.DataFrame(records)
    result = {"dates": len(frame), "multi_candidate_dates": int(frame.candidates.ge(2).sum()),
              "singleton_dates": frame.loc[frame.candidates.eq(1), "date"].astype(int).tolist(),
              "singleton_date_exposures": float(frame.loc[frame.candidates.eq(1), "all_exposures"].sum()),
              "zero_volume_dates": frame.loc[frame.all_exposures.eq(0), "date"].astype(int).tolist(),
              "joint_exposures": float(frame.joint_exposures.sum()),
              "joint_log_loss": float(frame.joint_nll.sum()/frame.joint_exposures.sum())}
    for label in ["seen", "unseen"]:
        joint_volume = frame[label+"_joint_exposures"].sum()
        conditional_volume = frame[label+"_conditional_exposures"].sum()
        result[label] = {
            "joint_exposures": float(joint_volume),
            "observed_group_exposure_mass": float(joint_volume/frame.joint_exposures.sum()),
            "predicted_group_exposure_mass": float(frame[label+"_predicted_mass_times_day_volume"].sum()/frame.joint_exposures.sum()),
            "joint_pool_log_loss_contribution_per_group_exposure": float(frame[label+"_joint_nll"].sum()/joint_volume) if joint_volume > 0 else None,
            "within_group_conditional_exposures": float(conditional_volume),
            "within_group_conditional_log_loss": float(frame[label+"_conditional_nll"].sum()/conditional_volume) if conditional_volume > 0 else None,
            "within_group_singleton_dates": frame.loc[frame[label+"_candidates"].eq(1), "date"].astype(int).tolist(),
            "within_group_singleton_exposures": float(frame[label+"_singleton_exposures"].sum()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("new_upload_protocol.json"))
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    def progress(stage, **details):
        point = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.monotonic()-clock,3), **details)
        with (args.output/"progress.jsonl").open("a") as stream:
            stream.write(json.dumps(point)+"\n")
        print(json.dumps(point), flush=True)
    assert sha256(args.source) == config["source_sha256"]
    frame = pd.read_csv(args.source,usecols=["video_id","date","author_id","upload_dt","video_duration","show_cnt"])
    assert not frame.duplicated(["video_id","date"]).any()
    frame["day"] = pd.to_datetime(frame.date.astype(str),format="%Y%m%d")
    frame["upload_day"] = pd.to_datetime(frame.upload_dt)
    frame["duration_bucket"] = duration_bucket(frame.video_duration,config["duration_bins_milliseconds"])
    first_days = frame.loc[frame.day.eq(frame.upload_day)].copy()
    assert not first_days.video_id.duplicated().any()
    dates = np.sort(frame.date.unique())
    beta = config["beta_grid"]
    priors = config["history_prior_item_days"]
    specifications = {
        "uniform_videos":[{}], "duration_base":[{}],
        "catalogue_stock_Q":[{"beta_Q":q} for q in beta],
        "publication_flow_F":[{"beta_F":f} for f in beta],
        "stock_flow_QF":[{"beta_Q":q,"beta_F":f} for q,f in itertools.product(beta,beta)],
        "history_H":[{"history_prior_item_days":h} for h in priors],
        "history_stock_flow_HQF":[{"history_prior_item_days":h,"beta_Q":q,"beta_F":f} for h,q,f in itertools.product(priors,beta,beta)],
    }
    progress("first_day_source_ready",source_rows=len(frame),observed_first_day_candidates=len(first_days),
             model_grid_sizes={k:len(v) for k,v in specifications.items()})
    reports, daily, predictions = [], [], []
    for number,start in enumerate(config["origin_start_indices"],1):
        D,C,T = config["training_days"],config["calibration_days"],config["test_days"]
        train_dates = dates[start:start+D]
        cal_dates = dates[start+D:start+D+C]
        test_dates = dates[start+D+C:start+D+C+T]
        train_begin = pd.to_datetime(str(train_dates[0]),format="%Y%m%d")
        train_end = pd.to_datetime(str(train_dates[-1]),format="%Y%m%d")
        training = frame.loc[frame.date.isin(train_dates)]
        prefix = frame.loc[frame.day.le(train_end)].drop_duplicates("video_id")
        Q = prefix.groupby("author_id").video_id.nunique()
        F = prefix.loc[prefix.upload_day.between(train_begin,train_end)].groupby("author_id").video_id.nunique()
        old = training.loc[training.upload_day.lt(train_begin)]
        old_history = old.groupby("author_id").show_cnt.agg(["sum","size"])
        mu = float(old.show_cnt.mean())
        assert mu > 0
        train_first = first_days.loc[first_days.date.isin(train_dates)]
        global_first_mean = float(train_first.show_cnt.mean())
        assert global_first_mean > 0
        duration_summary = train_first.groupby("duration_bucket").show_cnt.agg(["sum","size"])
        duration_base = ((duration_summary["sum"]+config["duration_prior_video_count"]*global_first_mean)/
                         (duration_summary["size"]+config["duration_prior_video_count"]))
        seen_accounts = set(training.author_id)
        def make_features(part):
            f = part[["video_id","date","author_id","duration_bucket"]].copy()
            f["duration_base"] = f.duration_bucket.map(duration_base).fillna(global_first_mean)
            f["Q"] = f.author_id.map(Q).fillna(0).astype(int)
            f["F"] = f.author_id.map(F).fillna(0).astype(int)
            f["old_exposures"] = f.author_id.map(old_history["sum"]).fillna(0)
            f["old_item_days"] = f.author_id.map(old_history["size"]).fillna(0)
            f["seen_in_training"] = f.author_id.isin(seen_accounts)
            return f
        calibration = first_days.loc[first_days.date.isin(cal_dates)].reset_index(drop=True)
        test = first_days.loc[first_days.date.isin(test_dates)].reset_index(drop=True)
        assert not set(test.video_id).intersection(training.video_id)
        assert not set(test.video_id).intersection(prefix.video_id)
        cal_features, test_features = make_features(calibration), make_features(test)
        calibration["seen_in_training"] = cal_features.seen_in_training
        test["seen_in_training"] = test_features.seen_in_training
        cal_counts = calibration.groupby("date").video_id.transform("size").to_numpy()
        eligible = cal_counts >= 2
        y_cal = calibration.show_cnt.to_numpy(dtype=float)
        assert y_cal[eligible].sum() > 0
        origin = "R"+str(number)
        results = []
        for model,settings in specifications.items():
            cal_prob = probabilities(candidate_scores(cal_features,settings,model,mu),cal_features.date)
            cal_losses = -(y_cal[None,eligible]*np.log(np.maximum(cal_prob[:,eligible],1e-15))).sum(axis=1)/y_cal[eligible].sum()
            chosen = int(cal_losses.argmin())
            score = candidate_scores(test_features,[settings[chosen]],model,mu)
            prediction = probabilities(score,test_features.date)[0]
            records = evaluate(test,prediction)
            evaluation = summarize_records(records)
            results.append({"model":model,"candidate_count":len(settings),"selected_setting":settings[chosen],
                            "calibration_joint_log_loss":float(cal_losses[chosen]),"test":evaluation})
            for record in records:
                daily.append(dict(record,origin=origin,model=model))
            for i,row in test.iterrows():
                predictions.append({"origin":origin,"model":model,"video_id":int(row.video_id),"date":int(row.date),
                                    "author_id":int(row.author_id),"seen_in_training":bool(row.seen_in_training),
                                    "observed_first_day_exposure":int(row.show_cnt),"predicted_share":float(prediction[i])})
        group_counts = {label:{"test_works":int(test_features.seen_in_training.eq(flag).sum()),
                              "test_accounts":int(test_features.loc[test_features.seen_in_training.eq(flag),"author_id"].nunique()),
                              "without_old_work_history":int(test_features.loc[test_features.seen_in_training.eq(flag),"old_item_days"].eq(0).sum())}
                        for label,flag in [("seen",True),("unseen",False)]}
        reports.append({"origin":origin,"training_dates":train_dates.astype(int).tolist(),
                        "calibration_dates":cal_dates.astype(int).tolist(),"test_dates":test_dates.astype(int).tolist(),
                        "training_first_day_works":len(train_first),"calibration_first_day_works":len(calibration),
                        "test_first_day_works":len(test),"test_group_counts":group_counts,
                        "training_old_work_item_days":len(old),"known_catalogue_items_at_train_end":len(prefix),
                        "training_duration_bucket_counts":duration_summary["size"].to_dict(),"models":results})
        (args.output/"origins_progress.json").write_text(json.dumps(reports,indent=2)+"\n")
        progress("new_upload_origin_complete",origin=origin,test_first_day_works=len(test),
                 test_groups=group_counts,models=[{"model":m["model"],"loss":m["test"]["joint_log_loss"]} for m in results])
    daily_frame = pd.DataFrame(daily)
    daily_frame.to_csv(args.output/"daily_metrics.csv",index=False)
    pd.DataFrame(predictions).to_parquet(args.output/"test_predictions.parquet",index=False)
    pooled = {model:summarize_records(daily_frame.loc[daily_frame.model.eq(model)].to_dict("records")) for model in specifications}
    reference = daily_frame.loc[daily_frame.model.eq("history_H")].sort_values(["origin","date"])
    assert not reference.date.duplicated().any()
    rng = np.random.default_rng(config["bootstrap_seed"])
    samples = rng.integers(0,len(reference),size=(config["bootstrap_repetitions"],len(reference)))
    origin_samples = rng.integers(0,len(reports),size=(config["bootstrap_repetitions"],len(reports)))
    comparisons = []
    for model,baseline in [("history_stock_flow_HQF","history_H"),("history_H","duration_base"),
                           ("stock_flow_QF","duration_base"),("duration_base","uniform_videos")]:
        left = daily_frame.loc[daily_frame.model.eq(model)].sort_values(["origin","date"])
        right = daily_frame.loc[daily_frame.model.eq(baseline)].sort_values(["origin","date"])
        for label,nll_col,volume_col in [("all","joint_nll","joint_exposures"),
                                         ("seen_joint","seen_joint_nll","seen_joint_exposures"),
                                         ("unseen_joint","unseen_joint_nll","unseen_joint_exposures"),
                                         ("seen_conditional","seen_conditional_nll","seen_conditional_exposures"),
                                         ("unseen_conditional","unseen_conditional_nll","unseen_conditional_exposures")]:
            delta = left[nll_col].to_numpy()-right[nll_col].to_numpy()
            volume = left[volume_col].to_numpy()
            if volume.sum() == 0:
                continue
            bootstrap = delta[samples].sum(axis=1)/volume[samples].sum(axis=1)
            block_delta = np.asarray([x.sum() for x in np.split(delta,len(reports))])
            block_volume = np.asarray([x.sum() for x in np.split(volume,len(reports))])
            block_boot = block_delta[origin_samples].sum(axis=1)/block_volume[origin_samples].sum(axis=1)
            comparisons.append({"model":model,"baseline":baseline,"evaluation":label,
                                "test_delta_log_loss":float(delta.sum()/volume.sum()),
                                "paired_date_bootstrap_95":np.quantile(bootstrap,[.025,.975]).tolist(),
                                "origin_block_bootstrap_95":np.quantile(block_boot,[.025,.975]).tolist()})
    report = {"study_id":config["study_id"],"started_utc":started,"completed_utc":datetime.now(timezone.utc).isoformat(),
              "protocol":config,"protocol_sha256":sha256(args.protocol),"source_sha256":sha256(args.source),
              "origins":reports,"pooled_models":pooled,"comparisons":comparisons,
              "models_share_the_same_observed_candidate_pools":True,"test_target_feedback_used_as_feature":False,
              "raw_first_day_source_works":len(first_days),"all_test_works":int(sum(x["test_first_day_works"] for x in reports)),
              "test_seen_works":int(sum(x["test_group_counts"]["seen"]["test_works"] for x in reports)),
              "test_unseen_works":int(sum(x["test_group_counts"]["unseen"]["test_works"] for x in reports)),
              "limitations":["Conditional source-recorded first-day pool, not all platform releases.",
                             "Unseen in the training window is not newly registered.",
                             "Q/F are past-observed catalogue proxies; neither full production nor active inventory.",
                             "HQF has a larger global calibration grid and can overfit seven-day calibration windows.",
                             "Repeated accounts and four historical origins limit uncertainty interpretation.",
                             "No owner linkage, AI content label, exogenous cost, causal alpha, or equilibrium identified."]}
    (args.output/"new_upload_first_day.json").write_text(json.dumps(report,indent=2)+"\n")
    progress("new_upload_complete",all_test_works=report["all_test_works"],seen=report["test_seen_works"],unseen=report["test_unseen_works"],
             pooled_joint_losses={k:v["joint_log_loss"] for k,v in pooled.items()},analysis_complete=True)


if __name__ == "__main__":
    main()

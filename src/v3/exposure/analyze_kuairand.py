"""Held-out prediction of actual logged exposure, conditional on a fixed pool.

Random exposure estimates a demand proxy; ordinary exposure is the prediction
target. No AI effect, ownership mapping, production equilibrium or causal alpha
is identified. Monthly video statistic features are deliberately not loaded.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRAIN_END = 20220426
VALID_END = 20220428
FREEZE_EXCLUSIVE = pd.Timestamp("2022-04-22")
AGE_CAP = 2.0


def normalize(values):
    values = np.asarray(values, dtype=float)
    return values / values.sum()


def quality_proxy(random_train, random_valid, videos, static_duration):
    bins = [0, 7000, 18000, 30000, 60000, np.inf]
    duration = random_train.groupby("video_id").duration_ms.median().reindex(videos)
    duration = duration.fillna(pd.Series(static_duration, index=videos)).fillna(random_train.duration_ms.median())
    bucket = np.digitize(duration.to_numpy(), bins[1:-1])
    train = random_train.copy()
    train["bucket"] = np.digitize(train.duration_ms.to_numpy(), bins[1:-1])
    global_q = float(train.long_view.mean())
    priors = train.groupby("bucket").long_view.mean().to_dict()
    prior = np.array([priors.get(int(b), global_q) for b in bucket])
    grouped = train.groupby("video_id").long_view.agg(["sum", "count"]).reindex(videos, fill_value=0)
    index = pd.Index(videos)
    validation_index = index.get_indexer(random_valid.video_id)
    validation_y = random_valid.long_view.to_numpy(dtype=float)
    options = []
    for shrink in (10, 50, 200):
        q = (grouped["sum"].to_numpy() + shrink * prior) / (grouped["count"].to_numpy() + shrink)
        pred = np.clip(q[validation_index], 1e-6, 1-1e-6)
        loss = float(-np.mean(validation_y*np.log(pred)+(1-validation_y)*np.log(1-pred)))
        options.append((loss, shrink, q))
    best = min(options, key=lambda row: row[0])
    def binary_loss(pred):
        pred = np.clip(pred,1e-6,1-1e-6)
        return float(-np.mean(validation_y*np.log(pred)+(1-validation_y)*np.log(1-pred)))
    return best[2], {"selected_shrinkage":best[1], "random_validation_log_loss":best[0],
                    "random_training_rows":len(train),"random_validation_rows":len(random_valid),
                    "random_validation_global_prior_log_loss":binary_loss(np.full(len(validation_y),global_q)),
                    "random_validation_duration_prior_log_loss":binary_loss(prior[validation_index]),
                    "pool_videos_with_zero_random_training_exposures":int(grouped["count"].eq(0).sum()),
                    "random_training_global_long_view":global_q,
                    "candidates":[{"shrinkage":s,"validation_log_loss":l} for l,s,_ in options]}


def evaluate_tab(tab, standard, random_log, basic, output):
    standard = standard.loc[standard.tab.eq(tab)].copy()
    random_log = random_log.loc[random_log.tab.eq(tab)].copy()
    random_train = random_log.loc[random_log.date.le(TRAIN_END)]
    random_valid = random_log.loc[random_log.date.gt(TRAIN_END) & random_log.date.le(VALID_END)]
    train = standard.loc[standard.date.le(TRAIN_END)]
    valid = standard.loc[standard.date.gt(TRAIN_END) & standard.date.le(VALID_END)]
    test = standard.loc[standard.date.gt(VALID_END)]
    if min(len(random_train),len(random_valid),len(train),len(valid),len(test)) < 100:
        return {"tab":int(tab),"status":"insufficient_rows","counts":{
            "random_train":len(random_train),"random_validation":len(random_valid),
            "standard_train":len(train),"standard_validation":len(valid),"standard_test":len(test)}}
    videos = basic.video_id.to_numpy()
    authors, author_index = np.unique(basic.author_id.to_numpy(), return_inverse=True)
    num_authors = len(authors)
    inventory = np.bincount(author_index, minlength=num_authors).astype(float)
    log_inventory = np.log(inventory)
    lookup = pd.Series(author_index, index=videos)
    q, demand_audit = quality_proxy(random_train, random_valid, videos, basic.video_duration.to_numpy(dtype=float))
    dates = sorted(standard.date.unique())
    outcomes = []
    for date in dates:
        indices = standard.loc[standard.date.eq(date),"video_id"].map(lookup).to_numpy(dtype=int)
        outcomes.append(np.bincount(indices,minlength=num_authors))
    outcomes = np.asarray(outcomes,dtype=float)
    days = pd.to_datetime(pd.Series(dates).astype(str),format="%Y%m%d")
    ages = np.maximum((days.to_numpy()[:,None] - basic.upload_date.to_numpy()[None,:]) / np.timedelta64(1,"D"),0)
    log_age = np.log1p(ages)
    dates = np.asarray(dates)
    train_mask, valid_mask, test_mask = dates<=TRAIN_END, (dates>TRAIN_END)&(dates<=VALID_END), dates>VALID_END
    log_q = np.log(np.clip(q,1e-6,1-1e-6))
    all_rows = []
    predictions = {}

    def probabilities(params,kind):
        if kind=="historical":
            base=normalize(outcomes[train_mask].sum(axis=0)+.5)
            return np.tile(base,(len(dates),1))
        if kind=="uniform":
            return np.full_like(outcomes,1/num_authors)
        if kind=="quantity":
            base=normalize(inventory**params[0])
            return np.tile(base,(len(dates),1))
        alpha,beta,gamma=params
        row_predictions=[]
        for row in range(len(dates)):
            log_weights=beta*log_q-gamma*log_age[row]
            weights=np.exp(log_weights-log_weights.max())
            author_score=np.bincount(author_index,weights=weights,minlength=num_authors)*np.exp((alpha-1)*log_inventory)
            row_predictions.append(normalize(author_score))
        return np.asarray(row_predictions)

    def cross_entropy(pred,mask):
        y=outcomes[mask]
        return float(-(y*np.log(np.maximum(pred[mask],1e-15))).sum()/y.sum())

    specs=[
        ("uniform_accounts","uniform",[],[]),
        ("linear_quantity","quantity",[1.],[(1.,1.)]),
        ("concave_quantity","quantity",[.5],[(.05,1.)]),
        ("free_quantity","quantity",[1.],[(.05,2.)]),
        ("linear_demand_age","quality",[1.,1.,min(.5,AGE_CAP)],[(1.,1.),(0.,4.),(0.,AGE_CAP)]),
        ("concave_demand_age","quality",[.7,1.,min(.5,AGE_CAP)],[(.05,1.),(0.,4.),(0.,AGE_CAP)]),
        ("free_demand_age","quality",[1.,1.,min(.5,AGE_CAP)],[(.05,2.),(0.,4.),(0.,AGE_CAP)]),
        ("historical_share","historical",[],[]),
    ]
    for name,kind,initial,bounds in specs:
        if initial and any(lo!=hi for lo,hi in bounds):
            fitted=minimize(lambda p:cross_entropy(probabilities(p,kind),train_mask),initial,
                            method="L-BFGS-B",bounds=bounds,options={"maxiter":300,"ftol":1e-11})
            params=fitted.x.tolist()
            convergence={"success":bool(fitted.success),"message":str(fitted.message)}
        else:
            params=initial
            convergence={"success":True,"message":"fixed baseline"}
        pred=probabilities(params,kind)
        predictions[name]=pred
        y_test=outcomes[test_mask].sum(axis=0)
        p_test=(pred[test_mask]*outcomes[test_mask].sum(axis=1)[:,None]).sum(axis=0)/y_test.sum()
        row={"model":name,"parameters":params,"parameter_bounds":bounds,"convergence":convergence,
             "parameters_at_bound":[i for i,(v,(lo,hi)) in enumerate(zip(params,bounds)) if lo!=hi and min(abs(v-lo),abs(v-hi))<1e-5],
             "training_log_loss":cross_entropy(pred,train_mask),
             "validation_log_loss":cross_entropy(pred,valid_mask),
             "test_log_loss":cross_entropy(pred,test_mask),
             "test_share_total_variation":float(.5*np.abs(normalize(y_test)-p_test).sum())}
        all_rows.append(row)
    # Test comparisons are reported after every model has been fitted on training.
    rng=np.random.default_rng(20261104)
    comparisons=[]
    for model,baseline in [("concave_quantity","linear_quantity"),
                           ("free_quantity","linear_quantity"),
                           ("concave_demand_age","linear_demand_age"),
                           ("free_demand_age","linear_demand_age")]:
        delta_per_day=-(outcomes[test_mask]*(np.log(predictions[model][test_mask])-np.log(predictions[baseline][test_mask]))).sum(axis=1)
        totals=outcomes[test_mask].sum(axis=1)
        sampled=rng.integers(0,len(totals),size=(2000,len(totals)))
        deltas=delta_per_day[sampled].sum(axis=1)/totals[sampled].sum(axis=1)
        comparisons.append({"model":model,"baseline":baseline,
                            "test_delta_log_loss":float(delta_per_day.sum()/totals.sum()),
                            "day_block_bootstrap_95_interval":np.quantile(deltas,[.025,.975]).tolist(),
                            "interpretation":"Negative means lower held-out loss; exploratory day-block uncertainty, not causal inference."})
    primary=min(all_rows,key=lambda row:row["validation_log_loss"])["model"]
    report={"tab":int(tab),"status":"complete","pool_videos":len(videos),"pool_author_accounts":num_authors,
            "author_inventory_quantiles":{str(z):float(np.quantile(inventory,z)) for z in [0,.5,.9,.99,1]},
            "single_item_author_fraction":float(np.mean(inventory==1)),
            "multi_item_author_fraction":float(np.mean(inventory>=2)),
            "standard_train_rows":len(train),"standard_validation_rows":len(valid),"standard_test_rows":len(test),
            "test_days":dates[test_mask].tolist(),"zero_test_exposure_authors":int(np.sum(outcomes[test_mask].sum(axis=0)==0)),
            "random_demand_proxy":demand_audit,"validation_selected_model":primary,"models":all_rows,"comparisons":comparisons,
            "inventory_definition":"Number of fixed candidate-pool videos uploaded by 2022-04-21; not complete creator production."}
    print(json.dumps({"tab":int(tab),"complete":True,"selected":primary,"models":[{"name":x["model"],"params":x["parameters"],"test_loss":x["test_log_loss"]} for x in all_rows]}),flush=True)
    return report


def main():
    global AGE_CAP
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("--age-cap",type=float,default=2.)
    parser.add_argument("--result-name",default="exposure_prediction.json")
    args=parser.parse_args()
    AGE_CAP=args.age_cap
    data=next((args.root/"raw").rglob("video_features_basic_pure.csv")).parent
    basic=pd.read_csv(data/"video_features_basic_pure.csv")
    metadata_summary={"files":{p.name:{"bytes":p.stat().st_size} for p in data.glob("*.csv")},"basic_columns":list(basic)}
    raw_upload=basic.upload_dt.astype(str).str.replace(r"\.0$","",regex=True)
    basic["upload_date"]=pd.to_datetime(raw_upload,errors="coerce")
    metadata_summary.update({"basic_rows":len(basic),"unparseable_upload_dates":int(basic.upload_date.isna().sum()),"raw_upload_examples":raw_upload.head(5).tolist(),"parsed_upload_examples":basic.upload_date.head(5).astype(str).tolist()})
    basic=basic.loc[basic.author_id.ge(0)&basic.upload_date.notna()&basic.upload_date.lt(FREEZE_EXCLUSIVE)].copy().sort_values("video_id")
    metadata_summary["eligible_fixed_pool_rows"]=len(basic)
    columns=["user_id","video_id","date","is_rand","tab","long_view","duration_ms"]
    standard=pd.read_csv(data/"log_standard_4_22_to_5_08_pure.csv",usecols=columns)
    random_log=pd.read_csv(data/"log_random_4_22_to_5_08_pure.csv",usecols=columns)
    metadata_summary.update({"standard_rows_before_pool_filter":len(standard),"random_rows_before_pool_filter":len(random_log),
                             "standard_date_range":[int(standard.date.min()),int(standard.date.max())],
                             "random_date_range":[int(random_log.date.min()),int(random_log.date.max())],
                             "standard_is_rand_counts":standard.is_rand.value_counts().to_dict(),
                             "random_is_rand_counts":random_log.is_rand.value_counts().to_dict()})
    standard=standard.loc[standard.video_id.isin(basic.video_id)&standard.is_rand.eq(0)]
    random_log=random_log.loc[random_log.video_id.isin(basic.video_id)&random_log.is_rand.eq(1)]
    tab_counts=standard.loc[standard.date.le(TRAIN_END)].tab.value_counts()
    chosen_tabs=tab_counts.index[:3].tolist()
    print(json.dumps({"stage":"loaded","fixed_pool":len(basic),"standard":len(standard),"random":len(random_log),"train_selected_tabs":chosen_tabs}),flush=True)
    reports=[evaluate_tab(t,standard,random_log,basic,args.root/"results") for t in chosen_tabs]
    result={"completed_utc":datetime.now(timezone.utc).isoformat(),"protocol":{"freeze_date":"all uploads before 2022-04-22","training":"2022-04-22..26","validation":"2022-04-27..28","test":"2022-04-29..2022-05-08","tabs":"up to three largest by ordinary training exposure count","future_monthly_features_used":False,"test_outcomes_used_to_fit_parameters":False,"quantity_alpha_bounds":"concave .05..1; unrestricted-positive .05..2 (does not test nonpositive alpha)","score":"n_a^(alpha-1) sum_v q_v^beta (1+age_vd)^(-gamma)"},
            "metadata_summary":metadata_summary,"tabs":reports,
            "scope":"Held-out conditional exposure prediction on historical non-AIGC Kuaishou data. Does not identify AI cost effects, latent owners, creator best responses, or a causal platform alpha."}
    result["protocol"]["age_exponent_cap"]=AGE_CAP
    output=args.root/"results"
    output.mkdir(exist_ok=True)
    (output/args.result_name).write_text(json.dumps(result,indent=2))
    print(json.dumps({"stage":"complete","result":args.result_name}),flush=True)


if __name__=="__main__":
    main()

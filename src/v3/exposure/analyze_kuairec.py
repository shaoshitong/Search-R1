"""Temporal held-out exposure prediction in a balanced KuaiRec item panel.

The retrospective common-item panel avoids treating absent daily rows as zero.
Inventory is selected-item inventory, never total creator production or owners.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def normalized(x):
    x = np.maximum(np.asarray(x, dtype=float), 1e-15)
    return x / x.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    columns = ["video_id", "date", "author_id", "upload_dt", "show_cnt", "long_time_play_cnt"]
    frame = pd.read_csv(args.root / "item_daily_features.csv", usecols=columns)
    dates = np.sort(frame.date.unique())
    train_end, valid_end = int(dates[29]), int(dates[41])
    counts = frame.groupby("video_id").date.nunique()
    complete = counts.index[counts.eq(len(dates))]
    metadata = frame.drop_duplicates("video_id").set_index("video_id")
    uploaded = pd.to_datetime(metadata.upload_dt, errors="coerce")
    first_day = pd.to_datetime(str(dates[0]), format="%Y%m%d")
    eligible = uploaded.index[uploaded.le(first_day) & metadata.author_id.ge(0)]
    videos = np.sort(complete.intersection(eligible).to_numpy())
    metadata = metadata.loc[videos]
    selected = frame.loc[frame.video_id.isin(videos)].copy()
    if len(videos) == 0:
        raise ValueError("No complete common-item panel")
    authors, mapping = np.unique(metadata.author_id.to_numpy(), return_inverse=True)
    inventory = np.bincount(mapping).astype(float)
    lookup = pd.Series(mapping, index=videos)
    selected["author_index"] = selected.video_id.map(lookup)
    outcomes = selected.groupby(["date", "author_index"]).show_cnt.sum().unstack(fill_value=0).reindex(index=dates, columns=np.arange(len(authors))).to_numpy(dtype=float)
    train_mask, valid_mask, test_mask = dates <= train_end, (dates > train_end) & (dates <= valid_end), dates > valid_end
    # A pre-test demand proxy, not intrinsic quality. Neither test plays nor
    # test show counts enter the score or fitting objective.
    train_totals = selected.loc[selected.date.le(train_end)].groupby("video_id")[["show_cnt", "long_time_play_cnt"]].sum().reindex(videos)
    numerator = train_totals.long_time_play_cnt.to_numpy(dtype=float)
    denominator = train_totals.show_cnt.to_numpy(dtype=float)
    global_q = float(numerator.sum() / denominator.sum())
    q_raw = (numerator + 50 * global_q) / (denominator + 50)
    q = np.clip(q_raw, 1e-6, 1-1e-6)
    days = pd.to_datetime(pd.Series(dates).astype(str), format="%Y%m%d")
    ages = np.maximum((days.to_numpy()[:, None] - uploaded.loc[videos].to_numpy()[None, :]) / np.timedelta64(1, "D"), 0)
    log_age, log_q, log_n = np.log1p(ages), np.log(q), np.log(inventory)

    def probabilities(parameters, kind):
        if kind == "historical":
            return np.tile(normalized(outcomes[train_mask].sum(axis=0)+.5), (len(dates), 1))
        if kind == "uniform":
            return np.full_like(outcomes, 1/len(authors))
        if kind == "quantity":
            return np.tile(normalized(inventory**parameters[0]), (len(dates), 1))
        alpha, beta, gamma = parameters
        return np.asarray([normalized(np.bincount(mapping, weights=np.exp(beta*log_q-gamma*age), minlength=len(authors))*np.exp((alpha-1)*log_n)) for age in log_age])

    def loss(pred, mask):
        y = outcomes[mask]
        return float(-(y*np.log(np.maximum(pred[mask], 1e-15))).sum()/y.sum())

    specs = [
        ("uniform_accounts", "uniform", [], []),
        ("linear_quantity", "quantity", [1.], [(1., 1.)]),
        ("concave_quantity", "quantity", [.5], [(.05, 1.)]),
        ("free_quantity", "quantity", [1.], [(.05, 2.)]),
        ("linear_demand_age", "demand", [1., 1., .5], [(1., 1.), (0., 4.), (0., 2.)]),
        ("concave_demand_age", "demand", [.7, 1., .5], [(.05, 1.), (0., 4.), (0., 2.)]),
        ("free_demand_age", "demand", [1., 1., .5], [(.05, 2.), (0., 4.), (0., 2.)]),
        ("historical_share", "historical", [], []),
    ]
    reports, predictions = [], {}
    for name, kind, initial, bounds in specs:
        if any(lo != hi for lo, hi in bounds):
            fit = minimize(lambda p: loss(probabilities(p, kind), train_mask), initial, method="L-BFGS-B", bounds=bounds, options={"maxiter":300, "ftol":1e-11})
            parameters = fit.x.tolist()
            convergence = {"success":bool(fit.success), "message":str(fit.message)}
        else:
            parameters, convergence = initial, {"success":True, "message":"fixed baseline"}
        pred = probabilities(parameters, kind)
        predictions[name] = pred
        y = outcomes[test_mask].sum(axis=0)
        share_pred = (pred[test_mask]*outcomes[test_mask].sum(axis=1)[:, None]).sum(axis=0)/y.sum()
        reports.append({"model":name, "parameters":parameters, "convergence":convergence,
                        "training_log_loss":loss(pred, train_mask), "validation_log_loss":loss(pred, valid_mask), "test_log_loss":loss(pred, test_mask),
                        "test_share_total_variation":float(.5*np.abs(normalized(y)-share_pred).sum())})
        print(json.dumps({"model":name, "parameters":parameters, "test_loss":reports[-1]["test_log_loss"]}), flush=True)
    rng, comparisons = np.random.default_rng(20261104), []
    for model, baseline in [("concave_quantity", "linear_quantity"), ("free_quantity", "linear_quantity"), ("concave_demand_age", "linear_demand_age"), ("free_demand_age", "linear_demand_age")]:
        delta = -(outcomes[test_mask]*(np.log(predictions[model][test_mask])-np.log(predictions[baseline][test_mask]))).sum(axis=1)
        totals = outcomes[test_mask].sum(axis=1)
        sampled = rng.integers(0, len(totals), size=(2000, len(totals)))
        bootstrap = delta[sampled].sum(axis=1)/totals[sampled].sum(axis=1)
        comparisons.append({"model":model, "baseline":baseline, "test_delta_log_loss":float(delta.sum()/totals.sum()), "day_block_bootstrap_95_interval":np.quantile(bootstrap, [.025,.975]).tolist()})
    result = {"completed_utc":datetime.now(timezone.utc).isoformat(),
              "protocol":{"panel":"Retrospective fixed items observed on all 63 dates, uploaded by first date; absent rows never imputed as zero.", "training_dates":[int(dates[0]), train_end], "validation_dates":[int(dates[30]), valid_end], "test_dates":[int(dates[42]), int(dates[-1])], "future_outcomes_used_for_fitting":False, "future_row_presence_used_for_sample_definition":True, "demand_proxy":"training long_time_play_cnt / show_cnt with 50-exposure shrinkage", "bounds":"alpha .05..1 or .05..2; beta 0..4; gamma 0..2"},
              "sample":{"source_rows":len(frame), "balanced_rows":len(selected), "videos":len(videos), "authors":len(authors), "dates":len(dates), "single_item_author_fraction":float(np.mean(inventory==1)), "multi_item_authors":int(np.sum(inventory>=2)), "inventory_max":int(inventory.max()), "exposure_totals":{"training":int(outcomes[train_mask].sum()), "validation":int(outcomes[valid_mask].sum()), "test":int(outcomes[test_mask].sum())}, "nonpositive_training_inventory_shows":int(np.sum(denominator<=0)), "training_proxy_before_clip_outside_0_1":int(np.sum((q_raw<0)|(q_raw>1)))},
              "models":reports, "comparisons":comparisons, "validation_selected_model":min(reports, key=lambda x:x["validation_log_loss"])["model"],
              "scope":"Conditional descriptive exposure prediction in a selected historical non-AI panel; no causal alpha, AI adoption effect, full production, owner mapping or Nash equilibrium identified."}
    output = args.root / "results"
    output.mkdir(exist_ok=True)
    (output / "exposure_prediction.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"complete":True, "sample":result["sample"]}), flush=True)


if __name__ == "__main__":
    main()

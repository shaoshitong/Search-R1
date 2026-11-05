"""Prespecified V5 cross-work predictive transfer; no causal owner/quality claim."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(values):
    values = np.maximum(np.asarray(values, dtype=float), 1e-15)
    return values / values.sum(axis=-1, keepdims=True)


def daily_nll(y, probabilities):
    return -(y * np.log(np.maximum(probabilities, 1e-15))).sum(axis=-1)


def loss(y, probabilities):
    return float(daily_nll(y, probabilities).sum() / y.sum())


def content_probability(parameters, n, item_accounts, q, log_age):
    alpha, beta, gamma = parameters
    log_weights = beta * np.log(q)[None, :] - gamma * log_age
    weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    scores = np.asarray([np.bincount(item_accounts, weights=row, minlength=len(n))
                         for row in weights]) * n[None, :] ** (alpha - 1)
    return normalize(scores)


def fit_content(y, n, item_accounts, q, log_age, config):
    quantity_varies = bool(np.unique(n).size > 1)
    bounds = [tuple(config["quantity_alpha_bounds"]) if quantity_varies else (1., 1.),
              tuple(config["demand_beta_bounds"]), tuple(config["age_gamma_bounds"])]
    fits = []
    for initial in config["demand_age_starts"]:
        initial = list(initial)
        if not quantity_varies:
            initial[0] = 1.
        fit = minimize(lambda p: loss(y, content_probability(p, n, item_accounts, q, log_age)),
                       initial, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-12})
        fits.append(fit)
    converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not converged:
        raise RuntimeError("Prespecified content fits did not converge: " + str([x.message for x in fits]))
    best = min(converged, key=lambda fit: fit.fun)
    return best.x, {"success": bool(best.success), "message": str(best.message),
                    "calibration_loss": float(best.fun), "bounds": bounds,
                    "alpha_estimable_from_quantity_variation": quantity_varies,
                    "nominal_global_parameters": 3 if quantity_varies else 2,
                    "converged_starts": len(converged),
                    "parameters_at_bound": [i for i, (v, (lo, hi)) in enumerate(zip(best.x, bounds))
                                            if lo != hi and min(abs(v-lo), abs(v-hi)) < 1e-5]}


def pick_candidates(cal_prob, test_prob, cal_y, test_y):
    # Shape: candidate × identity mapping × day × account. Every identity
    # mapping selects its own calibration candidate, with identical capacity.
    cal_losses = -(cal_y[None, None] * np.log(np.maximum(cal_prob, 1e-15))).sum(axis=(-2, -1)) / cal_y.sum()
    choices = cal_losses.argmin(axis=0)
    selected = test_prob[choices, np.arange(test_prob.shape[1])]
    nll = -(test_y[None] * np.log(np.maximum(selected, 1e-15))).sum(axis=-1)
    return selected, choices, cal_losses[choices, np.arange(test_prob.shape[1])], nll


def flow_stock_audit(frame, origins, dates):
    meta = frame.drop_duplicates("video_id")[["video_id", "author_id", "upload_dt"]].copy()
    meta["upload"] = pd.to_datetime(meta.upload_dt, errors="coerce")
    meta["first_seen"] = pd.to_datetime(meta.video_id.map(frame.groupby("video_id").date.min()).astype(str), format="%Y%m%d")
    meta["last_seen"] = pd.to_datetime(meta.video_id.map(frame.groupby("video_id").date.max()).astype(str), format="%Y%m%d")
    beginning = pd.to_datetime(str(dates[0]), format="%Y%m%d")
    ending = pd.to_datetime(str(dates[-1]), format="%Y%m%d")
    new = meta.loc[meta.upload.between(beginning, ending)]
    observed_pairs = pd.MultiIndex.from_frame(frame[["video_id", "date"]])
    upload_pairs = pd.MultiIndex.from_arrays([new.video_id, new.upload.dt.strftime("%Y%m%d").astype(int)])
    row_uploads = pd.to_datetime(frame.upload_dt, errors="coerce")
    row_dates = pd.to_datetime(frame.date.astype(str), format="%Y%m%d")
    daily_flow = new.groupby(new.upload.dt.strftime("%Y%m%d")).video_id.nunique()
    eligibility = []
    for origin in origins:
        training_dates, test_dates = origin["training_dates"], origin["test_dates"]
        seen_accounts = set(frame.loc[frame.date.isin(training_dates), "author_id"])
        start = pd.to_datetime(str(test_dates[0]), format="%Y%m%d")
        stop = pd.to_datetime(str(test_dates[-1]), format="%Y%m%d")
        launches = new.loc[new.upload.between(start, stop)]
        observed = frame.loc[frame.date.isin(test_dates) & frame.video_id.isin(launches.video_id)]
        eligibility.append({"origin": origin["origin"], "new_catalogue_items_uploaded_in_test_window": len(launches),
                            "new_items_with_any_test_row": int(observed.video_id.nunique()),
                            "new_items_from_accounts_seen_in_training": int(launches.author_id.isin(seen_accounts).sum()),
                            "new_items_from_accounts_not_seen_in_training": int((~launches.author_id.isin(seen_accounts)).sum()),
                            "observed_new_item_test_rows": len(observed)})
    return {"source_catalogue_items": len(meta), "source_account_ids": int(meta.author_id.nunique()),
            "source_date_range": [int(dates[0]), int(dates[-1])],
            "new_catalogue_items_uploaded_in_observation_period": len(new),
            "new_catalogue_accounts": int(new.author_id.nunique()),
            "new_items_per_account_quantiles": {str(k): float(v) for k, v in new.groupby("author_id").size().quantile([0,.5,.9,.99,1]).items()},
            "new_item_first_seen_minus_upload_days_quantiles": {str(k): float(v) for k, v in (new.first_seen-new.upload).dt.days.quantile([0,.5,.9,.99,1]).items()},
            "rows_before_upload_date": int((row_dates < row_uploads).sum()),
            "new_items_without_row_on_upload_date": int((~upload_pairs.isin(observed_pairs)).sum()),
            "new_items_first_seen_after_upload_date": int((new.first_seen > new.upload).sum()),
            "observed_pool_stock_on_first_date": int(meta.upload.le(beginning).sum()),
            "observed_pool_stock_on_last_date": int(meta.upload.le(ending).sum()),
            "daily_catalogue_upload_flow": {str(k): int(v) for k, v in daily_flow.items()},
            "new_upload_test_window_feasibility": eligibility,
            "interpretation": "Upload dates support selected-catalogue publication flow and cumulative-stock proxies, not complete creator production or historical active inventory. Missing item-days remain unobserved. No new-upload prediction or cost response is fitted here."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.monotonic()
    config = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / "progress.jsonl"

    def progress(stage, **details):
        record = dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                      elapsed_seconds=round(time.monotonic()-start_clock, 3), **details)
        with progress_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    source_sha = sha256(args.source)
    assert source_sha == config["source_sha256"], "Source differs from frozen protocol"
    frame = pd.read_csv(args.source, usecols=["video_id", "date", "author_id", "upload_dt", "show_cnt", "long_time_play_cnt"])
    assert not frame.duplicated(["video_id", "date"]).any()
    assert frame.show_cnt.notna().all() and frame.show_cnt.ge(0).all()
    dates = np.sort(frame.date.unique())
    assert len(dates) == 63
    counts = frame.groupby("video_id").date.nunique()
    metadata = frame.drop_duplicates("video_id").set_index("video_id")
    uploaded = pd.to_datetime(metadata.upload_dt, errors="coerce")
    first_day = pd.to_datetime(str(dates[0]), format="%Y%m%d")
    eligible = counts.index[counts.eq(len(dates))].intersection(uploaded.index[uploaded.le(first_day) & metadata.author_id.ge(0)])
    common = metadata.loc[eligible].copy()
    common_counts = common.groupby("author_id").size()
    authors = np.sort(common_counts.index[common_counts.ge(2)].to_numpy())
    assert (len(common), common.author_id.nunique(), len(authors)) == (
        config["expected_common_videos"], config["expected_common_accounts"], config["expected_multi_item_accounts"])
    split_rows = []
    for author in authors:
        videos = common.loc[common.author_id.eq(author)].index.to_list()
        hashes = {v: hashlib.sha256((config["split_salt"] + str(int(v))).encode()).hexdigest() for v in videos}
        for rank, video in enumerate(sorted(videos, key=lambda v: (hashes[v], v))):
            split_rows.append({"video_id": int(video), "author_id": int(author),
                               "side": "A" if rank % 2 == 0 else "B", "split_hash": hashes[video]})
    split = pd.DataFrame(split_rows).sort_values("video_id")
    split.to_csv(args.output / "split_assignments.csv", index=False)
    selected = frame.merge(split[["video_id", "side"]], on="video_id", how="inner")
    sides, inventory, item_accounts, item_dates, q_stats = {}, {}, {}, {}, {}
    for side in ["A", "B"]:
        item_meta = split.loc[split.side.eq(side)].sort_values("video_id")
        item_ids = item_meta.video_id.to_numpy()
        mapping = pd.Index(authors).get_indexer(item_meta.author_id)
        inventory[side] = np.bincount(mapping, minlength=len(authors)).astype(float)
        assert np.all(inventory[side] > 0)
        y = selected.loc[selected.side.eq(side)].groupby(["date", "author_id"]).show_cnt.sum().unstack().reindex(index=dates, columns=authors)
        assert y.notna().all().all(), "Missing rows must not be imputed as zero"
        sides[side] = y.to_numpy(dtype=float)
        item_accounts[side] = mapping
        item_dates[side] = item_ids
    strata = {}
    for index, values in enumerate(zip(inventory["A"], inventory["B"])):
        strata.setdefault(tuple(map(int, values)), []).append(index)
    rng = np.random.default_rng(config["permutation_seed"])
    identity_maps = [np.arange(len(authors))]
    for _ in range(config["permutations"]):
        mapping = np.arange(len(authors))
        for indices in strata.values():
            mapping[indices] = rng.permutation(indices)
        identity_maps.append(mapping)
    identity_maps = np.asarray(identity_maps)
    np.savez_compressed(args.output / "identity_permutations.npz", indices=identity_maps)
    sample = {"common_old_items": len(common), "common_accounts": int(common.author_id.nunique()),
              "matched_multi_item_accounts": len(authors), "matched_items": len(split),
              "A_items": int(split.side.eq("A").sum()), "B_items": int(split.side.eq("B").sum()),
              "inventory_strata": [{"A_items": key[0], "B_items": key[1], "accounts": len(indices)} for key, indices in sorted(strata.items())],
              "permutable_accounts": sum(len(indices) for indices in strata.values() if len(indices) > 1),
              "future_row_presence_used_for_selection": True,
              "split_assignments_sha256": sha256(args.output / "split_assignments.csv"),
              "identity_permutations_sha256": sha256(args.output / "identity_permutations.npz")}
    progress("cohort_and_split_ready", **sample)
    all_model_daily, all_permutation_daily = {}, {"raw": [], "residual": []}
    daily_records, prediction_records, reports, origin_metadata = [], [], [], []
    for origin_number, begin in enumerate(config["origin_start_indices"], 1):
        D, C, T = config["training_days"], config["calibration_days"], config["test_days"]
        train = np.arange(begin, begin+D)
        calibration = np.arange(begin+D, begin+D+C)
        test = np.arange(begin+D+C, begin+D+C+T)
        assert dates[train[-1]] < dates[calibration[0]] <= dates[calibration[-1]] < dates[test[0]]
        name = "R" + str(origin_number)
        origin = {"origin": name, "training_dates": dates[train].astype(int).tolist(),
                  "calibration_dates": dates[calibration].astype(int).tolist(), "test_dates": dates[test].astype(int).tolist()}
        origin_metadata.append(origin)
        y_cal, y_test = sides["B"][calibration], sides["B"][test]
        assert sides["A"][train].sum() > 0 and y_cal.sum() > 0 and y_test.sum() > 0
        models = []

        def register(model, probabilities, calibration_loss, **details):
            daily = daily_nll(y_test, probabilities)
            average_pred = (probabilities * y_test.sum(axis=1)[:, None]).sum(axis=0) / y_test.sum()
            row = dict(model=model, calibration_log_loss=float(calibration_loss),
                       test_log_loss=float(daily.sum()/y_test.sum()),
                       aggregate_share_total_variation=float(.5*np.abs(normalize(y_test.sum(axis=0))-average_pred).sum()), **details)
            models.append(row)
            all_model_daily.setdefault(model, []).append(daily)
            for d, volume, nll in zip(dates[test], y_test.sum(axis=1), daily):
                daily_records.append({"origin": name, "date": int(d), "model": model,
                                      "B_exposures": int(volume), "nll_total": float(nll),
                                      "log_loss": float(nll/volume) if volume > 0 else None})
            for day_index, day in enumerate(dates[test]):
                for author_index, author in enumerate(authors):
                    prediction_records.append({"origin": name, "date": int(day), "author_id": int(author),
                                               "model": model, "B_exposures": int(y_test[day_index, author_index]),
                                               "predicted_share": float(probabilities[day_index, author_index])})

        # Primary: no B training outcomes enter the transferred A rate.
        grid = np.asarray(config["history_prior_days"], dtype=float)
        H_A = sides["A"][train].sum(axis=0)
        mu_A = H_A.sum() / (D*inventory["A"].sum())
        rates = (H_A[None, :]/inventory["A"][None, :] + grid[:, None]*mu_A)/(D+grid[:, None])
        mapped = normalize(rates[:, identity_maps] * inventory["B"][None, None, :])
        cal_tensor = np.broadcast_to(mapped[:, :, None, :], (*mapped.shape[:2], C, len(authors)))
        test_tensor = np.broadcast_to(mapped[:, :, None, :], (*mapped.shape[:2], T, len(authors)))
        selected_p, choices, selected_cal, permutation_daily = pick_candidates(cal_tensor, test_tensor, y_cal, y_test)
        register("A_transfer_history", selected_p[0], selected_cal[0], selected_prior_days=float(grid[choices[0]]),
                 information="A training exposure history, metadata inventory, and global B calibration choice")
        all_permutation_daily["raw"].append(permutation_daily)
        raw_origin_rank = float((1+np.sum(permutation_daily[1:].sum(axis=1) <= permutation_daily[0].sum()))/(config["permutations"]+1))

        uniform = np.full((T, len(authors)), 1/len(authors))
        register("uniform_accounts", uniform, loss(y_cal, uniform))
        linear = np.broadcast_to(normalize(inventory["B"]), (T, len(authors)))
        register("linear_quantity_B", linear, loss(y_cal, linear), parameters=[1.])
        if np.unique(inventory["B"]).size > 1:
            alpha_loss = lambda alpha: loss(y_cal, normalize(inventory["B"]**alpha))
            fit = minimize_scalar(alpha_loss, bounds=config["quantity_alpha_bounds"], method="bounded")
            candidates = [(float(fit.x), float(fit.fun))] + [(float(v), alpha_loss(v)) for v in config["quantity_alpha_bounds"]]
            alpha, cal_loss = min(candidates, key=lambda value: value[1])
            register("free_quantity_B", np.broadcast_to(normalize(inventory["B"]**alpha), (T,len(authors))),
                     cal_loss, parameters=[alpha], convergence={"success": bool(fit.success)}, alpha_estimable=True)
        else:
            register("free_quantity_B", linear, loss(y_cal, linear), parameters=[1.], alpha_estimable=False)
        H_B = sides["B"][train].sum(axis=0)
        mu_B = H_B.sum()/(D*inventory["B"].sum())
        own_p = normalize(H_B[None, :] + grid[:, None]*mu_B*inventory["B"][None, :])
        own_losses = np.asarray([loss(y_cal, p) for p in own_p])
        own_choice = int(own_losses.argmin())
        register("B_own_history", np.broadcast_to(own_p[own_choice], (T,len(authors))), own_losses[own_choice],
                 selected_prior_days=float(grid[own_choice]), information="Richer-information memory benchmark using B training outcomes")

        # Training-only ordinary-traffic demand proxies, with explicit source.
        train_rows = selected.loc[selected.date.isin(dates[train])]
        grouped = train_rows.groupby("video_id")[["show_cnt", "long_time_play_cnt"]].sum()
        global_all_q = float(grouped.long_time_play_cnt.sum()/grouped.show_cnt.sum())
        q, age = {}, {}
        all_days = pd.to_datetime(pd.Series(dates).astype(str), format="%Y%m%d").to_numpy()
        for side in ["A", "B"]:
            totals = grouped.reindex(item_dates[side])
            q[side] = np.clip((totals.long_time_play_cnt.to_numpy()+config["demand_prior_exposures"]*global_all_q)/
                              (totals.show_cnt.to_numpy()+config["demand_prior_exposures"]), 1e-6, 1-1e-6)
            uploads = uploaded.loc[item_dates[side]].to_numpy()
            age[side] = np.log1p(np.maximum((all_days[:,None]-uploads[None,:])/np.timedelta64(1,"D"),0))
        # Primary comparator's B-only global prior, as specified in protocol.
        b_totals = grouped.reindex(item_dates["B"])
        b_global_q = float(b_totals.long_time_play_cnt.sum()/b_totals.show_cnt.sum())
        q_B_only = np.clip((b_totals.long_time_play_cnt.to_numpy()+config["demand_prior_exposures"]*b_global_q)/
                           (b_totals.show_cnt.to_numpy()+config["demand_prior_exposures"]), 1e-6, 1-1e-6)
        params_B, fit_B = fit_content(y_cal, inventory["B"], item_accounts["B"], q_B_only, age["B"][calibration], config)
        base_B = content_probability(params_B, inventory["B"], item_accounts["B"], q_B_only, age["B"][test])
        register("quantity_demand_age_B", base_B, fit_B["calibration_loss"], parameters=params_B.tolist(),
                 convergence=fit_B, information="B training feedback proxy; global parameters fitted on B calibration")

        # Prespecified secondary: source content-adjusted account residual.
        params_A, fit_A = fit_content(sides["A"][calibration], inventory["A"], item_accounts["A"], q["A"], age["A"][calibration], config)
        source_base = content_probability(params_A, inventory["A"], item_accounts["A"], q["A"], age["A"][train])
        expected_A = (source_base*sides["A"][train].sum(axis=1)[:,None]).sum(axis=0)
        residual_grid = np.asarray(config["secondary_residual_prior_mean_count_multipliers"])*expected_A.mean()
        residuals = (H_A[None,:]+residual_grid[:,None])/(expected_A[None,:]+residual_grid[:,None])
        base_cal = content_probability(params_A, inventory["B"], item_accounts["B"], q["B"], age["B"][calibration])
        base_test = content_probability(params_A, inventory["B"], item_accounts["B"], q["B"], age["B"][test])
        mapped_residuals = residuals[:, identity_maps]
        residual_cal = normalize(mapped_residuals[:,:,None,:]*base_cal[None,None,:,:])
        residual_test = normalize(mapped_residuals[:,:,None,:]*base_test[None,None,:,:])
        residual_p, residual_choice, residual_cal_loss, residual_daily = pick_candidates(residual_cal,residual_test,y_cal,y_test)
        register("A_fitted_content_base_on_B", base_test, loss(y_cal,base_cal), parameters=params_A.tolist(),
                 convergence=fit_A, information="A calibration global fit; A+B training feedback for content proxies")
        register("A_residual_transfer_secondary", residual_p[0], residual_cal_loss[0],
                 selected_prior_count=float(residual_grid[residual_choice[0]]),
                 selected_prior_mean_count_multiplier=float(config["secondary_residual_prior_mean_count_multipliers"][residual_choice[0]]),
                 information="Secondary B-informed content base times shrunk source A actual/expected count residual")
        all_permutation_daily["residual"].append(residual_daily)
        residual_rank = float((1+np.sum(residual_daily[1:].sum(axis=1)<=residual_daily[0].sum()))/(config["permutations"]+1))
        zero_accounts = {side:{segment:int(np.sum(sides[side][indices].sum(axis=0)==0))
                               for segment,indices in [("training",train),("calibration",calibration),("test",test)]}
                         for side in ["A","B"]}
        reports.append(dict(origin, models=models, zero_exposure_accounts_retained=zero_accounts,
                            B_test_exposures=int(y_test.sum()), raw_identity_reference_rank=raw_origin_rank,
                            secondary_residual_identity_reference_rank=residual_rank,
                            A_content_fit=fit_A, B_content_fit=fit_B))
        (args.output / "origins_progress.json").write_text(json.dumps(reports,indent=2)+"\n")
        progress("origin_complete", origin=name, models_completed=len(models), test_exposures=int(y_test.sum()),
                 raw_identity_reference_rank=raw_origin_rank, residual_reference_rank=residual_rank)

    daily_frame = pd.DataFrame(daily_records)
    daily_frame.to_csv(args.output / "daily_losses.csv", index=False)
    pd.DataFrame(prediction_records).to_parquet(args.output / "test_predictions.parquet",index=False)
    volume = daily_frame.loc[daily_frame.model.eq("A_transfer_history"),"B_exposures"].to_numpy(dtype=float)
    day_ids = daily_frame.loc[daily_frame.model.eq("A_transfer_history"),"date"].to_numpy()
    assert len(np.unique(day_ids)) == len(day_ids), "Test windows overlap"
    pooled, references = {}, {}
    for model, pieces in all_model_daily.items():
        pooled[model] = float(np.concatenate(pieces).sum()/volume.sum())
    permutation_rows = {"mapping":np.arange(config["permutations"]+1)}
    for kind,pieces in all_permutation_daily.items():
        values = np.concatenate(pieces,axis=1).sum(axis=1)/volume.sum()
        references[kind] = {"correct_mapping_test_log_loss":float(values[0]),
                            "permutation_test_loss_quantiles":{str(z):float(np.quantile(values[1:],z)) for z in [0,.025,.5,.975,1]},
                            "one_sided_conditional_reference_rank":float((1+np.sum(values[1:]<=values[0]))/len(values))}
        permutation_rows[kind+"_pooled_test_log_loss"] = values
    pd.DataFrame(permutation_rows).to_csv(args.output / "permutation_reference.csv",index=False)
    rng_boot = np.random.default_rng(config["bootstrap_seed"])
    resamples = rng_boot.integers(0,len(volume),size=(config["bootstrap_repetitions"],len(volume)))
    origin_samples = rng_boot.integers(0,len(reports),size=(config["bootstrap_repetitions"],len(reports)))
    comparisons = []
    for model,baseline in [("A_transfer_history","uniform_accounts"),("A_transfer_history","linear_quantity_B"),
                           ("A_transfer_history","B_own_history"),("A_residual_transfer_secondary","A_fitted_content_base_on_B")]:
        delta = np.concatenate(all_model_daily[model])-np.concatenate(all_model_daily[baseline])
        date_boot = delta[resamples].sum(axis=1)/volume[resamples].sum(axis=1)
        block_delta = np.asarray([x.sum() for x in np.split(delta,len(reports))])
        block_volume = np.asarray([x.sum() for x in np.split(volume,len(reports))])
        block_boot = block_delta[origin_samples].sum(axis=1)/block_volume[origin_samples].sum(axis=1)
        comparisons.append({"model":model,"baseline":baseline,"pooled_test_delta_log_loss":float(delta.sum()/volume.sum()),
                            "paired_date_block_bootstrap_95":np.quantile(date_boot,[.025,.975]).tolist(),
                            "origin_block_bootstrap_95":np.quantile(block_boot,[.025,.975]).tolist(),
                            "interpretation":"Negative means lower loss; temporal conditional uncertainty, not causal identification."})
    flow = flow_stock_audit(frame,origin_metadata,dates)
    (args.output / "flow_stock_audit.json").write_text(json.dumps(flow,indent=2)+"\n")
    result = {"study_id":config["study_id"],"started_utc":started,"completed_utc":datetime.now(timezone.utc).isoformat(),
              "source_sha256":source_sha,"protocol_sha256":sha256(args.protocol),
              "protocol":config,"sample":sample,"origins":reports,"pooled_test_log_losses":pooled,
              "primary_raw_identity_reference":references["raw"],"secondary_residual_identity_reference":references["residual"],
              "comparisons":comparisons,"unique_test_days":len(volume),"test_exposures":int(volume.sum()),
              "flow_stock_audit":flow,"all_selected_content_fits_converged":True,
              "no_test_based_selection":True,"no_missing_item_days_imputed_zero":True,
              "limitations":["Selected retrospective common-item cohort uses future row presence.",
                             "Only 46 multi-item accounts; most source accounts are excluded.",
                             "A raw transfer is coarse association, not source-content-adjusted.",
                             "Secondary demand proxy uses endogenous ordinary feedback, not intrinsic quality.",
                             "B self history is a richer-information memory benchmark.",
                             "Four rolling origins share accounts and some historical windows.",
                             "No economic owners, AI labels, cost shock, causal quality-versus-memory separation or equilibrium identified."]}
    (args.output / "cross_work_transfer.json").write_text(json.dumps(result,indent=2)+"\n")
    manifest = {p.name:{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in args.output.iterdir()
                if p.is_file() and p.name not in ["artifact_manifest.json","progress.jsonl"]}
    (args.output / "artifact_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    progress("complete", pooled_test_log_losses=pooled, primary_raw_reference=references["raw"],
             secondary_residual_reference=references["residual"], analysis_complete=True)


if __name__ == "__main__":
    main()

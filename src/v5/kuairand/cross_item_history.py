"""Cross-item, rolling-origin diagnostics of persistent account information.

No media, monthly statistics, economic-owner identification or causal parameters.
See PROTOCOL_ZH.md; all granular output must stay on remote storage.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize


WINDOWS = [(20220422, 20220426, 20220428, 20220501),
           (20220425, 20220429, 20220501, 20220504),
           (20220428, 20220502, 20220504, 20220507)]
SEEDS = list(range(20261105, 20261115))
SHRINKAGES = (1., 10., 100., 1000.)
POWERS = (0., .5, 1.)


def normalize_rows(x):
    return x / x.sum(axis=-1, keepdims=True)


def aggregate(values, mapping, n_authors):
    return np.asarray([np.bincount(mapping, weights=row, minlength=n_authors) for row in values])


def entropy(y, p):
    return float(-(y * np.log(np.maximum(p, 1e-300))).sum() / y.sum())


def demand_proxy(log, videos, hist_start, hist_end, valid_end):
    train = log.loc[log.date.between(hist_start, hist_end)].copy()
    valid = log.loc[log.date.gt(hist_end) & log.date.le(valid_end)]
    duration = train.groupby('video_id').duration_ms.median().reindex(videos.video_id)
    duration = duration.fillna(pd.Series(videos.video_duration.to_numpy(), index=videos.video_id))
    cuts = np.array([7000, 18000, 30000, 60000])
    train['bucket'] = np.digitize(train.duration_ms, cuts)
    global_mean = float(train.long_view.mean())
    prior_map = train.groupby('bucket').long_view.mean().to_dict()
    prior = np.array([prior_map.get(int(x), global_mean) for x in np.digitize(duration, cuts)])
    counts = train.groupby('video_id').long_view.agg(['sum', 'count']).reindex(videos.video_id, fill_value=0)
    validation_index = pd.Index(videos.video_id).get_indexer(valid.video_id)
    y = valid.long_view.to_numpy(dtype=float)
    choices = []
    for amount in (10., 50., 200.):
        q = (counts['sum'].to_numpy()+amount*prior)/(counts['count'].to_numpy()+amount)
        pred = np.clip(q[validation_index], 1e-6, 1-1e-6)
        loss = float(-(y*np.log(pred)+(1-y)*np.log(1-pred)).mean())
        choices.append((loss, amount, q))
    loss, amount, q = min(choices, key=lambda x:x[0])
    return np.clip(q, 1e-6, 1-1e-6), {'shrinkage':amount, 'random_validation_log_loss':loss,
             'random_history_rows':len(train), 'random_validation_rows':len(valid),
             'zero_random_history_videos':int(counts['count'].eq(0).sum())}


def make_permutations(mapping_a, mapping_b, videos_b, n_authors, seed, count):
    n_a = np.bincount(mapping_a, minlength=n_authors)
    n_b = np.bincount(mapping_b, minlength=n_authors)
    duration = pd.DataFrame({'author':mapping_b, 'duration':videos_b.video_duration.to_numpy(),
                            'upload':videos_b.upload_date.astype('int64').to_numpy()}).groupby('author').median().reindex(range(n_authors))
    duration_bin = pd.qcut(duration.duration, 3, labels=False, duplicates='drop').fillna(0).to_numpy()
    upload_bin = pd.qcut(duration.upload, 3, labels=False, duplicates='drop').fillna(0).to_numpy()
    keys = list(zip(np.minimum(n_a,3), np.minimum(n_b,3), duration_bin, upload_bin))
    groups = {}
    for author, key in enumerate(keys):
        groups.setdefault(key, []).append(author)
    rng = np.random.default_rng(seed)
    permutations = np.tile(np.arange(n_authors), (count,1))
    movable = 0
    for members in groups.values():
        if len(members)>1:
            movable += len(members)
        for p in range(count):
            permutations[p,members] = rng.permutation(members)
    return permutations, {'strata':len(groups), 'authors_in_nontrivial_strata':movable,
                          'movable_author_fraction':movable/n_authors}


def run_cell(seed, direction, window_index, source, target, data, permutations, n_permutations):
    dates, videos, y_video, q, author_of_video, n_authors = data
    start, end, valid_end, test_end = WINDOWS[window_index]
    hist = (dates>=start)&(dates<=end)
    valid = (dates>end)&(dates<=valid_end)
    test = (dates>valid_end)&(dates<=test_end)
    all_masks = (hist, valid, test)
    mapping_a, mapping_b = author_of_video[source], author_of_video[target]
    y_a, y_bv = y_video[:,source], y_video[:,target]
    y_aa, y_ba = aggregate(y_a,mapping_a,n_authors), aggregate(y_bv,mapping_b,n_authors)
    age = np.maximum((pd.to_datetime(pd.Series(dates).astype(str), format='%Y%m%d').to_numpy()[:,None]-videos.upload_date.to_numpy()[None,:])/np.timedelta64(1,'D'),0)
    log_q, log_age = np.log(q), np.log1p(age)
    counts_a, counts_b = np.bincount(mapping_a,minlength=n_authors), np.bincount(mapping_b,minlength=n_authors)

    def weights(ids, mapping, counts, params):
        alpha,beta,gamma = params
        logs = (alpha-1)*np.log(counts[mapping])[None,:]+beta*log_q[ids][None,:]-gamma*log_age[:,ids]
        return np.exp(logs-logs.max(axis=1,keepdims=True))

    def source_loss(params):
        p_video = normalize_rows(weights(source,mapping_a,counts_a,params))
        return entropy(y_aa[hist],aggregate(p_video,mapping_a,n_authors)[hist])

    def fit(linear):
        bounds = [(1.,1.) if linear else (.05,2.), (0.,4.), (0.,4.)]
        starts = [[1.,1.,.5],[1. if linear else .5,2.,2.]]
        fits = [minimize(source_loss,initial,method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':200,'ftol':1e-10}) for initial in starts]
        chosen = min(fits,key=lambda x:x.fun)
        return chosen.x, {'parameters':chosen.x.tolist(),'success':bool(chosen.success),
             'message':str(chosen.message),'training_loss':float(chosen.fun),
             'at_bound':[i for i,(v,(lo,hi)) in enumerate(zip(chosen.x,bounds)) if lo!=hi and min(abs(v-lo),abs(v-hi))<1e-5]}

    params, fit_audit = fit(False)
    linear_params, linear_fit = fit(True)
    wa = weights(source,mapping_a,counts_a,params)
    wb = weights(target,mapping_b,counts_b,params)
    p_a_video, p_b0_video = normalize_rows(wa), normalize_rows(wb)
    p_a_author, p_b0_author = aggregate(p_a_video,mapping_a,n_authors), aggregate(p_b0_video,mapping_b,n_authors)
    h_a = y_aa[hist].sum(axis=0)
    e_a = (p_a_author[hist]*y_a[hist].sum(axis=1)[:,None]).sum(axis=0)
    h_ba, h_bv = y_ba[hist].sum(axis=0), y_bv[hist].sum(axis=0)
    e_bv = (p_b0_video[hist]*y_bv[hist].sum(axis=1)[:,None]).sum(axis=0)
    e_ba = np.bincount(mapping_b,weights=e_bv,minlength=n_authors)
    linear_wb=weights(target,mapping_b,counts_b,linear_params)
    linear_pa=aggregate(normalize_rows(weights(source,mapping_a,counts_a,linear_params)),mapping_a,n_authors)
    linear_ea=(linear_pa[hist]*y_a[hist].sum(axis=1)[:,None]).sum(axis=0)
    predictions = {
        'uniform_author':normalize_rows(np.ones_like(wb)/counts_b[mapping_b][None,:]),
        'uniform_video_quantity':normalize_rows(np.ones_like(wb)),
        'linear_demand_age':normalize_rows(linear_wb),
        'source_fitted_quantity_demand_age':p_b0_video,
    }
    tuning = {}
    for name, observed, expected, account_signal, base_weights in [
        ('A_account_transfer',h_a,e_a,True,wb),
        ('A_account_transfer_linear_quantity',h_a,linear_ea,True,linear_wb),
        ('B_self_account_memory',h_ba,e_ba,True,wb),
        ('B_self_item_memory',h_bv,e_bv,False,wb)]:
        best = None
        for shrinkage in SHRINKAGES:
            signal = (observed+shrinkage)/(expected+shrinkage)
            if account_signal:
                signal = signal[mapping_b]
            for power in POWERS:
                p_video = normalize_rows(base_weights * signal[None,:]**power)
                loss = entropy(y_ba[valid],aggregate(p_video,mapping_b,n_authors)[valid])
                if best is None or loss<best[0]:
                    best = (loss,shrinkage,power,p_video)
        tuning[name] = {'shrinkage':best[1],'power':best[2],'validation_loss':best[0]}
        predictions[name] = best[3]

    # Every permuted mapping gets the identical global hyperparameter selection.
    best_null_validation = np.full(n_permutations,np.inf)
    selected_null_test = np.zeros(n_permutations)
    for shrinkage in SHRINKAGES:
        signals = ((h_a+shrinkage)/(e_a+shrinkage))[permutations]
        for power in POWERS:
            factor = signals**power
            logfactor = np.log(factor)
            losses=[]
            for mask in (valid,test):
                y = y_ba[mask]
                totals = y.sum(axis=1)
                norm = factor @ p_b0_author[mask].T
                loss = entropy(y,p_b0_author[mask]) - (logfactor @ y.sum(axis=0))/totals.sum() + (np.log(norm) @ totals)/totals.sum()
                losses.append(loss)
            improve = losses[0]<best_null_validation
            best_null_validation[improve]=losses[0][improve]
            selected_null_test[improve]=losses[1][improve]

    metrics, contributions, day_contributions = [], {}, {}
    for name,pv in predictions.items():
        pa = aggregate(pv,mapping_b,n_authors)
        losses = [entropy(y_ba[mask],pa[mask]) for mask in all_masks]
        y_test = y_ba[test]
        share = (pa[test]*y_test.sum(axis=1)[:,None]).sum(axis=0)/y_test.sum()
        true_share = y_test.sum(axis=0)/y_test.sum()
        metrics.append({'seed':seed,'direction':direction,'origin':window_index,'model':name,
                'history_author_loss':losses[0],'validation_author_loss':losses[1],
                'test_author_loss':losses[2],'test_video_loss':entropy(y_bv[test],pv[test]),
                'test_share_total_variation':float(.5*np.abs(share-true_share).sum()),
                'test_exposures':int(y_test.sum()),'tuning':tuning.get(name)})
        contributions[name] = -(y_test*np.log(np.maximum(pa[test],1e-300))).sum(axis=0)
        day_contributions[name] = -(y_test*np.log(np.maximum(pa[test],1e-300))).sum(axis=1)
    cell = {'seed':seed,'direction':direction,'origin':window_index,'dates':WINDOWS[window_index],
            'source_videos':len(source),'target_videos':len(target),
            'history_source_exposures':int(y_a[hist].sum()),'test_target_exposures':int(y_ba[test].sum()),
            'zero_source_history_authors':int(np.sum(h_a==0)),
            'zero_target_test_authors':int(np.sum(y_ba[test].sum(axis=0)==0)),
            'baseline_fit':fit_audit,'linear_fit':linear_fit,'tuning':tuning}
    return cell,metrics,contributions,y_ba[test].sum(axis=0),day_contributions,dates[test],selected_null_test


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=10,help='Protocol uses 10; smaller counts are smoke runs only.')
    parser.add_argument('--permutations',type=int,default=200)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    (args.root/'results').mkdir(exist_ok=True)
    started=time.monotonic()
    videos=pd.read_csv(args.data/'video_features_basic_pure.csv')
    videos['upload_date']=pd.to_datetime(videos.upload_dt)
    videos=videos.loc[videos.author_id.ge(0)&videos.upload_date.lt('2022-04-22')].sort_values('video_id').reset_index(drop=True)
    author_counts=videos.groupby('author_id').size()
    authors=author_counts.index[author_counts.ge(2)].to_numpy()
    n_authors=len(authors)
    author_of_video=pd.Index(authors).get_indexer(videos.author_id)
    eligible=np.flatnonzero(author_of_video>=0)
    standard=pd.read_csv(args.data/'log_standard_4_22_to_5_08_pure.csv',usecols=['date','video_id','tab','is_rand'])
    standard=standard.loc[standard.tab.eq(1)&standard.is_rand.eq(0)&standard.video_id.isin(videos.video_id)]
    random=pd.read_csv(args.data/'log_random_4_22_to_5_08_pure.csv',usecols=['date','video_id','tab','is_rand','long_view','duration_ms'])
    random=random.loc[random.tab.eq(1)&random.is_rand.eq(1)&random.video_id.isin(videos.video_id)]
    dates=np.sort(standard.date.unique())
    y_video=pd.crosstab(standard.date,standard.video_id).reindex(index=dates,columns=videos.video_id,fill_value=0).to_numpy(dtype=float)
    proxies=[demand_proxy(random,videos,*window[:3]) for window in WINDOWS]
    print(json.dumps({'stage':'loaded','pool_videos':len(videos),'eligible_videos':len(eligible),'authors':n_authors,'dates':dates.tolist()}),flush=True)
    cells,rows=[],[]
    totals,loss_contributions=np.zeros(n_authors),{}
    null_numerators=np.zeros(args.permutations)
    total_denominator=0.
    day_totals,day_losses={},{}
    for seed in SEEDS[:args.seeds]:
        rng=np.random.default_rng(seed)
        left,right=[],[]
        for author in range(n_authors):
            ids=rng.permutation(np.flatnonzero(author_of_video==author))
            half=len(ids)//2
            left.extend(ids[:half]);right.extend(ids[half:])
        for direction,source,target in [('A_to_B',np.array(left),np.array(right)),('B_to_A',np.array(right),np.array(left))]:
            permutations,permutation_audit=make_permutations(author_of_video[source],author_of_video[target],videos.iloc[target],n_authors,seed+(0 if direction=='A_to_B' else 1000),args.permutations)
            for origin,(q,q_audit) in enumerate(proxies):
                result=run_cell(seed,direction,origin,source,target,(dates,videos,y_video,q,author_of_video,n_authors),permutations,args.permutations)
                cell,metrics,contributions,denominator,daily,test_dates,null_losses=result
                cell['permutation_audit']=permutation_audit
                cells.append(cell);rows.extend(metrics)
                totals+=denominator
                total_denominator+=denominator.sum()
                null_numerators+=null_losses*denominator.sum()
                for name,value in contributions.items():
                    loss_contributions[name]=loss_contributions.get(name,np.zeros(n_authors))+value
                    for date,num,den in zip(test_dates,daily[name],y_video[np.isin(dates,test_dates)][:,target].sum(axis=1)):
                        day_losses.setdefault(name,{})[str(date)]=day_losses.get(name,{}).get(str(date),0.)+float(num)
                        if name=='uniform_author':
                            day_totals[str(date)]=day_totals.get(str(date),0.)+float(den)
                print(json.dumps({'stage':'cell_complete','seed':seed,'direction':direction,'origin':origin,
                       'test_exposures':int(denominator.sum()),'elapsed_seconds':round(time.monotonic()-started,2),
                       'losses':{x['model']:x['test_author_loss'] for x in metrics}}),flush=True)
    summary=[]
    base='source_fitted_quantity_demand_age'
    rng=np.random.default_rng(20261105)
    bootstrap_indices=rng.integers(0,n_authors,size=(1000,n_authors))
    for name,nums in loss_contributions.items():
        delta=nums-loss_contributions[base]
        boot=delta[bootstrap_indices].sum(axis=1)/totals[bootstrap_indices].sum(axis=1)
        model_rows=[row for row in rows if row['model']==name]
        summary.append({'model':name,'test_author_loss':float(nums.sum()/totals.sum()),
             'test_delta_vs_content_baseline':float(delta.sum()/totals.sum()),
             'conditional_account_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),
             'test_video_loss':float(sum(x['test_video_loss']*x['test_exposures'] for x in model_rows)/totals.sum()),
             'origin_losses':{str(o):float(sum(x['test_author_loss']*x['test_exposures'] for x in model_rows if x['origin']==o)/sum(x['test_exposures'] for x in model_rows if x['origin']==o)) for o in range(3)},
             'seed_losses':{str(seed):float(sum(x['test_author_loss']*x['test_exposures'] for x in model_rows if x['seed']==seed)/sum(x['test_exposures'] for x in model_rows if x['seed']==seed)) for seed in SEEDS[:args.seeds]}})
    null_losses=null_numerators/total_denominator
    true_transfer=next(x['test_author_loss'] for x in summary if x['model']=='A_account_transfer')
    comparisons=[]
    for model,reference in [('A_account_transfer','source_fitted_quantity_demand_age'),
                            ('A_account_transfer','A_account_transfer_linear_quantity'),
                            ('B_self_item_memory','A_account_transfer')]:
        delta=loss_contributions[model]-loss_contributions[reference]
        bootstrap=delta[bootstrap_indices].sum(axis=1)/totals[bootstrap_indices].sum(axis=1)
        comparisons.append({'model':model,'reference':reference,'test_delta_author_loss':float(delta.sum()/totals.sum()),
             'conditional_account_bootstrap_95_interval':np.quantile(bootstrap,[.025,.975]).tolist()})
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,
       'protocol':'PROTOCOL_ZH.md','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'sample':{'pool_videos':len(videos),'eligible_videos':len(eligible),'multi_item_authors':n_authors,
          'split_seeds':args.seeds,'directions':2,'origins':3,'unique_test_exposures':int(total_denominator/args.seeds),
          'exposure_evaluations_including_repeated_seed_predictions':int(total_denominator)},
       'random_proxy_audits':[x[1] for x in proxies],'models':summary,'comparisons':comparisons,
       'negative_control':{'permutations':args.permutations,'mean_test_loss':float(null_losses.mean()),
          'test_loss_quantiles':{str(q):float(np.quantile(null_losses,q)) for q in [0,.025,.5,.975,1]},
          'correct_mapping_loss':true_transfer,'fraction_null_loss_at_most_correct':float(np.mean(null_losses<=true_transfer)),
          'interpretation':'Matched observational negative controls; tail fraction is not a causal randomization p-value.'},
       'optimization':{'cells':len(cells),'failed_baseline_fits':sum(not c['baseline_fit']['success'] for c in cells),
            'failed_linear_fits':sum(not c['linear_fit']['success'] for c in cells),
            'at_bound_cells':sum(bool(c['baseline_fit']['at_bound']) for c in cells)},
       'scope':'Historical non-AI, selected old videos and accounts; cross-item association, not account ownership, cold-start, causal feedback or AI cost effects.'}
    (args.root/'results/summary.json').write_text(json.dumps(result,indent=2))
    (args.root/'results/cells.json').write_text(json.dumps(cells,indent=2))
    pd.DataFrame([{k:json.dumps(v) if isinstance(v,dict) else v for k,v in row.items()} for row in rows]).to_csv(args.root/'results/cell_metrics.csv',index=False)
    np.savez_compressed(args.root/'results/granular_account_contributions.npz',authors=authors,denominator=totals,**loss_contributions)
    (args.root/'results/daily_metrics.json').write_text(json.dumps({'totals':day_totals,'losses':day_losses},indent=2))
    print(json.dumps({'stage':'complete','summary':result}),flush=True)


if __name__=='__main__':
    main()

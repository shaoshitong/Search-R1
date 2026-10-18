"""Remote CPU-only analysis of saved pre-update probe predictions and own strict EM.

No rollout, GPU allocation, checkpoint load, or training modification. Row-level
predictions/labels stay on HPC3; only scalar aggregates may be copied locally.
"""
import argparse
import collections
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUN = 'cc0de51f2d68495c9664aa888074839c'
BASE = Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r2')
ORIGINAL = BASE.with_name('dynrpo-prefix5-thread5-20261017')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def entropy(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log(p) + (1-p) * np.log1p(-p))


def statistics(p, q):
    mu = p.mean(axis=1)
    var = p.var(axis=1)
    h = entropy(p).mean(axis=1)
    hmu = entropy(mu)
    # Average probability of opposite outcomes over the ten distinct pairs.
    pair = sum(p[:, i]*(1-p[:, j]) + p[:, j]*(1-p[:, i])
               for i in range(5) for j in range(i+1, 5)) / 10
    return dict(q=q, mean_p=mu, one_minus_mean_p=1-mu,
                mean_uncertainty=4*mu*(1-mu), mean_binary_entropy=h,
                entropy_of_mean=hmu, probability_variance=var,
                probability_std=np.sqrt(var), probability_range=np.ptp(p, axis=1),
                pair_disagreement=pair, entropy_gap=hmu-h,
                best_opposite_pair=p.max(axis=1)*(1-p.min(axis=1)),
                min_two_sided_confidence=np.minimum(p.max(axis=1), 1-p.min(axis=1)),
                mean_bernoulli_variance=(p*(1-p)).mean(axis=1),
                centrality=1-np.abs(2*mu-1),
                q_logit=np.log(np.clip(q, 1e-15, 1-1e-15) /
                               np.clip(1-q, 1e-15, 1-1e-15)))


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-14 and np.std(y) > 0 else None


def metric(y, s, probability=False):
    order = np.argsort(-s, kind='stable')[:512]
    result = dict(n=len(y), mixed=int(y.sum()), prevalence=float(y.mean()),
                  pearson=corr(s, y),
                  spearman=float(spearmanr(s, y).statistic) if np.std(s)>1e-14 else None,
                  auroc=float(roc_auc_score(y, s)),
                  average_precision=float(average_precision_score(y, s)),
                  top512_precision=float(y[order].mean()),
                  mean_score=float(s.mean()), score_std=float(s.std()))
    if probability:
        result.update(brier=float(brier_score_loss(y, s)),
                      log_loss=float(log_loss(y, np.clip(s, 1e-12, 1-1e-12), labels=[0, 1])))
    return result


def load_outer(n, periodic):
    root = (ORIGINAL if n <= 9 else BASE) / 'outers' / ('outer-%06d' % n)
    score_dir = root/'probe-score'
    manifest = read(score_dir/'manifest.json')
    assert manifest['outer'] == n and manifest['run_id'] == RUN
    if periodic:
        score_path = score_dir/'candidate-scores.jsonl'
        raw = score_path.read_bytes()
        assert sha(raw) == manifest['candidate_scores_sha256']
        groups = [json.loads(line) for line in raw.splitlines() if line]
        complete = read(root/'full-pool/complete.json')
    else:
        score_path = score_dir/'selection.json'
        raw = score_path.read_bytes()
        selection = json.loads(raw)
        assert not selection['outcome_labels_used'] and sha(raw) == manifest['selection_sha256']
        groups = selection['selected']
        complete = read(root/'full/complete.json')
    evidence = complete['own_em']
    raw_em = Path(evidence['path']).read_bytes()
    assert len(raw_em) == evidence['bytes'] and sha(raw_em) == evidence['sha256']
    labels = {}
    for line in raw_em.splitlines():
        if not line:
            continue
        row = json.loads(line)
        tid = row['trajectory_id']
        assert tid not in labels and row['strict_em'] in (0, 1)
        assert row['actor_version'] == manifest['actor_version']
        labels[tid] = row
    assert len({str(g['qid']) for g in groups}) == len(groups)
    p = np.array([g['trajectory_probabilities'] for g in groups], dtype=float)
    q = np.array([g['q'] for g in groups], dtype=float)
    ys = np.array([[labels[t]['strict_em'] for t in g['trajectory_ids']] for g in groups], dtype=int)
    assert len(labels) == 5*len(groups)
    assert set(labels) == {t for g in groups for t in g['trajectory_ids']}
    assert all(str(labels[t]['qid']) == str(g['qid']) for g in groups for t in g['trajectory_ids'])
    assert p.shape == ys.shape and p.shape[1] == 5 and np.isfinite(p).all()
    assert np.max(np.abs(1-p.prod(axis=1)-(1-p).prod(axis=1)-q)) < 1e-10
    k = ys.sum(axis=1)
    y = ((k > 0) & (k < 5)).astype(int)
    ids = np.array([str(g['qid']) for g in groups])
    return dict(outer=n, p=p, q=q, ys=ys, y=y, ids=ids, stats=statistics(p, q),
                source=dict(outer=n, scope='all_candidates' if periodic else 'selected512',
                            score_path=str(score_path), score_sha256=sha(raw),
                            em_path=evidence['path'], em_sha256=evidence['sha256'],
                            probe_before_sha256=manifest['probe_before_checkpoint_sha256'],
                            actor_version=manifest['actor_version'], groups=len(groups)))


def bins(y, score, count=10):
    return [dict(n=len(ix), mean_prediction=float(score[ix].mean()),
                 observed_mixed=float(y[ix].mean()),
                 score_min=float(score[ix].min()), score_max=float(score[ix].max()))
            for ix in np.array_split(np.argsort(score, kind='stable'), count)]


def paired_bootstrap(data, challenger, repetitions=1000):
    y = np.concatenate([r['y'] for r in data])
    q = np.concatenate([r['stats']['q'] for r in data])
    s = np.concatenate([r['stats'][challenger] for r in data])
    ids = np.concatenate([r['ids'] for r in data])
    unique, inverse = np.unique(ids, return_inverse=True)
    rng = np.random.default_rng(20261018)
    auc, precision = [], []
    lengths = [len(r['y']) for r in data]
    offsets = np.cumsum([0] + lengths)
    orders = [[np.argsort(-v, kind='stable') for v in (r['stats']['q'], r['stats'][challenger])]
              for r in data]
    for _ in range(repetitions):
        weights = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[inverse]
        auc.append(float(roc_auc_score(y, s, sample_weight=weights)-roc_auc_score(y, q, sample_weight=weights)))
        vals = []
        for i, r in enumerate(data):
            w = weights[offsets[i]:offsets[i+1]]
            rates = []
            for order in orders[i]:
                ww = w[order]; before = np.cumsum(ww)-ww
                used = np.minimum(ww, np.maximum(0, 512-before))
                rates.append(float(np.dot(used, r['y'][order])/used.sum()))
            vals.append(rates[1]-rates[0])
        precision.append(float(np.mean(vals)))
    return dict(unit='qid cluster, shared across held-out outers; seed20261018', repetitions=repetitions,
                pooled_auroc_delta_ci95=np.quantile(auc, [.025, .975]).tolist(),
                macro_top512_precision_delta_ci95=np.quantile(precision, [.025, .975]).tolist())


def run(out, through):
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'analysis.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_sha = sha(Path(__file__).read_bytes())
    result_path = out/'summary.json'
    if result_path.exists():
        prior = read(result_path)
        assert prior['analysis_source_sha256'] == source_sha
        print(json.dumps(dict(status='EXISTING_VERIFIED_ANALYSIS', path=str(result_path), sha256=sha(result_path.read_bytes()))))
        return
    started = time.time()
    write(out/'progress.json', dict(stage='LOAD_ALL_CANDIDATES', started=started))
    data = [load_outer(n, True) for n in range(10, through+1, 10)]
    keys = list(data[0]['stats'])
    per_round = []
    for r in data:
        for key in keys:
            per_round.append(dict(outer=r['outer'], statistic=key,
                                  **metric(r['y'], r['stats'][key], key=='q')))
    early = [r for r in data if r['outer'] <= 70]
    validation = [r for r in data if r['outer'] in (80, 90)]
    test = [r for r in data if r['outer'] in (100, 110)]
    assert len(early)==7 and len(validation)==len(test)==2

    def aggregate(subset, key):
        mm = [metric(r['y'], r['stats'][key]) for r in subset]
        yy = np.concatenate([r['y'] for r in subset]); ss = np.concatenate([r['stats'][key] for r in subset])
        centered_s = np.concatenate([(r['stats'][key]-r['stats'][key].mean())/max(r['stats'][key].std(), 1e-12) for r in subset])
        centered_y = np.concatenate([r['y']-r['y'].mean() for r in subset])
        return dict(statistic=key, pooled_pearson=corr(ss, yy),
                    within_round_standardized_pearson=corr(centered_s, centered_y),
                    macro_auroc=float(np.mean([m['auroc'] for m in mm])),
                    macro_average_precision=float(np.mean([m['average_precision'] for m in mm])),
                    macro_top512_precision=float(np.mean([m['top512_precision'] for m in mm])),
                    pooled_auroc=float(roc_auc_score(yy, ss)))

    training_stats = [aggregate(early, key) for key in keys]
    # Preselect the one hand statistic using early rounds only, before test evaluation.
    best = max(training_stats, key=lambda x:x['macro_auroc'])['statistic']
    all_stats = [aggregate(data, key) for key in keys]
    validation_stats = [aggregate(validation, key) for key in keys]
    test_stats = [aggregate(test, key) for key in keys]
    write(out/'progress.json', dict(stage='TIME_HELD_OUT_MODELS_AND_QID_CLUSTER_BOOTSTRAP', best_early_statistic=best))
    features = ['q', 'mean_p', 'mean_uncertainty', 'probability_variance',
                'probability_range', 'mean_binary_entropy', 'entropy_gap']
    models = []
    holdout_ids = set(np.concatenate([r['ids'] for r in validation+test]))
    for name, ff in [('calibrated_q', ['q']), ('multistat_logistic', features)]:
        x = np.concatenate([np.column_stack([r['stats'][k] for k in ff]) for r in early])
        y = np.concatenate([r['y'] for r in early]); ids = np.concatenate([r['ids'] for r in early])
        keep = np.array([qid not in holdout_ids for qid in ids])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=20261018))
        model.fit(x[keep], y[keep])
        for r in validation+test:
            prediction = model.predict_proba(np.column_stack([r['stats'][k] for k in ff]))[:, 1]
            r['stats'][name] = prediction
            models.append(dict(model=name, outer=r['outer'], train_rows=int(keep.sum()),
                               excluded_overlap_rows=int((~keep).sum()), features=ff,
                               **metric(r['y'], prediction, True)))
    boot = {key:paired_bootstrap(test, key) for key in {best, 'multistat_logistic', 'calibrated_q'} if key!='q'}
    diagnostics = []
    for r in data:
        k = r['ys'].sum(axis=1)
        top = np.argsort(-r['q'], kind='stable')[:512]
        diagnostics.append(dict(outer=r['outer'], actual_mixed=float(r['y'].mean()),
            mean_q=float(r['q'].mean()), q_std=float(r['q'].std()),
            mean_p=float(r['p'].mean()), actual_trajectory_em=float(r['ys'].mean()),
            identical_five_p_fraction=float((np.ptp(r['p'], axis=1)<1e-10).mean()),
            top512_mean_q=float(r['q'][top].mean()), top512_actual_mixed=float(r['y'][top].mean()),
            top512_identical_five_p_fraction=float((np.ptp(r['p'][top],axis=1)<1e-10).mean()),
            K_histogram={str(i):int((k==i).sum()) for i in range(6)},
            q_calibration_bins=bins(r['y'], r['q'])))
    write(out/'progress.json', dict(stage='SELECTED_ONLY_SECONDARY_ANALYSIS'))
    secondary = []
    for n in range(1, min(read(BASE/'latest-joint.json')['outer'], 119)+1):
        if n % 10 == 0:
            continue
        r = load_outer(n, False)
        secondary.append(dict(outer=n, n=512, actual_mixed=float(r['y'].mean()),
                              statistics={key:metric(r['y'], r['stats'][key], key=='q')
                                          for key in ['q', best, 'probability_variance', 'mean_uncertainty']},
                              source=r['source']))
    # All row-level predictions and labels remain exclusively on remote shared storage.
    with gzip.open(out/'periodic-all-candidate-records.csv.gz', 'wt', newline='') as handle:
        fields=['outer','qid','K','mixed']+[f'p{j}' for j in range(5)]+[f'em{j}' for j in range(5)]+['q']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in data:
            for i in range(len(r['y'])):
                writer.writerow(dict(outer=r['outer'],qid=r['ids'][i],K=int(r['ys'][i].sum()),mixed=int(r['y'][i]),
                                     q=r['q'][i],**{f'p{j}':r['p'][i,j] for j in range(5)},
                                     **{f'em{j}':int(r['ys'][i,j]) for j in range(5)}))
    summary=dict(status='SAVED_PROBE_P_Q_MIXED_ANALYSIS_COMPLETE', epoch=time.time(),
        elapsed_seconds=time.time()-started, analysis_source_sha256=source_sha,
        run_id=RUN, snapshots='All saved predictions are pre-update; no current/post-fit re-inference',
        primary_scope='Complete periodic candidate pools; ordinary selected512 are secondary, selection-biased',
        primary_outers=[r['outer'] for r in data], primary_groups=sum(len(r['y']) for r in data),
        primary_trajectories=sum(r['ys'].size for r in data), sources=[r['source'] for r in data],
        definitions=dict(label='0<sum(strict_EM[0:5])<5',q='1-prod(p)-prod(1-p)',
                         correlation='Pearson against binary Mixed (point-biserial); Spearman also supplied',
                         precision='top512 per outer, descending score, original candidate order breaks ties',
                         fitting='Train10..70; validation80,90; untouched time test100,110; learned models exclude all validation/test qids from training',
                         scope_limit='Offline observational comparison; does not measure new-policy training benefit or prove why independence fails'),
        statistics_all=all_stats, statistics_train=training_stats,
        statistics_validation=validation_stats, statistics_test=test_stats,
        selected_on_train_only=best, paired_qid_cluster_bootstrap_test=boot,
        learned_model_results=models, per_round=per_round, diagnostics=diagnostics,
        selected_only_secondary=secondary, training_modified=False, gpu_used=False,
        row_level_remote_path=str(out/'periodic-all-candidate-records.csv.gz'))
    write(result_path,summary)
    write(out/'progress.json',dict(stage='COMPLETE',summary_sha256=sha(result_path.read_bytes()),elapsed_seconds=time.time()-started))
    print(json.dumps(dict(status=summary['status'],path=str(result_path),sha256=sha(result_path.read_bytes()),
                          primary_groups=summary['primary_groups'],early_selected_statistic=best,
                          elapsed_seconds=summary['elapsed_seconds'])))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--through',type=int,default=110)
    args=parser.parse_args()
    target=Path(args.output)
    if not target.is_absolute() or not str(target).startswith(str(BASE/'analysis')+'/') or target.resolve()!=target:
        raise ValueError('Analysis outputs must stay inside this owned HPC3 namespace')
    if args.through != 110:
        raise ValueError('Frozen analysis split is10..70 /80..90 /100..110')
    run(target,args.through)

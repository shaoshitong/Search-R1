"""CPU analysis of already-viewed scores; no model inference or final4096 access."""
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

A = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
FULL = A / 'online-packed-full169028-M3to10-P20-20261022-v1'
OUT = A / 'viewed2048-full169028-packed-comparison-20261023-v1'
E1_SHA = '3d832d00b2d40eaf5ffad4568c1785c84270b0a81e033c57e6c7030a93cd0de8'
SEED, REPEATS = 2026102301, 2000


def read(p):
    return json.loads(Path(p).read_bytes())


def proof(p):
    p = Path(p)
    data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def rows(p):
    with gzip.open(p, 'rt') as f:
        return sorted((json.loads(line) for line in f), key=lambda r: r['qid'])


def data(rr):
    assert len(rr) == 2048 and len({r['qid'] for r in rr}) == 2048
    ids = [(r['qid'], r['trajectory_ids']) for r in rr]
    ys = np.array([r.get('labels', r.get('strict_em')) for r in rr], dtype=int)
    assert ys.shape == (2048, 5) and np.isin(ys, [0, 1]).all()
    return ids, ys


def rank(v):
    order = np.argsort(v, kind='stable')
    starts = np.r_[0, np.flatnonzero(v[order][1:] != v[order][:-1]) + 1]
    ends = np.r_[starts[1:], len(v)]
    result = np.empty(len(v), dtype=float)
    for start, end in zip(starts, ends):
        result[order[start:end]] = (start + end - 1) / 2
    return result


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None


def metric(ys, score, threshold, variance):
    k = ys.sum(1)
    y = (k > 0) & (k < 5)
    pred = score >= threshold
    tp, tn = int((pred & y).sum()), int((~pred & ~y).sum())
    top = np.argsort(-score, kind='stable')[:512]
    auc = (rank(score)[y].sum() - y.sum() * (y.sum() - 1) / 2) / (y.sum() * (~y).sum())
    result = dict(questions=2048, Mixed_questions=int(y.sum()),
                  Mixed_balanced_accuracy=.5 * (tp / y.sum() + tn / (~y).sum()),
                  Mixed_recall=tp / int(y.sum()), nonMixed_FPR=1 - tn / int((~y).sum()),
                  Mixed_ROC_AUC=float(auc), threshold=threshold,
                  Top512=dict(Mixed=int(y[top].sum()), fraction=float(y[top].mean())),
                  score_mean=float(score.mean()), score_std=float(score.std()),
                  score_quantiles=np.quantile(score, [0, .1, .25, .5, .75, .9, 1]).tolist(),
                  classwise={})
    target_v = (k / 5) * (1 - k / 5)
    for name, mask in [('all_wrong', k == 0), ('Mixed', y), ('all_correct', k == 5)]:
        value = dict(questions=int(mask.sum()), mean_score=float(score[mask].mean()),
                     predicted_Mixed_fraction=float(pred[mask].mean()),
                     error_rate=float((pred[mask] != y[mask]).mean()))
        if variance:
            value.update(variance_MSE=float(((score[mask] - target_v[mask]) ** 2).mean()),
                         variance_MAE=float(np.abs(score[mask] - target_v[mask]).mean()))
        result['classwise'][name] = value
    if variance:
        result.update(score_semantics='predicted_population_variance_not_Mixed_probability',
                      Mixed_BCE=None, variance_MSE=float(((score - target_v) ** 2).mean()),
                      variance_MAE=float(np.abs(score - target_v).mean()),
                      variance_Pearson=corr(score, target_v), variance_Spearman=corr(rank(score), rank(target_v)),
                      mean_only_oracle_MSE=float(((target_v - target_v.mean()) ** 2).mean()),
                      constant_zero_MAE=float(target_v.mean()))
    else:
        q = np.clip(score, 1e-12, 1 - 1e-12)
        result['Mixed_BCE'] = float(np.where(y, -np.log(q), -np.log1p(-q)).mean())
    return result


def run():
    assert os.uname().sysname == 'Linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.time()
    epath = A / 'joint-mixed-direct-official-169028-b128-v1/evaluations/epoch-01/complete.json'
    assert proof(epath)['sha256'] == E1_SHA
    e1 = read(epath)
    summaries, arrays, thresholds, sources, observed = {}, {}, {}, {'E1_receipt': proof(epath)}, {}
    refids = refys = None
    for state in ['raw', 'EMA']:
        ref = e1['metrics'][state]['fixed_selection2048']['predictions']
        assert proof(ref['path']) == {k: ref[k] for k in ['path', 'bytes', 'sha256']}
        rr = rows(ref['path'])
        ids, ys = data(rr)
        if refids is None:
            refids, refys = ids, ys
        assert ids == refids and np.array_equal(ys, refys)
        name = 'E1_' + state
        score = np.array([r['predicted_mixed'] for r in rr])
        summaries[name] = metric(ys, score, .5, False)
        arrays[name], thresholds[name], sources[name] = score, .5, ref
    pilot = A / 'online-mixed-layer-prefix-20261022-2gpu-v1/layer21-prefix80/epoch-03-fixed_validation2048-predictions.jsonl.gz'
    rr = rows(pilot)
    ids, ys = data(rr)
    assert ids == refids and np.array_equal(ys, refys)
    for state in ['raw', 'EMA']:
        score = 1 / (1 + np.exp(-np.array([r[state + '_logit'] for r in rr])))
        name = 'L21P80_E3_' + state
        summaries[name] = metric(ys, score, .5, False)
        arrays[name], thresholds[name], sources[name] = score, .5, proof(pilot)
    curve = {}
    for loss in ['mse', 'mae', 'z_mse_anchor']:
        base = FULL / loss
        curves = {}
        for ev in sorted(base.glob('evaluation-epoch-*.json')):
            evaluation = read(ev)
            epoch = evaluation['epoch']
            roles = {}
            for role in ['fixed_train2048_G5', 'fixed_validation2048_G5']:
                path = base / ('epoch-%02d-%s-predictions.jsonl.gz' % (epoch, role))
                rr = rows(path)
                ids, ys = data(rr)
                if role == 'fixed_validation2048_G5':
                    assert ids == refids and np.array_equal(ys, refys)
                roles[role] = {}
                for state in ['raw', 'EMA']:
                    score = np.array([r[state + '_predicted_variance'] for r in rr])
                    assert np.isfinite(score).all() and (score >= 0).all() and (score <= .25).all()
                    value = metric(ys, score, .08, True)
                    original = evaluation['metrics'][role][state]
                    assert abs(value['variance_MSE'] - original['variance_MSE']) < 1e-10
                    assert value['Top512']['Mixed'] == original['Top512']['Mixed']
                    roles[role][state] = value
                    if role == 'fixed_validation2048_G5' and epoch > 0:
                        name = f'{loss}_E{epoch}_{state}'
                        summaries[name], arrays[name], thresholds[name] = value, score, .08
                sources[f'{loss}_E{epoch}_{role}'] = proof(path)
            curves[str(epoch)] = dict(cursor=evaluation['cursor'], roles=roles, evaluation=proof(ev))
        curve[loss] = curves
        observed[loss] = dict(progress=read(base / 'progress-rank-0.json'),
                              complete=read(base / 'complete.json') if (base / 'complete.json').exists() else None)
    # Analyze each arm's latest finished epoch plus the historical references.
    selected = ['E1_raw', 'E1_EMA', 'L21P80_E3_raw', 'L21P80_E3_EMA']
    for loss, values in curve.items():
        epoch = max(map(int, values))
        if epoch > 0:
            selected.extend(f'{loss}_E{epoch}_{state}' for state in ['raw', 'EMA'])
    scores = np.column_stack([arrays[name] for name in selected])
    thr = np.array([thresholds[name] for name in selected])
    y = (refys.sum(1) > 0) & (refys.sum(1) < 5)
    rng = np.random.default_rng(SEED)
    ba, top = [], []
    for _ in range(REPEATS):
        ix = rng.integers(0, 2048, 2048)
        yy, zz = y[ix], scores[ix]
        pp = zz >= thr
        ba.append(.5 * (pp[yy].mean(0) + (~pp[~yy]).mean(0)))
        order = np.argsort(-zz, axis=0, kind='stable')[:512]
        top.append(np.take_along_axis(np.broadcast_to(yy[:, None], zz.shape), order, axis=0).mean(0))
    ba, top = np.array(ba), np.array(top)
    comparisons = {}
    for name in selected:
        if name.startswith(('E1_', 'L21P80_')):
            continue
        c = selected.index(name)
        for baseline in ['E1_EMA', 'L21P80_E3_raw']:
            b = selected.index(baseline)
            comparisons[f'{name}_minus_{baseline}'] = dict(
                candidate=name, baseline=baseline,
                BA_delta=summaries[name]['Mixed_balanced_accuracy'] - summaries[baseline]['Mixed_balanced_accuracy'],
                BA_percentile95=np.quantile(ba[:, c] - ba[:, b], [.025, .975]).tolist(),
                Top512_delta=summaries[name]['Top512']['fraction'] - summaries[baseline]['Top512']['fraction'],
                Top512_percentile95=np.quantile(top[:, c] - top[:, b], [.025, .975]).tolist())
    result = dict(status='VIEWED2048_PACKED_FULL_ACTUAL_ANALYSIS_COMPLETE', time=time.time(),
                  observed=observed, curves=curve, independent_summaries=summaries,
                  latest_epoch_paired_comparisons=comparisons, source_proofs=sources,
                  identical_question_own_G5_EM=True,
                  bootstrap=dict(repeats=REPEATS, seed=SEED, unit='question',
                                 limitation='Exploratory viewed pool only; not training-seed uncertainty or final confirmation.'),
                  candidate_frozen=False, threshold_tuned=False, final4096_read=False,
                  no_model_tensor_or_hidden_read=True, no_new_inference=True, seconds=time.time() - started)
    OUT.mkdir(exist_ok=True)
    dest = OUT / ('analysis-%d.json' % int(result['time']))
    raw = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
    dest.write_bytes(raw + b'\n')
    (OUT / 'latest.json').write_bytes(raw + b'\n')
    print(json.dumps(dict(receipt=proof(dest), result=result), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    run()

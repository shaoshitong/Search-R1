"""Read-only, CPU-only train-fitted ablation of sealed five-probability scores.

No checkpoint/features/GPU are loaded. External diagnostic and selection labels
are used only after the scorer, regularization, and threshold have been frozen.
The output is a small JSON summary/coefficient artifact, never raw predictions.
Run with the already installed NumPy environment and OPENBLAS_NUM_THREADS=2.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path):
    return dict(path=str(path), bytes=Path(path).stat().st_size, sha256=sha(path))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(str(temporary), str(path))


def group_q(p):
    p = np.asarray(p, dtype=np.float64)
    return 1. - p.prod(axis=1) - (1. - p).prod(axis=1)


def feature_matrix(p):
    """Permutation-invariant probabilities only; no labels/IDs in features."""
    p = np.sort(np.asarray(p, dtype=np.float64), axis=1)
    assert p.ndim == 2 and p.shape[1] == 5
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    clipped = np.clip(p, 1e-7, 1. - 1e-7)
    z = np.log(clipped) - np.log1p(-clipped)
    mean = p.mean(axis=1)
    mean_clip = np.clip(mean, 1e-7, 1. - 1e-7)
    entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log1p(-clipped))
    uncertainty = p * (1 - p)
    log_a = np.log(clipped).sum(axis=1)
    log_b = np.log1p(-clipped).sum(axis=1)
    columns = [p[:, i] for i in range(5)] + [z[:, i] for i in range(5)]
    names = ['sorted_p_%d' % i for i in range(5)] + ['sorted_logit_%d' % i for i in range(5)]
    additional = [
        ('mean_p', mean), ('range_p', np.ptp(p, axis=1)),
        ('variance_p', p.var(axis=1)), ('mean_entropy_p', entropy.mean(axis=1)),
        ('entropy_mean_p', -(mean_clip * np.log(mean_clip) + (1 - mean_clip) * np.log1p(-mean_clip))),
        ('mean_p_one_minus_p', uncertainty.mean(axis=1)),
        ('minimum_p_one_minus_p', uncertainty.min(axis=1)),
        ('maximum_p_one_minus_p', uncertainty.max(axis=1)),
        ('q_independent_mixed', group_q(p)),
        ('prob_all_correct', p.prod(axis=1)), ('prob_all_wrong', (1-p).prod(axis=1)),
        ('absolute_mean_minus_half', abs(mean-.5)),
        ('mean_squared_distance_to_half', ((p-.5)**2).mean(axis=1)),
        ('log_prob_all_correct', log_a), ('log_prob_all_wrong', log_b),
        ('log_prob_nonmixed', np.logaddexp(log_a, log_b)),
        ('mean_logit', z.mean(axis=1)), ('range_logit', np.ptp(z, axis=1)),
        ('variance_logit', z.var(axis=1)), ('mean_absolute_logit', abs(z).mean(axis=1)),
    ]
    for name, values in additional:
        names.append(name)
        columns.append(values)
    matrix = np.column_stack(columns)
    assert np.isfinite(matrix).all()
    return matrix, names


def balanced_cohort(groups, seed, epoch, budget):
    """Match balanced_sampling.plan exactly, without importing GPU modules."""
    positive = [i for i, g in enumerate(groups) if 0 < sum(g['labels']) < 5]
    negative = [i for i, g in enumerate(groups) if sum(g['labels']) in (0, 5)]
    assert budget % 2 == 0 and min(len(positive), len(negative)) >= budget // 2
    rng = random.Random(seed + epoch * 100003)
    rng.shuffle(positive)
    rng.shuffle(negative)
    ids = [x for pair in zip(positive[:budget//2], negative[:budget//2]) for x in pair]
    batches = [ids[i:i+51] for i in range(0, len(ids), 51)]
    for batch in batches:
        rng.shuffle(batch)
    selected = [groups[i] for batch in batches for i in batch]
    return selected, hashlib.sha256(json.dumps(batches, separators=(',', ':')).encode()).hexdigest()


def validate_truth(groups, expected):
    assert len(groups) == expected
    result = {}
    trajectory_ids = set()
    for g in groups:
        qid = g['qid']
        assert qid not in result and len(g['labels']) == len(g['trajectory_ids']) == 5
        assert all(y in (0, 1) for y in g['labels'])
        assert len(set(g['trajectory_ids'])) == 5
        assert not trajectory_ids.intersection(g['trajectory_ids'])
        trajectory_ids.update(g['trajectory_ids'])
        result[qid] = g
    return result


def checked_rows(paths, truth, epoch, split, fingerprints):
    result, seen, seen_tids = [], set(), set()
    for path in paths:
        before = fingerprint(path)
        fingerprints.append(before)
        with gzip.open(str(path), 'rt') as f:
            for line in f:
                r = json.loads(line)
                qid = r['qid']
                assert qid not in seen and qid in truth
                g = truth[qid]
                assert r['trajectory_ids'] == g['trajectory_ids']
                assert not seen_tids.intersection(r['trajectory_ids'])
                seen_tids.update(r['trajectory_ids'])
                p = r['p_j']
                assert len(p) == 5 and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
                q = r['predicted_mixed']
                assert math.isfinite(q) and 0 <= q <= 1
                tolerance = 1e-12 if split == 'selection_validation' else 2e-6
                assert abs(float(group_q([p])[0]) - q) < tolerance
                if split != 'selection_validation':
                    assert r['epoch'] == epoch and r['split'] == split and r['weight_kind'] == 'EMA0.99'
                    assert r['strict_em'] == g['labels'] and r['true_mixed'] == (0 < sum(g['labels']) < 5)
                else:
                    assert set(r) == {'qid', 'trajectory_ids', 'p_j', 'predicted_mixed'}
                result.append(dict(qid=qid, p=p, q=q, k=sum(g['labels']), labels=g['labels']))
                seen.add(qid)
        assert sha(path) == before['sha256'], 'Prediction changed while read'
    assert seen == set(truth)
    return sorted(result, key=lambda r: r['qid'])


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def fit_logistic(x, y, penalty, max_iterations=25):
    """Mean logistic loss + lambda/2 ||w||^2; intercept not regularized."""
    design = np.column_stack([np.ones(len(x)), x])
    regularizer = np.full(design.shape[1], penalty)
    regularizer[0] = 0.
    coefficient = np.zeros(design.shape[1])
    coefficient[0] = math.log(y.mean()/(1-y.mean()))

    def objective(w):
        z = design.dot(w)
        return float((np.logaddexp(0., z) - y*z).mean() + .5 * (regularizer*w*w).sum())

    accepted = 0
    for iteration in range(max_iterations):
        p = sigmoid(design.dot(coefficient))
        gradient = design.T.dot(p-y)/len(y) + regularizer*coefficient
        hessian = design.T.dot(design*(p*(1-p))[:, None])/len(y)
        hessian.flat[::len(coefficient)+1] += regularizer + 1e-10
        delta = np.linalg.solve(hessian, gradient)
        old = objective(coefficient)
        scale = 1.
        while scale > 2**-18:
            trial = coefficient-scale*delta
            if objective(trial) <= old - 1e-4*scale*float(gradient.dot(delta)):
                coefficient = trial
                accepted += 1
                break
            scale *= .5
        else:
            break
        if np.max(np.abs(scale*delta)) < 1e-7:
            break
    assert np.isfinite(coefficient).all()
    return coefficient, dict(iterations=iteration+1, accepted_newton_steps=accepted,
                             objective=objective(coefficient), maximum_absolute_gradient=float(np.max(abs(gradient))))


def auc_ap(y, score):
    y, score = np.asarray(y, dtype=np.int64), np.asarray(score, dtype=np.float64)
    assert len(y) == len(score) and np.isfinite(score).all() and 0 < y.sum() < len(y)
    order = np.argsort(score, kind='mergesort')
    s, target = score[order], y[order]
    boundaries = np.r_[0, np.flatnonzero(s[1:] != s[:-1])+1, len(s)]
    rank_positive = 0.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rank_positive += target[start:end].sum() * ((start+1+end)/2.)
    positive, negative = int(y.sum()), int(len(y)-y.sum())
    auc = (rank_positive-positive*(positive+1)/2.)/(positive*negative)
    # Tie-block average precision is invariant to qid/order within equal scores.
    found = total = 0
    ap = 0.
    for start, end in zip(boundaries[-2::-1], boundaries[:0:-1]):
        block_positive = int(target[start:end].sum())
        found += block_positive
        total += end-start
        ap += block_positive/positive * found/total
    return dict(roc_auc=float(auc), average_precision=float(ap))


def choose_threshold(y, score):
    """Use only the internal held-out training questions, with deterministic ties."""
    order = np.argsort(-score, kind='mergesort')
    s, target = score[order], y[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1])+1, len(s)]
    tp = np.cumsum(target)[ends-1]
    fp = ends-tp
    balanced = .5*(tp/y.sum() + 1-fp/(len(y)-y.sum()))
    # Include the legal all-negative classifier (BA=.5).
    candidates = [(float(b), float(s[end-1])) for b, end in zip(balanced, ends)]
    candidates.append((.5, float(np.nextafter(s[0], np.inf))))
    best = max(candidates, key=lambda item: (item[0], -abs(item[1]-.5), item[1]))
    return best[1], best[0]


def metrics(rows, scores, threshold):
    """Labels enter assessment only; never the score calculation/sorting."""
    y = np.asarray([int(0 < r['k'] < 5) for r in rows])
    predicted = scores >= threshold
    tp = int((predicted & (y == 1)).sum())
    fn = int((~predicted & (y == 1)).sum())
    tn = int((~predicted & (y == 0)).sum())
    fp = int((predicted & (y == 0)).sum())
    ranking = sorted(range(len(rows)), key=lambda i: (-float(scores[i]), rows[i]['qid']))
    top = ranking[:min(512, len(rows))]
    histogram = [sum(rows[i]['k'] == k for i in top) for k in range(6)]
    per_class = {}
    for name, mask in [('all_wrong', np.array([r['k'] == 0 for r in rows])),
                       ('mixed', y == 1), ('all_correct', np.array([r['k'] == 5 for r in rows]))]:
        values = scores[mask]
        per_class[name] = dict(n=int(mask.sum()), mean_score=float(values.mean()),
            score_quantiles=np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist(),
            predicted_mixed_fraction=float((values >= threshold).mean()))
    result = dict(questions=len(rows), true_mixed=int(y.sum()), true_mixed_fraction=float(y.mean()),
        threshold=float(threshold), confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn),
        tpr=tp/(tp+fn), tnr=tn/(tn+fp), fpr=fp/(tn+fp), fnr=fn/(tp+fn),
        mixed_balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),
        predicted_mixed=int(predicted.sum()), classwise=per_class,
        top512=dict(questions=len(top), K_histogram=histogram, mixed=sum(histogram[1:5]),
                    mixed_fraction=sum(histogram[1:5])/len(top)))
    result.update(auc_ap(y, scores))
    clipped = np.clip(scores, 1e-12, 1-1e-12)
    result.update(mean_score=float(scores.mean()), brier=float(((scores-y)**2).mean()),
                  mixed_bce=float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()))
    return result


def verify_sealed_metrics(rows, sealed):
    score = np.asarray([r['q'] for r in rows])
    result = metrics(rows, score, .5)
    assert result['questions'] == sealed['groups']
    assert sealed['trajectories'] == 5*len(rows)
    assert result['confusion'] == sealed['confusion']
    assert abs(result['mixed_balanced_accuracy']-sealed['mixed_balanced_accuracy']) < 1e-12
    assert abs(float(score.mean())-sealed['mean_predicted_mixed']) < 1e-6
    y = np.asarray([0 < r['k'] < 5 for r in rows])
    wrong_saturation = int(((score == 0) & y).sum()+((score == 1) & ~y).sum())
    if not wrong_saturation:
        ce = -np.log(np.where(y, score, 1-score)).mean()
        assert abs(float(ce)-sealed['group_bce']) < 1e-4
    else:
        ce = None
    return dict(confusion_exact=True, group_bce_from_rounded_q=None if ce is None else float(ce),
                rounded_wrong_saturations=wrong_saturation)


def analyze(base, epoch, penalties, output):
    started = time.monotonic()
    base = Path(base)
    output = Path(output) if output else base/'analysis'/('epoch-%04d-group-score-ablation.json' % epoch)
    assert str(output.resolve()).startswith(str(base.resolve())+'/')
    progress_path = output.with_name(output.stem+'-progress.json')
    def progress(stage, **values):
        atomic(progress_path, dict(stage=stage, elapsed_seconds=time.monotonic()-started,
                                   time=time.time(), pid=os.getpid(), epoch=epoch, **values))
    progress('VALIDATING_SEALED_INPUTS')
    config_path, receipt_path, bank_path = base/'config.json', base/'probe'/('epoch-%04d.json' % epoch), base/'bank/groups.json'
    config, receipt, bank = read(config_path), read(receipt_path), read(bank_path)
    assert receipt['epoch'] == epoch and receipt['completed_epoch'] and receipt['cursor'] == 0
    assert receipt['independent_cpu_reload'] and receipt['actor_updates'] == 0
    assert sha(bank_path) == config['mixed_bank_sha256']
    cohort, plan_sha = balanced_cohort(bank['train'], config['seed'], epoch, config['fit_epoch_groups'])
    assert plan_sha == receipt['metrics']['sampling_plan']['sha256']
    train_truth = validate_truth(cohort, config['fit_epoch_groups'])
    diagnostic_truth = validate_truth(bank['diagnostic'], 512)
    held = Path(config['selection_base'])
    held_path = held/'bank/groups.json'
    held_complete = read(held/'bank/complete.json')
    assert sha(held_path) == held_complete['groups_sha256']
    selection_truth = validate_truth(read(held_path)['selection_validation'], 2048)
    sets = [set(train_truth), set(diagnostic_truth), set(selection_truth)]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i))
    # Also rule out leakage through a same-qid question outside this cohort.
    assert set(g['qid'] for g in bank['train']).isdisjoint(selection_truth)
    files = [fingerprint(p) for p in [config_path, receipt_path, bank_path, held_path, held/'bank/complete.json']]
    prediction_root = base/'probe/predictions'
    train_rows = checked_rows([prediction_root/('epoch-%04d-train-rank%d.jsonl.gz' % (epoch, rank)) for rank in range(4)], train_truth, epoch, 'train', files)
    diagnostic_rows = checked_rows([prediction_root/('epoch-%04d-diagnostic-rank%d.jsonl.gz' % (epoch, rank)) for rank in range(4)], diagnostic_truth, epoch, 'diagnostic', files)
    selection_metric = receipt['metrics']['selection2048']
    selection_path = Path(selection_metric['predictions']['path'])
    assert sha(selection_path) == selection_metric['predictions']['sha256']
    selection_rows = checked_rows([selection_path], selection_truth, epoch, 'selection_validation', files)
    sealed_checks = dict(train=verify_sealed_metrics(train_rows, receipt['metrics']['train']),
                         diagnostic=verify_sealed_metrics(diagnostic_rows, receipt['metrics']['diagnostic']))
    selected_q = metrics(selection_rows, np.array([r['q'] for r in selection_rows]), .5)
    assert selected_q['top512']['K_histogram'] == selection_metric['selected512']['K_histogram']
    pool_hist = [sum(r['k'] == k for r in selection_rows) for k in range(6)]
    assert pool_hist == selection_metric['pool']['K_histogram']
    assert sum(0 < r['k'] < 5 for r in train_rows)*2 == len(train_rows)
    progress('INPUTS_VALIDATED', train_questions=len(train_rows), diagnostic_questions=len(diagnostic_rows),
             selection_questions=len(selection_rows))

    x, names = feature_matrix([r['p'] for r in train_rows])
    y = np.asarray([0 < r['k'] < 5 for r in train_rows], dtype=np.float64)
    # qids choose the split only and are never exposed as fitting/scoring features.
    internal = np.array([int(hashlib.sha256(('group-score-ablation-v1:'+str(r['qid'])).encode()).hexdigest(), 16) % 5 == 0 for r in train_rows])
    fitting = ~internal
    assert min(y[fitting].sum(), (1-y[fitting]).sum(), y[internal].sum(), (1-y[internal]).sum()) > 0
    mean = x[fitting].mean(axis=0)
    scale = x[fitting].std(axis=0)
    scale[scale < 1e-10] = 1.
    standardized = (x-mean)/scale
    candidates = []
    chosen = None
    fitting_started = time.monotonic()
    for penalty in penalties:
        coefficient, fit = fit_logistic(standardized[fitting], y[fitting], penalty)
        internal_score = sigmoid(coefficient[0]+standardized[internal].dot(coefficient[1:]))
        threshold, balanced = choose_threshold(y[internal], internal_score)
        ranking = auc_ap(y[internal], internal_score)
        item = dict(penalty=penalty, internal_mixed_balanced_accuracy=balanced,
                    threshold=threshold, fit=fit, **ranking)
        candidates.append(item)
        # AUC is primary to match Top512 ranking, BA secondary, stronger L2 wins ties.
        key = (ranking['roc_auc'], balanced, penalty)
        if chosen is None or key > chosen[0]:
            chosen = (key, coefficient.copy(), threshold, penalty)
        progress('FITTING_SYMMETRIC_SCORER', completed_penalties=len(candidates), total_penalties=len(penalties),
                 last_candidate=item)
    _, coefficient, threshold, penalty = chosen
    # Freeze before using ANY external labels in assessment; no external tuning.
    scorer = dict(feature_names=names, feature_mean=mean.tolist(), feature_scale=scale.tolist(),
                  intercept=float(coefficient[0]), coefficients=coefficient[1:].tolist(),
                  threshold=float(threshold), l2_penalty=penalty,
                  hyperparameter_rule='training internal80/20 only; maximum AUC, then maximum BA, then stronger L2',
                  refit_after_selection=False, external_labels_used_in_fit_or_threshold=False)

    # Explicit monotonic q-only calibration. With a single standardized q
    # feature, a negative unconstrained optimum means the constrained optimum
    # at nonnegative slope is the constant class-prior model (convex objective).
    q_train = np.asarray([r['q'] for r in train_rows])
    q_mean, q_scale = float(q_train[fitting].mean()), float(q_train[fitting].std())
    if q_scale < 1e-10:
        q_scale = 1.
    q_standardized = ((q_train-q_mean)/q_scale)[:, None]
    q_coefficient, q_fit = fit_logistic(q_standardized[fitting], y[fitting], .001)
    constrained_boundary = bool(q_coefficient[1] < 0)
    if constrained_boundary:
        q_coefficient[:] = [math.log(y[fitting].mean()/(1-y[fitting].mean())), 0.]
    q_calibrated_internal = sigmoid(q_coefficient[0]+q_standardized[internal].dot(q_coefficient[1:]))
    q_calibrated_threshold, _ = choose_threshold(y[internal], q_calibrated_internal)
    q_calibrator = dict(feature='raw q standardized with fit80 moments', mean=q_mean, scale=q_scale,
                        intercept=float(q_coefficient[0]), nonnegative_slope=float(q_coefficient[1]),
                        l2_penalty=.001, threshold=q_calibrated_threshold, constrained_boundary=constrained_boundary,
                        internal_only=True, fit=q_fit,
                        monotonic_calibration_cannot_improve_top512_ranking=True)
    progress('SCORERS_FROZEN_BEFORE_EXTERNAL_ASSESSMENT', l2_penalty=penalty,
             threshold=threshold, monotonic_q_calibration=q_calibrator)

    def score_probabilities(probabilities):
        values, own_names = feature_matrix(probabilities)
        assert own_names == names
        return sigmoid(coefficient[0]+((values-mean)/scale).dot(coefficient[1:]))

    analysis = {}
    for label, rows in [('train_fit80', [r for r, flag in zip(train_rows, fitting) if flag]),
                        ('train_internal20', [r for r, flag in zip(train_rows, internal) if flag]),
                        ('fixed512', diagnostic_rows), ('independent2048', selection_rows)]:
        predicted = score_probabilities([r['p'] for r in rows])
        baseline = np.asarray([r['q'] for r in rows])
        calibrated_q = sigmoid(q_coefficient[0]+((baseline-q_mean)/q_scale)*q_coefficient[1])
        analysis[label] = dict(q_baseline=metrics(rows, baseline, .5),
                              monotonic_q_only_calibration=metrics(rows, calibrated_q, q_calibrated_threshold),
                              train_fitted_logistic=metrics(rows, predicted, threshold))
        if q_coefficient[1] > 0:
            assert analysis[label]['q_baseline']['top512'] == analysis[label]['monotonic_q_only_calibration']['top512']
            assert abs(analysis[label]['q_baseline']['roc_auc']-analysis[label]['monotonic_q_only_calibration']['roc_auc']) < 1e-12
    # A threshold tuned on the same training-only internal split is separately
    # useful to distinguish calibration effects from irreparable ordering.
    q_threshold, _ = choose_threshold(y[internal], np.array([r['q'] for r, flag in zip(train_rows, internal) if flag]))
    for label, rows in [('fixed512', diagnostic_rows), ('independent2048', selection_rows)]:
        analysis[label]['q_train_calibrated_threshold'] = metrics(rows, np.asarray([r['q'] for r in rows]), q_threshold)
    for manifest in files:
        assert sha(manifest['path']) == manifest['sha256'], 'Input changed during analysis'
    result = dict(status='READ_ONLY_TRAIN_FITTED_GROUP_SCORE_ABLATION', base=str(base), epoch=epoch,
        checkpoint_sha256=receipt['sha256'], steps=receipt['steps'],
        source_file_sha256=sha(__file__), time=time.time(), elapsed_seconds=time.monotonic()-started,
        fitting_seconds=time.monotonic()-fitting_started,
        restrictions=dict(checkpoint_loaded=False, training_updated=False, inference_deployed=False,
                          raw_predictions_saved_locally=False, diagnostic_or_2048_tuning=False,
                          model_input='symmetric functions of existing five trajectory probabilities only'),
        checks=dict(train_questions=len(train_rows), diagnostic_questions=len(diagnostic_rows),
                    selection_questions=len(selection_rows), train_plan_sha256=plan_sha,
                    disjoint_qids=True, unique_G5_tids=True, exact_labels_and_trajectory_order=True,
                    q_formula_checked=True, sealed_metrics_recomputed=sealed_checks,
                    selection_prediction_sha_presealed=True,
                    train_and_diagnostic_hash_note='Epoch receipt has no train/diagnostic per-file hash; hashes below are fresh, stability-checked, and labels/counts/confusion/CE match sealed metrics. They are not falsely described as presealed.',
                    all_fingerprints=files),
        split=dict(method='sha256(group-score-ablation-v1:+str(qid)) modulo5; residue0 internal heldout',
                   fit_questions=int(fitting.sum()), internal_questions=int(internal.sum()),
                   fit_mixed=int(y[fitting].sum()), internal_mixed=int(y[internal].sum()),
                   fit_qid_sha256=hashlib.sha256(json.dumps([r['qid'] for r, flag in zip(train_rows, fitting) if flag],separators=(',',':')).encode()).hexdigest(),
                   internal_qid_sha256=hashlib.sha256(json.dumps([r['qid'] for r, flag in zip(train_rows, internal) if flag],separators=(',',':')).encode()).hexdigest(),
                   heldout_scope='Only held out from this calibration fit; original probe saw training questions'),
        candidates=candidates, frozen_scorer=scorer, frozen_monotonic_q_calibrator=q_calibrator, results=analysis,
        interpretation='A positive result demonstrates exploitable p_j signal with a separate train-fitted group scorer; it does not repair the underlying trajectory probe or prove online policy gains.')
    # Prevent accidental raw-data writes anywhere else: only this small summary.
    atomic(output, result)
    progress('COMPLETE', output=str(output), output_sha256=sha(output))
    print(json.dumps(dict(status=result['status'], output=str(output), output_sha256=sha(output),
                          elapsed_seconds=result['elapsed_seconds'], fitting_seconds=result['fitting_seconds'],
                          candidates=candidates, split=result['split'], frozen_hyperparameters=dict(l2_penalty=penalty, threshold=threshold),
                          results=analysis, training_updated=False, inference_deployed=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--epoch', default=2, type=int)
    parser.add_argument('--penalties', default='0.0001,0.001,0.01,0.1,1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    analyze(args.base, args.epoch, [float(v) for v in args.penalties.split(',')], args.output)

"""Bounded CPU2, train-only two-parameter G10 calibration experiment.

Only one internally qualified frozen candidate can open the predeclared
external prediction file. No actor/probe/Adam updates, final-pool access, or
external threshold tuning. Learned scalars and full rows remain remote.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import resource
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from .g5_event_scores_dp import group_event_scores_dp
from .group_score_ablation import metrics as grouped_metrics

OBJECTIVES = ('trajectory', 'trajectory+three-event')
REGULARIZERS = (0., .01, .1)
LIMIT_SECONDS = 26 * 60
RSS_LIMIT = 3 * 1024 ** 3
MAX_ITER = 40
MAX_EVAL = 60
SEED = 2718
PROBABILITY_CLIP = 1e-6
COMBINATIONS = tuple(itertools.combinations(range(10), 5))


def metric(scores, counts, qids, threshold, top_k=None):
    """Reuse the already verified lightweight own-EM evaluator, without feature imports."""
    if top_k not in (None, min(512, len(qids))):
        raise ValueError('Predeclared Top512 budget required')
    r = grouped_metrics([dict(qid=q, k=int(k)) for q, k in zip(qids, counts)], scores, threshold)
    return dict(balanced_accuracy=r['mixed_balanced_accuracy'], Mixed_recall=r['tpr'],
        nonMixed_recall=r['tnr'], allwrong_FPR=r['classwise']['all_wrong']['predicted_mixed_fraction'],
        allcorrect_FPR=r['classwise']['all_correct']['predicted_mixed_fraction'],
        AUC=r['roc_auc'], confusion=r['confusion'], questions=r['questions'],
        fixed_threshold=float(threshold), classwise=r['classwise'],
        **{'Top%d' % min(512, len(qids)): dict(Mixed=r['top512']['mixed'],
            Mixed_fraction=r['top512']['mixed_fraction'], K_histogram=r['top512']['K_histogram'],
            questions=r['top512']['questions'])})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def checked(ref):
    p = Path(ref['path'])
    if p.stat().st_size != ref['bytes'] or sha(p) != ref['sha256']:
        raise ValueError('Input SHA/size mismatch: ' + str(p))
    return p


def atomic(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(t, p)


def fingerprint(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def same_state(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


def question_split(qids, budget=2048):
    """Frozen identity-only exact holdout, matching the external Top512/2048 quota."""
    if len(set(qids)) != len(qids) or not 0 < budget < len(qids):
        raise ValueError('Distinct question identities and a proper heldout budget required')
    selected = set(sorted(qids, key=lambda q: (
        hashlib.sha256(('platt-qid-%d:' % SEED + q).encode()).hexdigest(), q))[:budget])
    return np.asarray([q in selected for q in qids])


def fixed_indices(qid):
    seed = int(hashlib.sha256(('platt-G5-%d:' % SEED + qid).encode()).hexdigest()[:16], 16)
    return sorted(random.Random(seed).sample(range(10), 5))


def parameters(raw):
    log_scale = 3. * raw[0].tanh()
    bias = 10. * raw[1].tanh()
    return log_scale.exp(), bias, log_scale


def calibrated(logits, raw):
    a, b, _ = parameters(raw)
    return a * logits + b


def probability_logits(probabilities):
    """Identical declared observable interface for scored train and old external rows."""
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Finite probabilities in[0,1] required')
    p = np.clip(p, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return np.log(p) - np.log1p(-p)


def raw_probability_summary(probabilities, labels, qids):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != np.asarray(labels).shape or p.ndim != 2 or p.shape[1] != 5:
        raise ValueError('Five actual probabilities and labels per original question required')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Finite raw probabilities required')
    q = np.clip(1 - p.prod(1) - (1 - p).prod(1), 0, 1)
    return metric(q, np.asarray(labels).sum(1), qids, .5, min(512, len(qids)))


def chunk_objective(logits, labels, raw, objective):
    if logits.ndim != 2 or logits.shape != labels.shape or logits.shape[1] != 10:
        raise ValueError('Authentic logits/labels[Q,10] required')
    if objective not in OBJECTIVES or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('Declared objective and strict own EM required')
    z = calibrated(logits, raw)
    result = F.binary_cross_entropy_with_logits(z, labels, reduction='mean')
    if objective == OBJECTIVES[1]:
        ix = torch.tensor(COMBINATIONS, dtype=torch.long, device=z.device)
        view_z = z[:, ix].reshape(-1, 5)
        k = labels[:, ix].sum(-1).reshape(-1)
        scores = group_event_scores_dp(view_z)
        result = result - torch.where(k == 0, scores['log_allwrong'],
            torch.where(k == 5, scores['log_allcorrect'], scores['log_mixed'])).mean()
    return result


def fit(logits, labels, objective, regularizer, progress, max_iter=MAX_ITER, max_eval=MAX_EVAL, checkpoint_callback=None):
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    raw = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=1., max_iter=max_iter, max_eval=max_eval,
                                tolerance_grad=1e-8, tolerance_change=1e-10,
                                line_search_fn='strong_wolfe')
    calls = 0; losses = []

    def closure():
        nonlocal calls
        optimizer.zero_grad(); total = torch.zeros((), dtype=torch.float64)
        for start in range(0, len(z), 128):
            n = min(128, len(z) - start)
            part = chunk_objective(z[start:start+n], y[start:start+n], raw, objective) * n / len(z)
            part.backward(); total += part.detach()
        _, bias, log_scale = parameters(raw)
        reg = float(regularizer) * (log_scale.square() + bias.square())
        reg.backward(); total += reg.detach(); calls += 1
        losses.append(float(total))
        if not bool(torch.isfinite(total)) or not bool(torch.isfinite(raw.grad).all()):
            raise ValueError('Nonfinite calibration objective/gradient')
        progress(calls, float(total))
        return total

    optimizer.step(closure)
    with torch.no_grad():
        accepted = 0.
        for start in range(0, len(z), 128):
            n = min(128, len(z) - start)
            accepted += float(chunk_objective(z[start:start+n], y[start:start+n], raw, objective)) * n / len(z)
        _, bias, log_scale = parameters(raw)
        accepted += float(regularizer) * float(log_scale.square() + bias.square())
    report = dict(closure_calls=calls, initial_loss=losses[0], last_closure_loss=losses[-1],
                  accepted_final_regularized_loss=accepted, closure_loss_curve=losses,
                  optimizer_iterations=int(optimizer.state[raw].get('n_iter', 0)))
    if checkpoint_callback is not None:
        report['full_checkpoint'] = checkpoint_callback(dict(raw=raw.detach().clone(),
            optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            objective=objective, regularizer=float(regularizer),
            calibration_phase='ALL_TRAIN_LBFGS_UPDATES_COMPLETE_BEFORE_HOLDOUT_ASSESSMENT',
            fit=report.copy()))
    return raw.detach(), report


def read_scored(ref, truth):
    p = checked(ref); rows = []; probabilities = []
    with gzip.open(p, 'rt') as f:
        for line in f:
            r = json.loads(line); qid = str(r['qid'])
            if len(rows) >= len(truth) or qid != str(truth[len(rows)]['qid']):
                raise ValueError('Scored question order differs from accepted cohort')
            g = truth[len(rows)]
            if r['trajectory_ids'] != g['trajectory_ids'] or r['strict_em'] != g['labels']:
                raise ValueError('Own source G10 IDs/strict EM mismatch')
            z, p_j = np.asarray(r['logits'], dtype=np.float64), np.asarray(r['p_j'], dtype=np.float64)
            if z.shape != (10,) or p_j.shape != (10,) or not np.isfinite(z).all() or not np.isfinite(p_j).all():
                raise ValueError('Finite ten genuine source probabilities/logits required')
            if np.max(np.abs(torch.tensor(z).sigmoid().numpy() - p_j)) > 1e-6:
                raise ValueError('p_j does not match source logits')
            if 'question_hash' in r and r['question_hash'] != g['question_hash']:
                raise ValueError('Scored question hash mismatch')
            rows.append(probability_logits(p_j)); probabilities.append(p_j)
    if len(rows) != len(truth): raise ValueError('Incomplete scored G10 bank')
    return dict(logits=np.asarray(rows), probabilities=np.asarray(probabilities))


def score(logits, raw):
    with torch.no_grad():
        z = calibrated(torch.as_tensor(logits, dtype=torch.float64), raw)
        return group_event_scores_dp(z)['mixed_probability'].numpy()


def summary(logits, labels, qids, raw):
    scores = score(logits, raw)
    return metric(scores, np.asarray(labels).sum(1), qids, .5, min(512, len(qids)))


def objective_rank(metrics):
    return metrics['balanced_accuracy'] + metrics['Top512']['Mixed'] / 512.


def run(config, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / 'calibration.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'complete.json').exists(): raise RuntimeError('Existing completed calibration; refusing overwrite')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    start = time.time(); stop = threading.Event()

    def guard():
        while not stop.wait(.25):
            rss = int(next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1]) * 1024
            if rss > RSS_LIMIT or time.time() - start > LIMIT_SECONDS:
                atomic(out / 'failure.json', dict(status='CPU_CALIBRATION_RESOURCE_LIMIT', rss=rss, elapsed=time.time()-start))
                os._exit(74)

    threading.Thread(target=guard, daemon=True).start()
    try:
        atomic(out / 'predeclared-protocol.json', dict(
            config=config, objectives=OBJECTIVES, regularizers=REGULARIZERS,
            probability_clip=PROBABILITY_CLIP, max_LBFGS_iterations=MAX_ITER,
            LBFGS_max_eval_argument=MAX_EVAL, CPU_threads=2, GPUs=0,
            fixed_decision_threshold=.5, all_G5_views=252, source_size=10,
            whole_question_split_seed=SEED, internal_holdout_questions=2048, winner_count=1,
            ranking_rule='Internal joint-improvement candidates, largest BA+Top512 fraction, lexical id tie-break',
            parameterization='a=exp(3*tanh(u)), b=10*tanh(v)',
            input_interface='canonical inverse-logit clipped stored p_j for both train and external',
            native_probe_has_seen_training_cohort=True, external_opened=False,
            declared_epoch=time.time()))
        atomic(out / 'controller.json', dict(pid=os.getpid(), start_epoch=start, CPU_threads=2, GPUs=0,
               start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
               config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()))
        groups = json.loads(checked(config['groups']).read_text())
        if isinstance(groups, dict): groups = groups['groups']
        if len(groups) != 8192 or len({str(g['qid']) for g in groups}) != len(groups):
            raise ValueError('Exactly frozen8192 unique accepted questions required')
        qids = [str(g['qid']) for g in groups]
        if len({t for g in groups for t in g['trajectory_ids']}) != 81920:
            raise ValueError('Distinct authentic ten-trajectory IDs required')
        if any(len(g['trajectory_ids']) != 10 or len(g['labels']) != 10
               or any(type(v) is not int or v not in (0, 1) for v in g['labels']) for g in groups):
            raise ValueError('Ten strict own integer EM labels/IDs per question required')
        y = np.asarray([g['labels'] for g in groups], dtype=np.float64)
        if y.shape != (8192, 10) or not np.isin(y, [0, 1]).all(): raise ValueError('Strict G10 labels')
        mask = question_split(qids); ix = np.asarray([fixed_indices(q) for q in qids])
        if int(mask.sum()) != 2048: raise ValueError('Exact2048 whole-qid holdout required')
        hold_qids = [q for q, h in zip(qids, mask) if h]
        y5 = np.take_along_axis(y, ix, axis=1)[mask]
        atomic(out / 'isolation.json', dict(
            fitting_questions=int((~mask).sum()), parameter_holdout_questions=int(mask.sum()),
            fitting_qids_sha256=hashlib.sha256(json.dumps([q for q, h in zip(qids, mask) if not h]).encode()).hexdigest(),
            holdout_qids_sha256=hashlib.sha256(json.dumps(hold_qids).encode()).hexdigest(),
            fixed_G5_indices_sha256=hashlib.sha256(ix.astype('<i8').tobytes()).hexdigest(),
            split_before_labels=True, whole_qid_group_views_isolated=True,
            native_probe_has_seen_training_cohort=True, external_opened=False))
        summaries = []; candidates = []
        for source in config['sources']:
            observed = read_scored(source['predictions'], groups); z = observed['logits']
            z5 = np.take_along_axis(z, ix, axis=1)[mask]
            baseline = raw_probability_summary(np.take_along_axis(observed['probabilities'], ix, axis=1)[mask], y5, hold_qids)
            for objective in OBJECTIVES:
                for regularizer in REGULARIZERS:
                    identifier = '%s--%s--%g' % (source['name'], objective, regularizer)

                    def progress(calls, loss):
                        atomic(out / 'progress.json', dict(candidate=identifier, closure_calls=calls,
                               regularized_training_loss=loss, elapsed=time.time()-start, external_opened=False))

                    def checkpoint(state):
                        state.update(source=source, groups=config['groups'],
                                     isolation=fingerprint(out / 'isolation.json'),
                                     protocol=fingerprint(out / 'predeclared-protocol.json'))
                        path=out / 'checkpoints' / (identifier + '.pt')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary=path.with_suffix('.pt.tmp');torch.save(state, temporary);os.replace(temporary,path)
                        loaded=torch.load(path,map_location='cpu',weights_only=False)
                        if not same_state(state, loaded):
                            raise ValueError('Full scalar LBFGS/RNG checkpoint failed immediate load verification')
                        restored_raw=loaded['raw'].clone().requires_grad_()
                        restored=torch.optim.LBFGS([restored_raw]);restored.load_state_dict(loaded['optimizer'])
                        if restored.state[restored_raw].get('n_iter') != state['fit']['optimizer_iterations']:
                            raise ValueError('LBFGS optimizer iteration cursor not restorable')
                        receipt=dict(fingerprint(path), raw_LBFGS_RNG_loaded=True,
                                     optimizer_iterations=state['fit']['optimizer_iterations'])
                        atomic(path.with_suffix('.receipt.json'),receipt)
                        return receipt

                    raw, fitted = fit(z[~mask], y[~mask], objective, regularizer, progress,
                                      checkpoint_callback=checkpoint)
                    measured = summary(z5, y5, hold_qids, raw)
                    qualified = (measured['balanced_accuracy'] > baseline['balanced_accuracy']
                                 and measured['Top512']['Mixed'] > baseline['Top512']['Mixed'])
                    item = dict(id=identifier, source=source['name'], objective=objective, regularizer=regularizer,
                                internal=measured, uncalibrated_internal=baseline, fit=fitted,
                                internal_joint_improvement=qualified)
                    summaries.append(item); atomic(out / 'internal-candidates.json', summaries)
                    if qualified: candidates.append((objective_rank(measured), identifier, raw, item, source))
        if not candidates:
            atomic(out / 'complete.json', dict(status='NO_INTERNAL_JOINT_IMPROVEMENT_NO_EXTERNAL_SCORING',
                   internal_candidates=summaries, external_opened=False, final_pool_read=False, elapsed=time.time()-start))
            return
        candidates.sort(key=lambda row: (-row[0], row[1])); _, identifier, raw, item, source = candidates[0]
        # Learned scalars are a remote model artifact, never returned in summaries.
        a, b, _ = parameters(raw)
        model_path = out / 'frozen-calibration-model.json'
        atomic(model_path, dict(a=float(a), b=float(b), raw=raw.tolist(), id=identifier))
        freeze = dict(status='ONE_INTERNAL_WINNER_FROZEN_BEFORE_EXTERNAL_OPEN', winner=item,
                      model=fingerprint(model_path), frozen_epoch=time.time(), threshold=.5,
                      external_opened=False, final_pool_read=False)
        atomic(out / 'candidate-freeze.json', freeze)
        external = []
        with gzip.open(checked(source['external_predictions']), 'rt') as f:
            external = [json.loads(line) for line in f]
        if len(external) != 2048 or len({str(r['qid']) for r in external}) != 2048:
            raise ValueError('Exact frozen external2048 required')
        if set(qids) & {str(r['qid']) for r in external}: raise ValueError('External/train question overlap')
        if any(len(r['p_j']) != 5 or len(r['strict_em']) != 5 or len(r['trajectory_ids']) != 5
               or any(type(v) is not int or v not in (0, 1) for v in r['strict_em']) for r in external):
            raise ValueError('External realG5 required')
        tids = [t for r in external for t in r['trajectory_ids']]
        if len(set(tids)) != 10240 or set(tids) & {t for g in groups for t in g['trajectory_ids']}:
            raise ValueError('External own G5 trajectory collision')
        p = np.asarray([r['p_j'] for r in external], dtype=np.float64)
        ez = probability_logits(p)
        eq = [str(r['qid']) for r in external]; ey = np.asarray([r['strict_em'] for r in external])
        measured = summary(ez, ey, eq, raw)
        baseline = raw_probability_summary(p, ey, eq)
        atomic(out / 'complete.json', dict(status='FROZEN_TWO_PARAMETER_CALIBRATION_EXTERNAL_COMPLETE',
               winner=item, frozen_candidate=fingerprint(out / 'candidate-freeze.json'),
               external=measured, uncalibrated_same_model_external=baseline,
               external_opened=True, external_scored_models=1, external_threshold_fitted=False,
               probability_clip=PROBABILITY_CLIP, uncalibrated_baseline_probability_clipped=False,
               final_pool_read=False, elapsed=time.time()-start,
               peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               internal_candidates=summaries))
    except BaseException as exc:
        atomic(out / 'failure.json', dict(type=type(exc).__name__, error=str(exc), elapsed=time.time()-start))
        raise
    finally:
        stop.set(); lock.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('out', type=Path)
    a = p.parse_args(); run(json.loads(a.config.read_text()), a.out)


if __name__ == '__main__': main()

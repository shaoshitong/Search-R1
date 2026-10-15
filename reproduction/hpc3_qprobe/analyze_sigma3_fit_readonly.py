"""CPU diagnostics from existing sigma3 predictions and immutable probe heads.

No actor reload, optimizer step, generation, Slurm, or experiment mutation.
Ordinary rounds are explicitly PRE-FIT; saved heads are explicitly POST-FIT.
Only aggregate classification accuracy and sigma/variance errors are exported.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from search_g1.qprobe_model import CountMomentProbe, QProbeConfig, count_moment_loss

BASE = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015')
RUN = '70827e85fe584f0c84453aae907918a4'
PRODUCER = 'aca0e779e94ddc3fbd50e49c3d59e2e889fbcf12'
LOSS = dict(loss_mode='sigma3', mean_loss_weight=0, std_loss_weight=1,
            auxiliary_count_loss_weight=.1)


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def metrics(probabilities, counts, weights=None):
    p, k = np.asarray(probabilities, dtype=np.float64), np.asarray(counts, dtype=np.int64)
    assert p.shape == (len(k), 6) and len(k) and np.isfinite(p).all()
    assert (p >= 0).all() and (p <= 1).all() and np.allclose(p.sum(1), 1, atol=2e-6)
    assert ((k >= 0) & (k <= 5)).all()
    w = np.ones(len(k), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    assert w.shape == k.shape and np.isfinite(w).all() and (w > 0).all()
    average = lambda value: float(np.average(value, weights=w))
    pred = p.argmax(1)
    true_binary = (k > 0) & (k < 5)
    distribution = [average(k == i) for i in range(6)]
    variances = np.arange(6) * (5 - np.arange(6)) / 20
    stds = np.sqrt(variances)
    out = dict(questions=len(k), age_weighted=weights is not None,
        count_accuracy=average(pred == k),
        binary_probability_accuracy=average((p[:, 1:5].sum(1) >= .5) == true_binary),
        binary_argmax6_collapsed_accuracy=average(((pred > 0) & (pred < 5)) == true_binary),
        count_majority_baseline=max(distribution),
        binary_majority_baseline=max(average(true_binary), average(~true_binary)),
        count_distribution=distribution)
    for name, predicted, target in (('sigma', p @ stds, stds[k]),
                                     ('variance', p @ variances, variances[k])):
        error = predicted - target
        mse = average(error ** 2)
        target_variance = average((target - average(target)) ** 2)
        out.update({name+'_mae': average(abs(error)), name+'_mse': mse,
            name+'_rmse': math.sqrt(mse), name+'_r2': 1-mse/target_variance if target_variance else None,
            name+'_predicted_mean': average(predicted), name+'_actual_mean': average(target)})
    return out


def head_version(state):
    digest = hashlib.sha256(state['feature_spec_sha'].encode())
    for key, value in sorted(state['model'].items()):
        digest.update(key.encode()); digest.update(value.cpu().numpy().tobytes())
    return 'qprobe:%d:%d:%s' % (state['completed_outer'], state['update_count'], digest.hexdigest()[:16])


def snapshot(outer, rows):
    if outer == 0:
        ready, rp = read_json(BASE/'artifacts/question-probe/teacher-warmup-ready.json')
        path = Path(ready['head_checkpoint']['path'])
        expected_sha = ready['head_checkpoint']['sha256']
        logged = ready['head_metrics']
    else:
        path = BASE/('checkpoints/search-g1-common8/actor/global_step_%d/training_state/' % outer)/'question-probe-state.pt.head.pt'
        driver_path = path.with_name('question-probe-state.pt')
        raw_driver = driver_path.read_bytes()
        driver = torch.load(io.BytesIO(raw_driver), map_location='cpu', weights_only=False)
        assert driver['completed_outer'] == outer and driver['head_file'] == path.name
        assert driver['actor_snapshot_id'] == RUN+':'+str(outer)
        expected_sha = driver['head_sha256']
        rp = dict(path=str(driver_path), bytes=len(raw_driver), sha256=hashlib.sha256(raw_driver).hexdigest())
        logged = rows[outer-1]['head_update']
    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    assert digest == expected_sha
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    assert state['completed_outer'] == state['update_count'] == outer
    assert state['actor_snapshot_id'] == RUN+':'+str(outer)
    assert all(state['config'][k] == v for k, v in LOSS.items())
    config = QProbeConfig.from_mapping(state['config'])
    with torch.random.fork_rng(devices=[]):
        model = CountMomentProbe(state['feature_dim'], config.hidden_size).cpu()
    model.load_state_dict(state['model'], strict=True); model.eval()
    labels, features = state['replay_labels'], state['replay_features'].float().cpu()
    counts = np.array([x['correct_count'] for x in labels])
    assert len(labels) == len(features) and all(x['source_outer'] <= outer for x in labels)
    assert all(x['correct_count'] == sum(x['strict_em']) for x in labels)
    with torch.inference_mode():
        probabilities = torch.cat([model(features[i:i+256])['probabilities']
            for i in range(0, len(features), 256)]).numpy()
        prediction = model(features)
        weights = torch.tensor([2 ** (-(outer-x['source_outer'])/config.replay_half_life_outer)
            for x in labels], dtype=torch.float32)
        weights *= len(weights)/weights.sum()
        loss, parts = count_moment_loss(prediction, torch.tensor(counts), config, weights)
    assert abs(float(loss)-logged['replay_weighted_loss']) < 2e-6
    for key, value in parts.items():
        assert abs(float(value)-logged[key]) < 2e-6
    version = head_version(state)
    assert version == logged.get('version', rows[outer-1]['head_version'] if outer else version)
    if outer < len(rows): assert rows[outer]['prediction_version'] == version
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    by_age = {}
    for age in sorted(set(x['source_outer'] for x in labels)):
        mask = np.array([x['source_outer'] == age for x in labels])
        by_age[str(age)] = metrics(probabilities[mask], counts[mask])
    return dict(outer=outer, phase='POST_FIT_SAVED_HEAD', head_version=version,
        head_file=dict(path=str(path), bytes=len(raw), sha256=digest), binding_file=rp,
        fit_loss=float(loss), fit_parts={k:float(v) for k,v in parts.items()},
        replay_source_outers=sorted(set(x['source_outer'] for x in labels)),
        replay_unweighted=metrics(probabilities, counts),
        replay_age_weighted=metrics(probabilities, counts, weights.numpy()),
        current_cohort=by_age[str(outer)], replay_by_source_outer=by_age)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--cutoff', type=int, required=True)
    args = parser.parse_args(); assert 50 <= args.cutoff <= 200
    assert os.getuid() == 14090 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    manifest, mp = read_json(BASE/'logs/run-manifest.json')
    assert manifest['run_id'] == RUN and manifest['source_commit'] == PRODUCER and manifest['training_world_size'] == 4
    assert all(manifest['question_probe'][k] == v for k,v in LOSS.items())
    updates, up = read_json_lines(BASE/'logs/actual-updates.jsonl')
    assert [u['step'] for u in updates] == list(range(1, len(updates)+1)) and len(updates) >= args.cutoff
    rows = []
    for update in updates[:args.cutoff]:
        step, aid = update['step'], update['attempt_id']
        directory = BASE/'artifacts/question-probe/attempts'/aid
        selection, sp = read_json(directory/'selection.json')
        rollout, rp = read_json(directory/'rollout-audit.json')
        head, hp = read_json(BASE/'artifacts/question-probe/completed'/(aid+'.json'))
        assert selection['step'] == rollout['step'] == head['outer'] == step
        assert selection['attempt_id'] == rollout['attempt_id'] == head['attempt_id'] == aid
        assert selection['head_version'] == head['prediction_version'] == rollout['question_probe_version']
        assert selection['prediction_before_label'] and not selection['post_rollout_em_filter']
        assert rollout['run_id'] == RUN and rollout['source_commit'] == PRODUCER and rollout['no_post_rollout_refill']
        chosen = {q['ordinal']:q for q in selection['questions'] if q['selected']}
        groups = rollout['groups']; assert len(groups) == len(chosen) == 512
        assert len(set(g['ordinal'] for g in groups)) == len(set(g['question_id'] for g in groups)) == 512
        assert all(chosen[g['ordinal']]['qid'] == g['question_id'] and
            chosen[g['ordinal']]['content_sha256'] == g['content_sha256'] and
            chosen[g['ordinal']]['selection_source'] == g['source'] for g in groups)
        p = np.array([chosen[g['ordinal']]['probabilities'] for g in groups])
        k = np.array([sum(g['strict_correctness']) for g in groups])
        assert all(len(g['strict_correctness']) == 5 and all(type(v) is int and v in (0,1)
            for v in g['strict_correctness']) for g in groups)
        observed = metrics(p, k)
        assert abs(observed['sigma_mae']-head['prequential']['all']['std_mae']) < 1e-7
        assert abs(observed['sigma_mse']-head['prequential']['all']['std_mse']) < 1e-7
        rows.append(dict(outer=step, phase='PRE_FIT_CURRENT_512', attempt_id=aid,
            prediction_version=head['prediction_version'], head_version=head['head_version'],
            head_update=head['head_update'], metrics=observed,
            files=dict(selection=sp, rollout=rp, head=hp)))
    snapshots = [snapshot(t, rows) for t in (0,5,50,100,150,200) if t <= args.cutoff]
    result = dict(schema='sigma3-accuracy-variance-diagnostic-v1', epoch=time.time(), run_id=RUN,
        cutoff=args.cutoff, manifest_file=mp, updates_file=up, per_outer_pre_fit=rows,
        post_fit_snapshots=snapshots,
        binary_definition='sum(p1..p4)>=0.5; separate argmax6-collapsed accuracy also retained',
        variance_definition='E[sigma^2]=sum(p_k*k*(5-k)/20), not E[sigma]^2',
        scope='Existing immutable data only; no actor load, fit, optimizer step, generation, Slurm or experiment writes',
        post_fit_accuracy_recorded_every_outer=False)
    print(json.dumps(result, allow_nan=False), flush=True)


def read_json_lines(path):
    raw = Path(path).read_bytes()
    return [json.loads(s) for s in raw.splitlines() if s.strip()], dict(path=str(path),
        bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


if __name__ == '__main__': main()

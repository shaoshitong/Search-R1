"""CPU-only analysis of immutable probe heads and saved pre-fit predictions.

Reads existing records through a fixed cutoff. It never generates, fits, queries
Slurm, loads an actor, or writes into an experiment. Output contains aggregates.
"""
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


ROOT = Path('/data/user/jxu957/experiments')
OLD = ROOT / 'qprobe-teacher-warmup200-4gpu-jxu957-recovery718509-20261014'
NEW = ROOT / 'qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014'
CUTOFF = 108
SLICES = ('all', 'random_exploration', 'predicted_diversity')


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    positive, negative = int(labels.sum()), int((1 - labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind='stable')
    total, start = 0.0, 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        total += float(labels[order[start:end]].sum()) * ((start + 1 + end) / 2)
        start = end
    return (total - positive * (positive + 1) / 2) / (positive * negative)


def metrics(probabilities, counts, weights=None):
    p = np.asarray(probabilities, dtype=np.float64)
    k = np.asarray(counts, dtype=np.int64)
    assert p.shape == (len(k), 6) and np.all(np.isfinite(p))
    assert np.all(p >= 0) and np.allclose(p.sum(1), 1, atol=1e-5, rtol=0)
    w = np.ones(len(k)) if weights is None else np.asarray(weights, dtype=np.float64)
    assert len(k) > 0 and w.shape == k.shape and np.all(w > 0)
    average = lambda values: float(np.dot(w, values) / w.sum())
    pred = p.argmax(1)
    mixed = (k > 0) & (k < 5)
    pm = p[:, 1:5].sum(1)
    binary = pm >= .5
    hard_binary = (pred > 0) & (pred < 5)
    assert not ((pred == k) & (hard_binary != mixed)).any()
    tp = average(mixed & binary)
    tn = average(~mixed & ~binary)
    fp = average(~mixed & binary)
    fn = average(mixed & ~binary)
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    dist = [average(k == i) for i in range(6)]
    std = np.sqrt(np.arange(6) * (5 - np.arange(6)) / 20)
    expected_std = p @ std
    confusion = [[int(((k == i) & (pred == j)).sum()) for j in range(6)] for i in range(6)]
    return dict(questions=len(k), weighted=weights is not None,
                count_accuracy=average(pred == k), count_majority_baseline=max(dist),
                count_class_fractions=dist, count_confusion=confusion,
                count_macro_recall=float(np.mean([average((pred == i) & (k == i)) / dist[i]
                                                 for i in range(6) if dist[i] > 0])),
                count_nll=average(-np.log(np.maximum(p[np.arange(len(k)), k], 1e-12))),
                count_empirical_prior_nll=-sum(x * math.log(x) for x in dist if x > 0),
                actual_mixed_fraction=average(mixed), predicted_mixed_mean=average(pm),
                predicted_mixed_positive_fraction=average(binary),
                mixed_accuracy=tp + tn, mixed_majority_baseline=max(average(mixed), average(~mixed)),
                coarsened_argmax_binary_accuracy=average(hard_binary == mixed),
                coarsened_argmax_mixed_recall=average(hard_binary & mixed) / average(mixed) if mixed.any() else None,
                binary_decision_disagreement=average(hard_binary != binary),
                six_correct_probability_sum_binary_wrong=average((pred == k) & (binary != mixed)),
                six_errors_crossing_mixed_boundary=average(hard_binary != mixed) / average(pred != k) if (pred != k).any() else None,
                mixed_precision=tp / (tp + fp) if tp + fp else None,
                mixed_recall=recall, mixed_specificity=specificity,
                mixed_balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
                mixed_brier=average((pm - mixed) ** 2),
                mixed_auc_unweighted=auc(mixed, pm), std_ranking_auc_unweighted=auc(mixed, expected_std),
                std_mae=average(abs(expected_std - std[k])),
                expected_std_mean=average(expected_std), actual_std_mean=average(std[k]))


def snapshot(outer):
    if outer == 0:
        path = OLD / 'artifacts/question-probe/teacher-warmup-head/initial.pt'
    else:
        base = OLD if outer == 5 else NEW
        path = base / ('checkpoints/search-g1-common8/actor/global_step_' + str(outer)) / 'training_state/question-probe-state.pt.head.pt'
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    assert state['completed_outer'] == outer and state['update_count'] == outer
    config = QProbeConfig.from_mapping(state['config'])
    with torch.random.fork_rng(devices=[]):
        model = CountMomentProbe(state['feature_dim'], config.hidden_size).cpu()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    labels = state['replay_labels']
    features = state['replay_features'].float().cpu()
    assert len(labels) == len(features) and all(x['source_outer'] <= outer for x in labels)
    counts = np.array([x['correct_count'] for x in labels])
    assert all(x['correct_count'] == sum(x['strict_em']) for x in labels)
    with torch.inference_mode():
        pieces = [model(features[i:i + 256])['probabilities'] for i in range(0, len(features), 256)]
        probabilities = torch.cat(pieces).numpy()
        pred = model(features)
        weights = torch.tensor([2 ** (-(outer - x['source_outer']) / config.replay_half_life_outer)
                                for x in labels], dtype=torch.float32)
        weights *= len(weights) / weights.sum()
        loss, parts = count_moment_loss(pred, torch.tensor(counts), config, weights)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    per_age = {}
    for age in sorted({x['source_outer'] for x in labels}):
        mask = np.array([x['source_outer'] == age for x in labels])
        per_age[str(age)] = metrics(probabilities[mask], counts[mask])
    per_source = {}
    for source in sorted({x['selection_source'] for x in labels}):
        mask = np.array([x['selection_source'] == source for x in labels])
        per_source[source] = dict(unweighted=metrics(probabilities[mask], counts[mask]),
                                  age_weighted=metrics(probabilities[mask], counts[mask], weights.numpy()[mask]))
    model_digest = hashlib.sha256(state['feature_spec_sha'].encode())
    for key, value in sorted(state['model'].items()):
        model_digest.update(key.encode()); model_digest.update(value.cpu().numpy().tobytes())
    version = 'qprobe:' + str(outer) + ':' + str(outer) + ':' + model_digest.hexdigest()[:16]
    return dict(outer=outer, path=str(path), bytes=len(raw), sha256=digest, version=version,
                actor_snapshot_id=state['actor_snapshot_id'], feature_dim=state['feature_dim'],
                parameter_count=sum(x.numel() for x in model.parameters()), config=state['config'],
                replay_unique_qids=len({x['qid'] for x in labels}), replay_groups=len(labels),
                replay_source_outers=sorted({x['source_outer'] for x in labels}),
                fit_loss=float(loss), fit_loss_parts={k:float(v) for k,v in parts.items()},
                training_unweighted=metrics(probabilities, counts),
                training_age_weighted=metrics(probabilities, counts, weights.numpy()),
                training_per_source_outer=per_age, training_per_selection_source=per_source)


def main():
    assert os.getuid() == 14090 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    # Check ties and direction independently before computing any real AUC.
    assert auc([0, 1], [0, 1]) == 1 and auc([0, 1], [1, 0]) == 0 and auc([0, 1], [1, 1]) == .5
    rows, buffers, seen_qids = [], {}, set()
    run_id = None
    for step in range(1, CUTOFF + 1):
        base = OLD if step <= 5 else NEW
        updates = [json.loads(x) for x in (base / 'logs/actual-updates.jsonl').read_text().splitlines()]
        actual = [x for x in updates if x['step'] == step]
        assert len(actual) == 1 and actual[0]['actual_optimizer_writes'] > 0
        aid = actual[0]['attempt_id']; directory = base / 'artifacts/question-probe/attempts' / aid
        selection, sp = read_json(directory / 'selection.json')
        rollout, rp = read_json(directory / 'rollout-audit.json')
        head, hp = read_json(base / 'artifacts/question-probe/completed' / (aid + '.json'))
        assert selection['step'] == rollout['step'] == head['outer'] == step
        assert selection['attempt_id'] == rollout['attempt_id'] == head['attempt_id'] == aid
        assert selection['head_version'] == head['prediction_version']
        assert selection['prediction_before_label'] and not selection['post_rollout_em_filter']
        assert selection['exploration_questions'] == 64 and rollout['no_post_rollout_refill']
        assert head['prequential'] == rollout['prequential']
        if run_id is None: run_id = rollout['run_id']
        assert rollout['run_id'] == run_id
        qs = {q['ordinal']: q for q in selection['questions'] if q['selected']}
        groups = rollout['groups']; assert len(qs) == len(groups) == 512
        p = np.array([qs[g['ordinal']]['probabilities'] for g in groups])
        k = np.array([sum(g['strict_correctness']) for g in groups])
        source = np.array([g['source'] for g in groups])
        qids = [g['question_id'] for g in groups]
        data_sources = np.array([json.loads(qid)[0] for qid in qids])
        unseen = np.array([qid not in seen_qids for qid in qids])
        seen_qids.update(qids)
        scalar = dict(step=step, attempt_id=aid, prediction_version=selection['head_version'],
                      head_version=head['head_version'], head_update=head['head_update'],
                      files=dict(selection=sp, rollout=rp, head=hp), slices={},
                      qid_unseen_in_formal_history_fraction=float(unseen.mean()), data_source_slices={},
                      candidate_data_source_counts={name:sum(json.loads(q['qid'])[0] == name
                          for q in selection['questions']) for name in sorted(set(data_sources))})
        for name in SLICES:
            mask = np.ones(len(k), dtype=bool) if name == 'all' else source == name
            expected = dict(all=512, random_exploration=64, predicted_diversity=448)[name]
            assert mask.sum() == expected
            z = metrics(p[mask], k[mask])
            for field in ('count_nll','mixed_brier','std_mae','actual_mixed_fraction'):
                assert abs(z[field] - head['prequential'][name][field]) < 1e-7
            scalar['slices'][name] = z
        for name, mask in [('qid_seen_in_formal_history', ~unseen), ('qid_unseen_in_formal_history', unseen)]:
            if mask.any(): scalar['slices'][name] = metrics(p[mask], k[mask])
        for data_source in sorted(set(data_sources)):
            scalar['data_source_slices'][data_source] = {}
            for name in SLICES:
                mask = (data_sources == data_source) & (np.ones(len(k), dtype=bool) if name == 'all' else source == name)
                if mask.any(): scalar['data_source_slices'][data_source][name] = metrics(p[mask], k[mask])
        rows.append(scalar); buffers[step] = (p, k, source, data_sources)
    windows = []
    for start in range(1, CUTOFF + 1, 10):
        end = min(start + 9, CUTOFF)
        p = np.concatenate([buffers[t][0] for t in range(start, end + 1)])
        k = np.concatenate([buffers[t][1] for t in range(start, end + 1)])
        sources = np.concatenate([buffers[t][2] for t in range(start, end + 1)])
        data_sources = np.concatenate([buffers[t][3] for t in range(start, end + 1)])
        window = dict(first=start, last=end, mean_fit_loss=float(np.mean([r['head_update']['replay_weighted_loss'] for r in rows[start-1:end]])), slices={}, data_source_slices={})
        for name in SLICES:
            mask = np.ones(len(k), dtype=bool) if name == 'all' else sources == name
            window['slices'][name] = metrics(p[mask], k[mask])
        for data_source in sorted(set(data_sources)):
            window['data_source_slices'][data_source] = {}
            for name in SLICES:
                mask = (data_sources == data_source) & (np.ones(len(k), dtype=bool) if name == 'all' else sources == name)
                if mask.any(): window['data_source_slices'][data_source][name] = metrics(p[mask], k[mask])
        windows.append(window)
    heads = [snapshot(t) for t in (0, 5, 50, 100)]
    for head in heads:
        following = rows[head['outer']]
        assert following['prediction_version'] == head['version']
        head['next_outer_pre_fit'] = dict(step=following['step'], slices=following['slices'])
        if head['outer']:
            logged = rows[head['outer']-1]['head_update']
            assert abs(logged['replay_weighted_loss'] - head['fit_loss']) < 1e-6
            head['fit_loss_equals_actual_logged_value'] = True
    print(json.dumps(dict(schema='qprobe-fit-diagnostic-v1', epoch=time.time(), cutoff=CUTOFF, run_id=run_id,
                          scope='Existing immutable CPU probe heads and existing pre-fit selected-question predictions; no actor reload, fitting, rollout, Slurm or experiment writes',
                          training_head_snapshots=heads, per_outer=rows, windows=windows,
                          caution='Pre-fit observations are temporal generalization, not a held-out question split. Training replay labels and policies change over time; classification targets are stochastic K of5, not deterministic difficulty.'), allow_nan=False))


if __name__ == '__main__':
    main()

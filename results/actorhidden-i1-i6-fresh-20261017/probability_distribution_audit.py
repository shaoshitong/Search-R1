"""CPU-only aggregation of existing predictions. No models, forwards or changes.

Execute on h100-8; only aggregate statistics and source hashes leave the node.
Pre-update trajectory predictions were not persisted by the trainer: this
reports their existing aggregate metrics without reconstructing them.
"""
import hashlib
import json
import math
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return math.fsum(values) / len(values) if values else None


def quantiles(values):
    values = sorted(values)
    def at(f):
        x = (len(values) - 1) * f
        i = int(x)
        return values[i] + (values[min(i + 1, len(values)-1)] - values[i]) * (x-i)
    return {str(q): at(q) for q in (0, .01, .1, .25, .5, .75, .9, .99, 1)}


def describe(values):
    return dict(count=len(values), mean=mean(values), quantiles=quantiles(values),
                below_001=mean([v < .01 for v in values]),
                above_099=mean([v > .99 for v in values]),
                between_02_08=mean([.2 <= v <= .8 for v in values]))


def auc(scores, labels):
    pairs = sorted(zip(scores, labels)); positives = sum(labels)
    if not positives or positives == len(labels):
        return None
    rank_sum = 0; i = 0
    while i < len(pairs):
        end = i + 1
        while end < len(pairs) and pairs[end][0] == pairs[i][0]:
            end += 1
        rank_sum += ((i + 1 + end) / 2) * sum(y for _, y in pairs[i:end])
        i = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * (len(labels)-positives))


def calibration(scores, labels):
    bins = []
    for b in range(10):
        ids = [i for i, p in enumerate(scores) if min(9, int(p*10)) == b]
        bins.append(dict(low=b/10, high=(b+1)/10, count=len(ids),
                         mean_prediction=mean([scores[i] for i in ids]),
                         empirical_rate=mean([labels[i] for i in ids])))
    return dict(brier=mean([(p-y)**2 for p, y in zip(scores, labels)]),
                auc=auc(scores, labels), bins=bins,
                ece=sum(b['count'] * abs(b['mean_prediction']-b['empirical_rate'])
                        for b in bins if b['count'])/len(scores))


def probabilities(group):
    return group.get('probabilities', group.get('trajectory_probabilities'))


def summarize(groups):
    ps = [p for row in groups for p in probabilities(row)]
    qs = [row.get('mixed_probability', row.get('q')) for row in groups]
    errors = []
    for row, q in zip(groups, qs):
        p = probabilities(row)
        assert len(p) == 5 and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
        errors.append(abs(q-(1-math.prod(p)-math.prod(1-v for v in p))))
    out = dict(questions=len(groups), p=describe(ps), q=describe(qs),
               max_formula_absolute_error=max(errors),
               mean_within_group_p_range=mean([max(probabilities(r))-min(probabilities(r)) for r in groups]),
               all_five_p_between_02_08=mean([all(.2 <= p <= .8 for p in probabilities(r)) for r in groups]))
    if 'strict_em' not in groups[0]:
        return out
    ys = [y for row in groups for y in row['strict_em']]
    ms = [int(0 < sum(row['strict_em']) < 5) for row in groups]
    assert all(row['mixed_label'] == m for row, m in zip(groups, ms))
    out.update(true_trajectory_correct_rate=mean(ys), true_mixed_rate=mean(ms),
               trajectory_calibration=calibration(ps, ys), group_calibration=calibration(qs, ms),
               trajectory_accuracy=mean([int(p >= .5)==y for p,y in zip(ps,ys)]))
    out['by_correct_count'] = {}
    for k in range(6):
        selected = [(r,q) for r,q in zip(groups,qs) if sum(r['strict_em']) == k]
        if selected:
            out['by_correct_count'][str(k)] = dict(count=len(selected),
                mean_p=mean([p for r,q in selected for p in probabilities(r)]),
                mean_q=mean([q for r,q in selected]),
                predicted_mixed_fraction=mean([q >= .5 for r,q in selected]))
    for label in (0, 1):
        out['p_given_em_'+str(label)] = describe([p for p,y in zip(ps,ys) if y==label])
    top = sorted(zip(qs, ms), reverse=True)[:max(1, len(groups)//10)]
    out['top_decile'] = dict(count=len(top), mean_q=mean([q for q,y in top]), true_mixed_rate=mean([y for q,y in top]))
    return out


def inspect(base, historical_base):
    base, historical_base = Path(base), Path(historical_base)
    config = read(base/'preparation/launch-config.json')
    output = dict(run_id=config['run_id'], base=str(base), observed_unix=time.time(),
                  training_modified=False, tensor_loaded=False, gpu_forward_performed=False,
                  predictions_or_ids_returned=False, periods=[], selections=[], missing=[])
    latest = read(base/'latest-joint.json')['outer']
    for outer in (20,40,60,80):
        if outer > latest:
            continue
        root = (historical_base if outer <= 50 else base)/'outers'/('outer-%06d'%outer)
        metrics = read(root/'probe-update/metrics.json')
        periodic = dict(outer=outer, sampling_actor_version=metrics['actor_version'],
                        learning_rate=metrics['actual_learning_rate'], before_update=metrics['before_update'],
                        after_metrics={key:metrics[key] for key in ('train','test')},
                        metrics_sha256=sha(root/'probe-update/metrics.json'), splits={})
        for split, expected in (('train',5120),('test',512)):
            path = root/'probe-update'/(split+'-group-predictions.jsonl')
            group_rows = rows(path)
            assert len(group_rows) == expected and len({r['qid'] for r in group_rows}) == expected
            periodic['splits'][split] = dict(summarize(group_rows), source_sha256=sha(path))
        output['periods'].append(periodic)
    wanted = sorted(set([1,19,20,21,39,40,41,50,51,59,60,61,65,71,latest]))
    for outer in wanted:
        if outer > latest:
            continue
        root = (historical_base if outer <= 50 else base)/'outers'/('outer-%06d'%outer)
        path = root/'probe-score/candidate-scores.jsonl'
        if not path.exists():
            output['missing'].append(dict(outer=outer, role='candidate_scores', path=str(path)))
            continue
        groups = rows(path); selection = read(root/'probe-score/selection.json')
        assert len(groups)==5120 and selection['outcome_labels_used'] is False
        labels = {r['trajectory_id']:r['strict_em'] for r in rows(root/'full/own-em.jsonl')}
        top = []
        for group in selection['selected']:
            row = dict(group, strict_em=[labels[t] for t in group['trajectory_ids']])
            row['mixed_label'] = int(0 < sum(row['strict_em']) < 5)
            top.append(row)
        output['selections'].append(dict(outer=outer, sampling_actor_version=selection['actor_version'],
            full_candidate_unlabelled=summarize(groups), selected_labelled=summarize(top),
            candidate_scores_sha256=sha(path), selection_sha256=sha(root/'probe-score/selection.json')))
    output['actual_complete_outer'] = latest
    return output

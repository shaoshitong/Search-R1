"""CPU-only small-metadata analysis of selection-time predictions and own outcomes.

Execute through the existing remote transport; never load tensors or modify training.
"""
import hashlib
import json
from pathlib import Path
import sys
import time


def collect(base):
    base = Path(base)
    sys.path.insert(0, str(base / 'repo'))
    from reproduction.prefix_selection200.selection_metrics import summarize_outer
    read = lambda p: json.loads(Path(p).read_text())
    latest = read(base / 'latest-joint.json')
    config = read(base / 'preparation/launch-config.json')
    result = dict(base=str(base), run_id=config['run_id'], observed_unix=time.time(),
                  complete_outer=latest['outer'], prediction_clock='selection_before_actor_and_probe_updates',
                  threshold=0.5, rows=[], periodic=[], missing=[], training_modified=False,
                  tensors_or_full_trajectories_loaded=False)
    for outer in range(1, latest['outer'] + 1):
        try:
            joint_path = base / 'outers' / ('outer-%06d' % outer) / 'complete.json'
            joint = read(joint_path)
            assert joint['run_id'] == config['run_id']
            facts = summarize_outer(joint_path)
            selected = read(facts['selection_path'])['selected']
            labels = {v['trajectory_id']: v for v in
                      map(json.loads, Path(facts['strict_em_path']).read_text().splitlines())}
            tp = tn = fp = fn = 0
            probabilities = []
            for group in selected:
                k = sum(labels[tid]['strict_em'] for tid in group['trajectory_ids'])
                true, pred = 0 < k < 5, group['q'] >= 0.5
                tp += int(true and pred)
                tn += int(not true and not pred)
                fp += int(not true and pred)
                fn += int(true and not pred)
                probabilities.append(group['q'])
            positive, negative = tp + fn, tn + fp
            recall_pos = tp / positive if positive else None
            recall_neg = tn / negative if negative else None
            row = {k: facts[k] for k in ('outer', 'actual_mixed_fraction',
                'mean_predicted_mixed_probability', 'mixed_questions', 'all_wrong_questions',
                'all_correct_questions', 'selection_sha256', 'strict_em_sha256', 'joint_sha256',
                'probe_used_to_select_sha256', 'sampling_actor_version')}
            row.update(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp+tn)/512,
                       mixed_recall=recall_pos, nonmixed_recall=recall_neg,
                       balanced_accuracy=(recall_pos+recall_neg)/2 if positive and negative else None,
                       predicted_mixed=tp+fp, q_min=min(probabilities), q_max=max(probabilities))
            result['rows'].append(row)
            if joint.get('periodic_probe_update'):
                # Imported completed rounds retain their authentic metadata source directory.
                root = Path(facts['selection_path']).parent.parent
                metrics_path = root / 'probe-update/metrics.json'
                loss_path = root / 'probe-update/global-loss-updates.jsonl'
                metrics = read(metrics_path)
                curve = [json.loads(line) for line in loss_path.read_text().splitlines()]
                assert len(curve) == 100
                assert metrics['actual_learning_rate'] == config['probe_learning_rate'] == 1e-5
                assert metrics['actual_adam_steps_per_rank'] == [outer//20*100]*8
                assert [v['step'] for v in curve] == list(range(1,101))
                result['periodic'].append(dict(outer=outer, before_update=metrics['before_update'],
                    after_update={s: metrics[s] for s in ('train','test')}, loss_rows=curve,
                    metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                    global_loss_sha256=hashlib.sha256(loss_path.read_bytes()).hexdigest()))
        except (OSError, ValueError, KeyError, AssertionError) as error:
            result['missing'].append(dict(outer=outer, error=repr(error)))
    result['status'] = 'COMPLETE_SELECTED_ACCURACY_READBACK' if not result['missing'] else 'PARTIAL_READBACK'
    return result


if __name__ == '__main__':
    print(json.dumps(collect(sys.argv[1]), allow_nan=False))

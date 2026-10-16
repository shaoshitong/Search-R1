"""CPU-only small accuracy report from already committed replay metrics.

No models, feature arrays, trajectory predictions, or GPU operations are read.
Iterations1..6 use their original test-only receipts; iterations7..70 use the
complete post-update train and fixed-test receipts from the resumed namespace.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

from reproduction.prefix_only_correctness_ab.features import canonical, once, remote


FORMULA = '1-prod(p_j)-prod(1-p_j)'
TRUE_MIXED = '0<sum(own_strict_em)<5'
COMPLETE_STATUS = 'QUESTION_PLUS20_REPLAY_ITERATION_COMPLETE'


def stable_json(path):
    path = Path(path)
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        data = handle.read()
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Small metric receipt changed while reading: ' + str(path))
    return json.loads(data), dict(path=str(path), bytes=len(data),
                                  sha256=hashlib.sha256(data).hexdigest())


def checked_accuracy(group, count):
    """Validate stored G5 confusion totals; never reinterpret trajectory accuracy."""
    if (group.get('count') != count or group.get('threshold') != .5 or
            group.get('formula') != FORMULA or group.get('true_mixed') != TRUE_MIXED):
        raise ValueError('G5 accuracy count/formula/threshold/own-label semantics differ')
    values = [group[k] for k in ('tp', 'fp', 'fn', 'tn')]
    if any(type(v) is not int or v < 0 for v in values) or sum(values) != count:
        raise ValueError('G5 confusion totals are not complete integer question counts')
    tp, fp, fn, tn = values
    accuracy = (tp + tn) / count
    if (group.get('accuracy') != accuracy or group.get('mixed_count') != tp + fn or
            group.get('nonmixed_count') != fp + tn):
        raise ValueError('Stored G5 accuracy or truth counts differ from its confusion matrix')
    return accuracy


def metric_row(metrics, complete, iteration, source_steps, source_epoch, legacy):
    """One immutable iteration, with absent historical train evaluation explicit."""
    for item in (metrics, complete):
        if (item.get('replay_iteration') != iteration or item.get('source_data_epoch') != iteration or
                item.get('source_epoch') != source_epoch or item.get('new_adam_steps_per_rank') != 100 or
                item.get('actual_adam_steps_per_rank') != [v + iteration * 100 for v in source_steps]):
            raise ValueError('Replay iteration/source data/actual Adam completion coordinates differ')
    if complete.get('status') != COMPLETE_STATUS:
        raise ValueError('Replay iteration is not committed complete')
    test = metrics['mixed_group'] if legacy else metrics['test_mixed_group']
    test_accuracy = checked_accuracy(test, 512)
    if test['mixed_count'] != 183 or test['nonmixed_count'] != 329:
        raise ValueError('Fixed512 G5 truth counts differ from the unchanged diagnostic')
    if (metrics.get('mixed_group_accuracy') != test_accuracy or
            complete.get('mixed_group_accuracy') != test_accuracy):
        raise ValueError('Test mixed accuracy alias differs from its G5 confusion matrix')
    if legacy:
        train_accuracy = None
        train_status = 'NOT_RECORDED_IN_ORIGINAL_TEST_ONLY_ITERATION'
    else:
        if metrics.get('mixed_group') != test:
            raise ValueError('Test alias and explicit test G5 metric differ')
        train_accuracy = checked_accuracy(metrics['train_mixed_group'], 5120)
        if (metrics.get('test_mixed_group_accuracy') != test_accuracy or
                complete.get('test_mixed_group_accuracy') != test_accuracy or
                metrics.get('train_mixed_group_accuracy') != train_accuracy or
                complete.get('train_mixed_group_accuracy') != train_accuracy):
            raise ValueError('Explicit post-update train/test accuracy differs from completion receipt')
        if metrics.get('mixed_metric_semantics') != 'complete train and fixed test eval/no_grad passes of the same post-update head':
            raise ValueError('Train/test predictions are not the same post-update head evaluation')
        train_status = 'COMPLETE_POST_UPDATE_OWN_BATCH_G5_EVALUATION'
    return dict(replay_iteration=iteration, source_data_epoch=iteration, status='COMPLETE',
                train_mixed_group_accuracy=train_accuracy, train_accuracy_status=train_status,
                test_mixed_group_accuracy=test_accuracy, train_question_count=5120 if not legacy else None,
                test_question_count=512, threshold=.5,
                test_manifest_sha256=metrics['test_manifest_sha256'],
                actual_adam_steps_per_rank=list(metrics['actual_adam_steps_per_rank']))


def accuracy_summary(rows):
    valid = [row for row in rows if row['status'] == 'COMPLETE']
    if not valid:
        return dict(latest_complete=None, best_test=None)
    best = max(row['test_mixed_group_accuracy'] for row in valid)
    tied = [row for row in valid if row['test_mixed_group_accuracy'] == best]
    return dict(latest_complete=valid[-1], best_test=dict(
        test_mixed_group_accuracy=best,
        replay_iterations=[row['replay_iteration'] for row in tied],
        corresponding_train_mixed_group_accuracy=[row['train_mixed_group_accuracy'] for row in tied]))


def collect(base, legacy_base, through=70, legacy_through=6):
    """Read only small committed metrics/complete receipts from both namespaces."""
    if through != 70 or legacy_through != 6:
        raise ValueError('This final report requires ordered replay1..70 with historical test-only1..6')
    base, legacy_base = Path(base), Path(legacy_base)
    config, config_proof = stable_json(base / 'preparation/launch-config.json')
    old_config, old_config_proof = stable_json(legacy_base / 'preparation/launch-config.json')
    if (config['run_id'] != old_config['run_id'] or config['source_epoch'] != old_config['source_epoch'] or
            config['source_adam_steps_per_rank'] != old_config['source_adam_steps_per_rank'] or
            config['source_checkpoint_sha256'] != old_config['source_checkpoint_sha256']):
        raise ValueError('Report namespaces do not continue the same source head/run')
    source_steps, source_epoch = config['source_adam_steps_per_rank'], config['source_epoch']
    if (len(source_steps) != 8 or any(type(v) is not int or v != source_epoch * 100 for v in source_steps) or
            config['iterations'] != 70 or old_config['iterations'] != 70):
        raise ValueError('Expected synchronized eight-rank source head and70 replay iterations')
    rows, issues, provenance, fixed_manifest = [], [], [], None
    for iteration in range(1, through + 1):
        legacy = iteration <= legacy_through
        root = legacy_base if legacy else base
        path = root / 'train/iterations' / ('iteration-%04d' % iteration)
        pending = dict(replay_iteration=iteration, source_data_epoch=iteration, status='PENDING',
                       train_mixed_group_accuracy=None, test_mixed_group_accuracy=None,
                       train_accuracy_status='PENDING_COMPLETION', threshold=.5)
        try:
            complete, complete_proof = stable_json(path / 'complete.json')
            metrics, metric_proof = stable_json(path / 'metrics.json')
            row = metric_row(metrics, complete, iteration, source_steps, source_epoch, legacy)
            if fixed_manifest is not None and row['test_manifest_sha256'] != fixed_manifest:
                raise ValueError('Fixed diagnostic feature manifest changed across replay iterations')
            fixed_manifest = row['test_manifest_sha256']
            rows.append(row)
            provenance.append(dict(replay_iteration=iteration, complete=complete_proof, metrics=metric_proof))
        except FileNotFoundError as error:
            rows.append(pending)
            issues.append(dict(replay_iteration=iteration, status='PENDING', missing=str(error.filename)))
        except (KeyError, TypeError, ValueError) as error:
            rows.append(dict(pending, status='INVALID_AGGREGATE'))
            issues.append(dict(replay_iteration=iteration, status='INVALID_AGGREGATE', reason=str(error)))
    completed = sum(row['status'] == 'COMPLETE' for row in rows)
    status = ('QUESTION20_REPLAY70_AGGREGATE_COMPLETE' if completed == through and not issues
              else 'INVALID_AGGREGATE' if any(i['status'] == 'INVALID_AGGREGATE' for i in issues)
              else 'PENDING_INCOMPLETE_REPLAY')
    return dict(schema_version=1, role='question20_replay70_mixed_group_accuracy_report', status=status,
                observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                run_id=config['run_id'], base=str(base), historical_test_only_base=str(legacy_base),
                requested_iterations=70, complete_iterations=completed, rows=rows, issues=issues,
                **accuracy_summary(rows), fixed_test_manifest_sha256=fixed_manifest,
                test_majority_accuracy=329 / 512, test_true_mixed_groups=183,
                test_true_nonmixed_groups=329, threshold=.5, formula=FORMULA, true_mixed=TRUE_MIXED,
                train_semantics='Full post-update inference on this iteration own5120G5 batch; batches differ between iterations; not a fixed-train learning curve',
                test_semantics='Same fixed512G5 cached trajectories across all70 post-update heads',
                historical_missing_train_iterations=list(range(1, 7)),
                metrics_independently_recomputed_from_predictions=False,
                aggregate_confusion_and_completion_receipts_crosschecked=True,
                model_loaded=False, gpu_used=False, full_prediction_files_read=False,
                training_modified=False, qa_em_f1_reported=False,
                provenance=dict(config=config_proof, historical_config=old_config_proof, iterations=provenance))


def markdown(report):
    def percent(value):
        return 'not recorded' if value is None else '%.4f%%' % (100 * value)
    lines = ['# Question +20-token replay: G5 mixed decision accuracy', '',
             'Status: `%s`; complete iterations: %d/70.' % (report['status'], report['complete_iterations']), '',
             'Decision: `1-prod(p_j)-prod(1-p_j) >= 0.5`; truth: `0<sum(own strict EM)<5`.', '',
             'Train: post-update full5120G5 evaluation on each iteration own batch. The batches differ; this is not a fixed-train curve.',
             'Test: the same fixed512G5 trajectories for every iteration. Train accuracy was not recorded for iterations1..6.', '',
             'Fixed-test majority accuracy: %.4f%%.' % (100 * report['test_majority_accuracy']), '']
    latest, best = report['latest_complete'], report['best_test']
    if latest is not None:
        lines.extend(['Latest complete iteration%d: train %s; test %s.' % (
            latest['replay_iteration'], percent(latest['train_mixed_group_accuracy']),
            percent(latest['test_mixed_group_accuracy'])),
            'Highest fixed-test accuracy: %s at iteration(s) %s.' % (
                percent(best['test_mixed_group_accuracy']), ', '.join(map(str, best['replay_iterations']))), ''])
    lines.extend(['| Replay iteration | Post-update own-batch train accuracy | Fixed-test accuracy |',
                  '|---:|---:|---:|'])
    for row in report['rows']:
        lines.append('| %d | %s | %s |' % (
            row['replay_iteration'], percent(row['train_mixed_group_accuracy']),
            percent(row['test_mixed_group_accuracy'])))
    lines.extend(['', 'Only saved aggregate confusion totals and completion receipts were checked. Full predictions and models were not loaded.', ''])
    return '\n'.join(lines)


def run(base, legacy_base, output):
    base, legacy_base, output = remote(base), remote(legacy_base), remote(output)
    report = collect(base, legacy_base)
    # Existing output is intentionally not overwritten; use an independent report tag.
    output.mkdir(parents=True, exist_ok=False)
    once(output / 'summary.json', report)
    path = output / 'README.md'
    with path.open('x', encoding='utf-8') as handle:
        handle.write(markdown(report))
        handle.flush()
        os.fsync(handle.fileno())
    small = dict(status=report['status'], complete_iterations=report['complete_iterations'],
                 output=str(output), summary_sha256=hashlib.sha256((output / 'summary.json').read_bytes()).hexdigest(),
                 markdown_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 latest_complete=report['latest_complete'], best_test=report['best_test'],
                 model_loaded=False, gpu_used=False, full_prediction_files_read=False)
    print(json.dumps(small, allow_nan=False), flush=True)
    return 0 if report['status'] == 'QUESTION20_REPLAY70_AGGREGATE_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--legacy-base', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    return run(args.base, args.legacy_base, args.output)


if __name__ == '__main__':
    raise SystemExit(main())

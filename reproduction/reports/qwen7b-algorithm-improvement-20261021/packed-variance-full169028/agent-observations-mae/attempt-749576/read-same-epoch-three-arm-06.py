"""Remote CPU-only paired diagnostic summaries of the already viewed pools."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

EPOCH = 6
suite = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1')
jobs = {'mse': '749568', 'mae': '749576', 'z_mse_anchor': '749577'}
roles = ('fixed_train2048_G5', 'fixed_validation2048_G5', 'fixed_validation2048_M3',
         'fixed_validation2048_M5', 'fixed_validation2048_M10')
report = {'epoch': EPOCH, 'time': time.time(), 'arms': {}, 'waiting': [],
          'final4096_read': False, 'thresholds_or_training_not_modified': True}
paired_identities = {}
for arm, job in jobs.items():
    base = suite / arm
    submission = json.loads((base / 'root-submission.json').read_text())
    assert submission['job_id'] == job and submission['loss'] == arm
    assert submission['publication_commit'] == '7ac274ac8ff5dea0a12dfd44cfb03fe57491458f'
    path = base / ('evaluation-epoch-%02d.json' % EPOCH)
    if not path.exists():
        report['waiting'].append({'arm': arm, 'job_id': job, 'missing': str(path)})
        continue
    evaluation = json.loads(path.read_text())
    assert evaluation['epoch'] == EPOCH and evaluation['final4096_scored'] is False
    result = {'job_id': job, 'cursor': evaluation['cursor'], 'evaluation_seconds': evaluation['seconds'],
              'producer_metrics': evaluation['metrics'], 'score_spread': {}}
    for role in roles:
        scores = {'raw': [], 'EMA': []}
        targets, identity = [], {}
        prediction = base / ('epoch-%02d-%s-predictions.jsonl.gz' % (EPOCH, role))
        with gzip.open(prediction, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                assert row['qid'] not in identity
                identity[row['qid']] = (row['trajectory_ids'], row['labels'])
                targets.append(row['target_variance'])
                for state in scores:
                    value = row[state + '_predicted_variance']
                    assert math.isfinite(value) and 0 <= value <= .25
                    scores[state].append(value)
        assert len(identity) == 2048
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if role in paired_identities:
            assert paired_identities[role] == digest, (arm, role, 'Mismatch in fixed questions/own trajectories/EM')
        paired_identities[role] = digest
        spread = {'identity_sha256': digest}
        for state, values in scores.items():
            counts = Counter(values)
            spread[state] = {'mean': statistics.mean(values), 'population_std': statistics.pstdev(values),
                             'minimum': min(values), 'maximum': max(values), 'exact_distinct_scores': len(counts),
                             'largest_exact_score_tie_questions': max(counts.values()),
                             'zero_score_questions': counts[0.0], 'maximum_score_questions': counts[.25]}
        gaps = [abs(raw-ema) for raw, ema in zip(scores['raw'], scores['EMA'])]
        mean_target = statistics.mean(targets)
        spread['raw_EMA'] = {'mean_absolute_gap': statistics.mean(gaps), 'maximum_absolute_gap': max(gaps),
                             'exact_equal_questions': sum(gap == 0 for gap in gaps),
                             'weights_not_loaded_or_compared_here': True}
        spread['target'] = {'mean': mean_target, 'population_std': statistics.pstdev(targets)}
        spread['zero_prediction_diagnostic'] = {'variance_MSE': statistics.mean(y*y for y in targets),
                                                'variance_MAE': statistics.mean(abs(y) for y in targets),
                                                'used_for_selection': False}
        spread['constant_mean_prediction_diagnostic'] = {'variance_MSE': statistics.mean((y-mean_target)**2 for y in targets),
                                                         'variance_MAE': statistics.mean(abs(y-mean_target) for y in targets),
                                                         'used_for_selection': False}
        result['score_spread'][role] = spread
    report['arms'][arm] = result
report['status'] = 'SAME_EPOCH_FIXED_INPUT_THREE_ARM_DIAGNOSTICS_COMPLETE' if not report['waiting'] else 'WAITING_FOR_MATCHED_EPOCH_IN_ALL_THREE_ARMS'
report['identity_sha256_by_role'] = paired_identities
print(json.dumps(report))

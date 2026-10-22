"""Summarize already saved MAE diagnostic scores remotely; no final data."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import statistics
import time

EPOCH = 7
base = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1/mae')
submission = json.loads((base / 'root-submission.json').read_text())
assert submission['job_id'] == '749576' and submission['replaces_failed_job'] == '749569'
assert submission['publication_commit'] == '7ac274ac8ff5dea0a12dfd44cfb03fe57491458f'
evaluation = base / ('evaluation-epoch-%02d.json' % EPOCH)
assert evaluation.exists(), 'Wait until this exact epoch evaluation is complete'
report = {'job_id': '749576', 'epoch': EPOCH, 'time': time.time(), 'pools': {}, 'final4096_read': False}
for role in ('fixed_train2048_G5', 'fixed_validation2048_G5', 'fixed_validation2048_M3',
             'fixed_validation2048_M5', 'fixed_validation2048_M10'):
    scores = {'raw': [], 'EMA': []}
    targets = []
    path = base / ('epoch-%02d-%s-predictions.jsonl.gz' % (EPOCH, role))
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            targets.append(row['target_variance'])
            for state in scores:
                value = row[state + '_predicted_variance']
                assert math.isfinite(value) and 0 <= value <= .25
                scores[state].append(value)
    assert len(scores['raw']) == len(scores['EMA']) == 2048
    result = {}
    for state, values in scores.items():
        ordered = sorted(values)
        counts = Counter(values)
        result[state] = {'questions': len(values), 'mean': statistics.mean(values),
                         'population_std': statistics.pstdev(values), 'minimum': ordered[0],
                         'maximum': ordered[-1], 'q05': ordered[int(.05*(len(ordered)-1))],
                         'q95': ordered[int(.95*(len(ordered)-1))], 'exact_distinct_scores': len(counts),
                         'largest_exact_score_tie_questions': max(counts.values()),
                         'zero_score_questions': counts[0.0], 'maximum_score_questions': counts[.25]}
    gaps = [abs(raw-ema) for raw, ema in zip(scores['raw'], scores['EMA'])]
    result['raw_EMA'] = {'mean_absolute_gap': statistics.mean(gaps), 'maximum_absolute_gap': max(gaps),
                         'exact_equal_questions': sum(gap == 0 for gap in gaps),
                         'weights_not_loaded_or_compared_here': True}
    mean_target = statistics.mean(targets)
    result['target'] = {'mean': mean_target, 'population_std': statistics.pstdev(targets)}
    result['zero_prediction_diagnostic'] = {'variance_MSE': statistics.mean(y*y for y in targets), 'variance_MAE': statistics.mean(abs(y) for y in targets), 'used_for_selection': False}
    result['constant_mean_prediction_diagnostic'] = {'variance_MSE': statistics.mean((y-mean_target)**2 for y in targets), 'variance_MAE': statistics.mean(abs(y-mean_target) for y in targets), 'used_for_selection': False}
    report['pools'][role] = result
print(json.dumps(report))

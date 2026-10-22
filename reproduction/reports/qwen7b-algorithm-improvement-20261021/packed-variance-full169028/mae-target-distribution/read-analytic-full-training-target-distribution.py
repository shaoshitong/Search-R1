"""Read only accepted train-group metadata; exactly integrate uniform own-M sampling."""
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
from math import comb
from pathlib import Path
import time

PATH = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1/fixture/inputs/train-groups.json')
before = PATH.stat()
raw = PATH.read_bytes()
groups = json.loads(raw)
assert isinstance(groups, dict) and len(groups) == 169028
counts = Counter()
for qid, group in groups.items():
    labels = group['labels']
    assert len(labels) == len(group['trajectory_ids']) == 10
    assert len(set(group['trajectory_ids'])) == 10
    assert all(type(y) is int and y in (0, 1) for y in labels)
    assert group['qid'] == qid
    counts[sum(labels)] += 1
assert sum(counts.values()) == 169028
after = PATH.stat()
assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
N = len(groups)

def median_interval(distribution):
    ordered = sorted(distribution)
    cumulative = F(0)
    for index, value in enumerate(ordered):
        cumulative += distribution[value]
        if cumulative > F(1, 2):
            return value, value
        if cumulative == F(1, 2):
            return value, ordered[index+1]
    raise AssertionError('Normalized finite distribution required')

def encode(value):
    return {'exact': str(value), 'value': float(value)}

def summarize(distribution):
    assert sum(distribution.values(), F(0)) == 1
    mean = sum((v*p for v,p in distribution.items()), F(0))
    lo, hi = median_interval(distribution)
    median = (lo+hi)/2
    baselines = {}
    for name, prediction in [('zero', F(0)), ('mean', mean), ('median', median)]:
        mae = sum((abs(v-prediction)*p for v,p in distribution.items()), F(0))
        mse = sum(((v-prediction)**2*p for v,p in distribution.items()), F(0))
        baselines[name] = {'constant_variance_prediction': encode(prediction),
                           'expected_variance_MAE': encode(mae),
                           'expected_variance_MSE': encode(mse),
                           'constant_actual_4variance_prediction': encode(4*prediction),
                           'expected_actual_4variance_MAE': encode(4*mae),
                           'expected_actual_4variance_MSE': encode(16*mse)}
    return {'P_zero': encode(distribution.get(F(0), F(0))),
            'P_positive_Mixed': encode(1-distribution.get(F(0), F(0))),
            'mean_variance': encode(mean), 'median_variance_interval': [encode(lo),encode(hi)],
            'mean_actual_4variance_target': encode(4*mean),
            'median_actual_4variance_target_interval': [encode(4*lo),encode(4*hi)],
            'constant_prediction_baselines_diagnostic_only': baselines,
            'variance_distribution': [{'variance': encode(v), 'actual_target_4variance': encode(4*v),
                                       'probability': encode(p)} for v,p in sorted(distribution.items())]}

mixture = defaultdict(F)
by_M = {}
for M in range(3,11):
    dist = defaultdict(F)
    for K, count in counts.items():
        for j in range(max(0,M-(10-K)), min(K,M)+1):
            probability = F(count, N)*F(comb(K,j)*comb(10-K,M-j), comb(10,M))
            variance = F(j*(M-j),M*M)
            dist[variance] += probability
            mixture[variance] += probability/8
    by_M[str(M)] = summarize(dist)
result = {'status':'ACTUAL_FULL169028_OWN_G10_HYPERGEOMETRIC_TARGET_DISTRIBUTION',
          'time':time.time(), 'source':{'path':str(PATH),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},
          'questions':N, 'trajectories_per_question':10,
          'K_correct_own_G10_histogram':{str(K):counts[K] for K in range(11)},
          'G10_homogeneous_questions':counts[0]+counts[10],
          'G10_Mixed_questions':N-counts[0]-counts[10],
          'question_probability':'uniform over all 169028 questions',
          'M_probability':'independent uniform over integers 3..10',
          'within_question_sampling':'uniform M-subset without replacement of own10 trajectories, independent of EM',
          'analytic_probability':'count(K)/169028 * 1/8 * C(K,j)*C(10-K,M-j)/C(10,M)',
          'variance_definition':'j/M * (1-j/M)', 'actual_training_target':'4*population_EM_variance',
          'exact_rational_calculation':True, 'mixture':summarize(mixture),'conditional_on_M':by_M,
          'not_claimed':'Marginal constant optima do not prove conditional feature-based predictions must collapse.',
          'no_CP_or_features_read':True,'no_training_or_Slurm_changes':True,'final4096_read':False}
print(json.dumps(result,sort_keys=True))

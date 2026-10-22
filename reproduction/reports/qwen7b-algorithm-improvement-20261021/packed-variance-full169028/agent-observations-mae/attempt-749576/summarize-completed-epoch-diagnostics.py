"""Condense saved fixed-input diagnostics; no remote reads, forward passes or threshold fitting."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
rows = []
sources = []
for path in sorted(HERE.glob('actual-same-epoch-three-arm-[0-9][0-9].json')):
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get('status') != 'SAME_EPOCH_FIXED_INPUT_THREE_ARM_DIAGNOSTICS_COMPLETE':
        continue
    sources.append({'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    for arm, result in report['arms'].items():
        for role, states in result['producer_metrics'].items():
            spread = result['score_spread'][role]
            for state in ('raw', 'EMA'):
                if state not in states:
                    continue
                metrics = states[state]
                classification = metrics.get('predeclared_G5_classification_diagnostic', {})
                selection = metrics.get('Top512', {})
                scores = spread[state]
                rows.append({
                    'epoch': report['epoch'], 'arm': arm, 'state': state,
                    'job_id': result['job_id'], 'cursor': result['cursor'], 'pool_role': role,
                    'input_identity_sha256': spread['identity_sha256'],
                    'questions': metrics['questions'],
                    'true_Mixed_fraction': metrics.get('true_Mixed_fraction'),
                    'variance_MSE': metrics['variance_MSE'], 'variance_MAE': metrics['variance_MAE'],
                    'Pearson': metrics.get('variance_Pearson'), 'Spearman': metrics.get('variance_Spearman'),
                    'Top512_Mixed': selection.get('Mixed'), 'Top512_Mixed_fraction': selection.get('Mixed_fraction'),
                    'diagnostic_BA': classification.get('balanced_accuracy'),
                    'diagnostic_Mixed_recall': classification.get('Mixed_recall'),
                    'diagnostic_nonMixed_FPR': classification.get('nonMixed_false_positive_rate'),
                    'diagnostic_threshold': classification.get('variance_threshold'),
                    'score_mean': scores['mean'], 'score_std': scores['population_std'],
                    'score_min': scores['minimum'], 'score_max': scores['maximum'],
                    'exact_distinct_scores': scores['exact_distinct_scores'],
                    'zero_prediction_diagnostic': spread['zero_prediction_diagnostic'],
                    'constant_mean_prediction_diagnostic': spread['constant_mean_prediction_diagnostic'],
                    'source': str(path)
                })
result = {'status': 'SAVED_FIXED_INPUT_DIAGNOSTICS_ONLY',
          'completed_compared_epochs': sorted({row['epoch'] for row in rows}),
          'threshold_fitted': False, 'candidate_selected': False, 'final4096_read': False,
          'checkpoint_CPU_acceptance_not_implied': True,
          'rows': rows, 'sources': sources}
(HERE / 'actual-all-completed-epoch-metrics.json').write_text(json.dumps(result, sort_keys=True) + '\n')
print(json.dumps({'epochs': result['completed_compared_epochs'], 'rows': len(rows)}))

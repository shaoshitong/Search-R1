"""Read-only terminal own-EM/score audit for the single all15504 DP run.

Pure CPU metadata/probability recomputation. Reuses published scalar metrics,
not any calibration/grid execution. No model, optimizer, feature payload or
untouched final is opened. Final CP file SHA is streamed independently.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import time
from . import e2_e4_fixed_ensemble as n

BASE = n.A / 'g20-exact-uniform15504-dp-v1'
REFERENCE = n.A / 'g20-matched6528-v1/random'
CONFIG_SHA = 'af679a2a8ff93642a4573bb297f23ff954cca38b6749f0c7ff10c111e2cce4f4'
E4 = n.B / 'frozen-validation-ema3-20261021/evaluation-output/E4/predictions.jsonl.gz'
E4_SHA = '2c76bdb5d0d34ddc2bf8966f8cde639b84d772dcbc68ef7c42f094a6b4498d5d'


def own(evaluation):
    if evaluation['role'] != 'validation':
        raise ValueError('Only designated diagnostic/recipe-selection validation allowed')
    groups = n.read(n.checked(evaluation['bank']))[evaluation['split']]
    plan = n.read(n.checked(evaluation['fixed_g5']))
    plan = plan.get('fixed_G5', plan)
    if (plan['summary']['role'] != 'validation' or plan['summary']['labels_consulted'] is not False
            or n.digest(plan['groups']) != plan['summary']['selected_g5_sha256']):
        raise ValueError('Label-blind frozen G5 identity changed')
    by, result, seen = {g['qid']: g for g in groups}, [], set()
    if len(groups) != len(by) or {g['qid'] for g in plan['groups']} != set(by):
        raise ValueError('Frozen evaluation question membership differs')
    for selected in plan['groups']:
        g = by[selected['qid']]
        labels = dict(zip(g['trajectory_ids'], g['labels']))
        tids = selected['trajectory_ids']
        if (len(tids) != 5 or len(set(tids)) != 5 or seen.intersection(tids)
                or len(labels) != len(g['labels'])
                or any(type(x) is not int or x not in (0, 1) for x in labels.values())):
            raise ValueError('Five authentic own trajectories/strict EM required')
        seen.update(tids)
        result.append(dict(qid=g['qid'], trajectory_ids=tids, strict_em=[labels[t] for t in tids]))
    if len(result) not in (512, 2048):
        raise ValueError('Only original512 and exact fixed2048 expected')
    return result, plan['summary']['selected_g5_sha256']


def prediction(reference, expected, declared=None):
    path = n.checked(reference)
    with gzip.open(path, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != len(expected) or len({r['qid'] for r in rows}) != len(rows):
        raise ValueError('Prediction own identities/count differ')
    error = 0.
    for row, gold in zip(rows, expected):
        if any(row[key] != gold[key] for key in ('qid', 'trajectory_ids', 'strict_em')):
            raise ValueError('Prediction IDs/EM differ from independently rejoined bank/fixed plan')
        ps, q = row['p_j'], row['predicted_mixed']
        if (len(ps) != 5 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in ps)
                or not math.isfinite(q) or not 0 <= q <= 1):
            raise ValueError('Finite probability rows required')
        error = max(error, abs(n.qscore(ps) - q))
        row['score'] = q
    if error >= 1e-6:
        raise ValueError('Label-blind Bernoulli G5 q formula differs')
    result = n.summary(rows)
    if declared is not None:
        for role in ('pool', 'selected512'):
            for key in ('questions', 'mixed_questions', 'mixed_fraction', 'k_histogram', 'confusion',
                    'mixed_recall', 'nonmixed_recall', 'mixed_balanced_accuracy', 'mixed_roc_auc'):
                value, want = result[role][key], declared[role][key]
                if abs(value - want) >= 1e-12 if isinstance(value, float) else value != want:
                    raise ValueError('Independent terminal metric differs: ' + role + ':' + key)
            for label in ('allwrong', 'mixed', 'allcorrect'):
                for key, value in result[role]['classwise'][label].items():
                    want = declared[role]['classwise'][label][key]
                    if abs(value - want) >= 1e-12 if isinstance(value, float) else value != want:
                        raise ValueError('Independent terminal classwise metric differs')
    result.update(predictions=n.proof(path), q_formula_max_absolute_error=error,
        own_EM_and_frozen_ID_order_exact=True,
        ordered_own_trajectory_ids_sha256=n.digest([r['trajectory_ids'] for r in rows]))
    return result


def run():
    config = n.read(BASE / 'config.json')
    done = n.read(BASE / 'complete.json')
    if (n.proof(BASE / 'config.json')['sha256'] != CONFIG_SHA
            or done['status'] != 'REAL_G20_QUESTION_WEIGHTED_PROBE_BUDGET_AND_VALIDATION_COMPLETE'
            or done['cursor'] != 136 or done['steps'] != 10600 or done['actor_updates'] != 0
            or done['optimizer_reset'] or done['config_sha256'] != CONFIG_SHA
            or done['checkpoint'] != n.read(BASE / 'latest.json')):
        raise ValueError('Actual complete136/fullAdam10600 accepted state required')
    cp = done['checkpoint']
    n.checked({k: cp[k] for k in ('path', 'bytes', 'sha256')})
    first = n.read(BASE / 'independent-cp-000001.json')
    if first['status'] != 'EXACT_G20_DP_INDEPENDENT_FULL_SHA_AND_CPU_SIX_COMPONENT_RELOAD_PASSED':
        raise ValueError('Actual independent first CP six-component qualification required')
    if n.read(BASE / 'next-update-replay.json')['status'] != 'PASSED':
        raise ValueError('Actual first GPU fullstate replay missing')
    if (config['actual_grouping']['combinations_per_question'] != 15504
            or config['source_group_size'] != 20 or config['global_questions'] != 48
            or config['update_budget'] != 136):
        raise ValueError('Actual G20 uniform all15504 matched protocol differs')
    if (BASE / 'failure.json').exists() or (BASE / 'interrupted.json').exists():
        raise ValueError('Terminal error/interruption must be resolved explicitly')
    old_config, old_done = n.read(REFERENCE / 'config.json'), n.read(REFERENCE / 'complete.json')
    if (n.proof(REFERENCE / 'config.json') != config['reference_config']
            or old_done['config_sha256'] != config['reference_config']['sha256']
            or old_done['cursor'] != 136 or old_done['steps'] != 10600
            or old_done['actor_updates'] != 0 or old_done['optimizer_reset']):
        raise ValueError('Original completed matched random4 reference lineage differs')
    for key in ('source_checkpoint', 'plan', 'g20_complete', 'evaluations',
            'global_questions', 'update_budget', 'micro_trajectories', 'fit_questions', 'fit_real_trajectories'):
        if config[key] != old_config[key]:
            raise ValueError('Matched random4 protocol reference changed: ' + key)
    results, fixed, fixed_sha = {}, None, None
    for evaluation in config['evaluations']:
        expected, freeze = own(evaluation)
        actual = done['metrics'][evaluation['name']]
        results[evaluation['name']] = prediction(actual['predictions'], expected, actual)
        if len(expected) == 2048:
            if fixed is not None:
                raise ValueError('One exact designated2048 selection pool required')
            fixed, fixed_sha = expected, freeze
    if fixed is None or fixed_sha != n.FIXED:
        raise ValueError('Same actual native fixed2048G5 identity required')
    fresh = next(value for value in results.values() if value['pool']['questions'] == 2048)
    e4 = prediction(dict(path=str(E4), bytes=155083, sha256=E4_SHA), fixed)
    random4_metrics = next(value for value in old_done['metrics'].values() if value['pool']['questions'] == 2048)
    random4 = prediction(random4_metrics['predictions'], fixed, random4_metrics)
    comparisons = {}
    for name, baseline in (('same_native_E4', e4), ('same_E1_G20_random4_B48x136', random4)):
        ba = fresh['pool']['mixed_balanced_accuracy'] - baseline['pool']['mixed_balanced_accuracy']
        top = fresh['selected512']['mixed_questions'] - baseline['selected512']['mixed_questions']
        comparisons[name] = dict(BA_delta=ba, Top512_Mixed_count_delta=top,
            AUC_delta=fresh['pool']['mixed_roc_auc'] - baseline['pool']['mixed_roc_auc'],
            both_BA_and_Top512_strictly_improved=ba > 0 and top > 0,
            exact_frozen_G5_and_native_backend_same=True,
            equal_training_budget_claim=name != 'same_native_E4')
    result = dict(status='ACTUAL_G20_ALL15504_DP_FINAL_OWN_EM_Q_METRICS_AND_NATIVE_BASELINES_AUDIT_PASS',
        config=n.proof(BASE / 'config.json'), complete=n.proof(BASE / 'complete.json'),
        checkpoint=cp, independent_CP1=n.proof(BASE / 'independent-cp-000001.json'),
        evaluations=results, native_E4=e4, matched_G20_random4=random4, comparisons=comparisons,
        freeze_sha256=fixed_sha, gpu_memory=n.read(BASE / 'gpu-memory.json'),
        model_or_optimizer_loaded=False, feature_payload_opened=False, final_accessed=False,
        full_final_checkpoint_SHA_fresh=True, actor_updates=0, epoch=time.time())
    n.atomic(BASE / 'independent-final-audit.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)

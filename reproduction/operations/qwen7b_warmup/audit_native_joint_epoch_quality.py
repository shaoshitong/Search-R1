"""CPU-only receipt/prediction audit; never loads models or scores final4096.

Read existing completed fixed512/2048 epoch outputs on HPC3. Recompute the
Mixed confusion and label-blind Top512, and expose losses by true class so a
lower aggregate loss cannot hide nonMixed-only improvement. Incomplete epochs
are explicitly pending. No output is written into the active training tree.
"""
import argparse
import gzip
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from functools import reduce
from operator import mul

BASE = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1')
CONFIG_SHA = '5aaa63c9348f410357be92ce491ba6d18286d618ab86dae6e63f193f2b4367b6'
FIXED_G5_SHA = 'c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'


def proof(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def checked_prediction(ref, folder):
    path = Path(ref['path'])
    if path.parent != folder or path.resolve() != path or proof(path) != ref:
        raise ValueError('Completed prediction path/size/SHA mismatch')
    return path


def summarize(rows, expected):
    if len(rows) != expected or len({r['qid'] for r in rows}) != expected:
        raise ValueError('Missing/duplicate heldout question')
    classes = {name: dict(n=0, predicted_Mixed=0, trajectory_BCE_sum=0.,
                         Mixed_BCE_sum=0., p_range_sum=0.,
                         almost_equal_p_questions=0)
               for name in ('allwrong', 'Mixed', 'allcorrect')}
    identities = []
    all_ids = set()
    for row in rows:
        ys, ps, ids, q = row['strict_em'], row['p_j'], row['trajectory_ids'], row['predicted_mixed']
        if (len(ys) != 5 or len(ps) != 5 or len(ids) != 5 or len(set(ids)) != 5
                or all_ids.intersection(ids)
                or any(type(y) is not int or y not in (0, 1) for y in ys)
                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in ps)
                or not math.isfinite(q) or not 0 <= q <= 1):
            raise ValueError('Distinct own G5, strict EM and finite scores required')
        all_ids.update(ids)
        identities.append([row['qid'], ids, ys])
        recomputed_q = 1 - reduce(mul, ps, 1.) - reduce(mul, (1 - p for p in ps), 1.)
        if abs(recomputed_q - q) > 1e-5:
            raise ValueError('Published Mixed probability differs from own p_j')
        k = sum(ys)
        name = 'allwrong' if k == 0 else 'allcorrect' if k == 5 else 'Mixed'
        c = classes[name]
        c['n'] += 1
        c['predicted_Mixed'] += int(q >= .5)
        clip = lambda p: min(1 - 1e-12, max(1e-12, p))
        c['trajectory_BCE_sum'] += sum(-math.log(clip(p) if y else 1 - clip(p)) for p, y in zip(ps, ys)) / 5
        c['Mixed_BCE_sum'] += -math.log(clip(q) if name == 'Mixed' else 1 - clip(q))
        spread = max(ps) - min(ps)
        c['p_range_sum'] += spread
        c['almost_equal_p_questions'] += int(spread < 1e-4)
    tp = classes['Mixed']['predicted_Mixed']
    fn = classes['Mixed']['n'] - tp
    fp = sum(classes[n]['predicted_Mixed'] for n in ('allwrong', 'allcorrect'))
    tn = expected - tp - fn - fp
    if tp + fn == 0 or tn + fp == 0:
        raise ValueError('Balanced accuracy requires both real classes')
    chosen = sorted(rows, key=lambda r: (-r['predicted_mixed'], str(r['qid'])))[:512]
    selected_mixed = sum(0 < sum(r['strict_em']) < 5 for r in chosen)
    pool_histogram = [sum(sum(r['strict_em']) == k for r in rows) for k in range(6)]
    selected_histogram = [sum(sum(r['strict_em']) == k for r in chosen) for k in range(6)]
    assert sum(selected_histogram) == 512 and sum(selected_histogram[1:5]) == selected_mixed
    cutoff = chosen[-1]['predicted_mixed']
    # Inspect class composition only after the label-blind ordering is fixed.
    # This makes a rise in allcorrect/allwrong contamination visible directly.
    selection = dict(
        strict_EM_K_histogram=selected_histogram,
        true_classes=dict(allwrong=selected_histogram[0], Mixed=selected_mixed,
                          allcorrect=selected_histogram[5]),
        Mixed_fraction=selected_mixed / 512,
        pool_Mixed_fraction=(tp + fn) / expected,
        Mixed_enrichment_pp=100 * (selected_mixed / 512 - (tp + fn) / expected),
        score_cutoff=cutoff,
        pool_questions_tied_at_cutoff=sum(r['predicted_mixed'] == cutoff for r in rows),
        selected_questions_tied_at_cutoff=sum(r['predicted_mixed'] == cutoff for r in chosen),
        deterministic_tie_break='string_qid_ascending',
        selected_qids_sha256=hashlib.sha256(json.dumps([r['qid'] for r in chosen]).encode()).hexdigest(),
        labels_used_to_select=False)
    for c in classes.values():
        n = c['n']
        for source, target in [('trajectory_BCE_sum', 'trajectory_BCE'),
                               ('Mixed_BCE_sum', 'Mixed_BCE'),
                               ('p_range_sum', 'mean_p_range')]:
            c[target] = c.pop(source) / n if n else None
    return dict(questions=expected, confusion=dict(tp=tp, fn=fn, tn=tn, fp=fp),
                Mixed_balanced_accuracy=.5 * (tp / (tp + fn) + tn / (tn + fp)),
                Mixed_recall=tp / (tp + fn), nonMixed_recall=tn / (tn + fp),
                selected512_Mixed=selected_mixed, selected512_Mixed_fraction=selected_mixed / 512,
                pool_strict_EM_K_histogram=pool_histogram,
                selected512_composition=selection,
                true_class_diagnostics=classes,
                own_G5_and_EM_identity_sha256=hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest(),
                threshold=.5, labels_used_for_selection=False,
                probability_BCE_clip=1e-12)


def run():
    if platform.system() != 'Linux' or not str(BASE).startswith('/data/user/'):
        raise ValueError('Actual prediction payloads remain on HPC3')
    config = proof(BASE / 'config.json')
    if config['sha256'] != CONFIG_SHA:
        raise ValueError('Immutable joint training config changed')
    results, missing, identity = [], [], {}
    for epoch in range(5):
        folder = BASE / 'evaluations' / ('epoch-%02d' % epoch)
        path = folder / 'complete.json'
        if not path.exists():
            missing.append(epoch)
            continue
        receipt_ref = proof(path)
        receipt = json.loads(path.read_text())
        if (receipt['epoch'] != epoch or receipt['cursor'] != 3882 * epoch
                or receipt['checkpoint']['cursor'] != receipt['cursor']
                or receipt['checkpoint']['steps'] != 14418 + receipt['cursor']
                or receipt['config_sha256'] != CONFIG_SHA
                or not receipt['training_components_unchanged']
                or not receipt['native_buffers_and_RNG_restored']
                or receipt['Count5120_scored'] is not False
                or receipt['final4096_scored'] is not False
                or receipt['validation_proof']['fixed_G5_sha256'] != FIXED_G5_SHA):
            raise ValueError('Completed epoch checkpoint/evaluation binding differs')
        states = {}
        for state in ('raw', 'EMA'):
            states[state] = {}
            for split, n in [('diagnostic512', 512), ('fixed_selection2048', 2048)]:
                stored = receipt['metrics'][state][split]
                pp = checked_prediction(stored['predictions'], folder)
                with gzip.open(pp, 'rt') as stream:
                    actual = summarize([json.loads(line) for line in stream], n)
                key = split
                fingerprint = actual['own_G5_and_EM_identity_sha256']
                if key in identity and identity[key] != fingerprint:
                    raise ValueError('Heldout identities or own EM changed across epoch/state')
                identity[key] = fingerprint
                pool = stored['pool']
                if (pool['confusion'] != actual['confusion']
                        or abs(pool['mixed_balanced_accuracy'] - actual['Mixed_balanced_accuracy']) > 1e-10
                        or stored['selected512']['mixed_questions'] != actual['selected512_Mixed']
                        or stored['labels_used_to_select'] is not False):
                    raise ValueError('Independent class/selection recomputation differs from saved metrics')
                states[state][split] = actual
        if proof(path) != receipt_ref:
            raise ValueError('Completed epoch receipt changed while auditing')
        results.append(dict(epoch=epoch, receipt=receipt_ref, checkpoint=receipt['checkpoint'], states=states))
    baseline = next((r for r in results if r['epoch'] == 0), None)
    if baseline is None:
        raise ValueError('Actual E0 baseline required')
    for r in results:
        for state in ('raw', 'EMA'):
            v = r['states'][state]['fixed_selection2048']
            b = baseline['states'][state]['fixed_selection2048']
            v['delta_Mixed_BA_vs_same_state_E0'] = v['Mixed_balanced_accuracy'] - b['Mixed_balanced_accuracy']
            v['delta_Top512_Mixed_vs_same_state_E0'] = v['selected512_Mixed'] - b['selected512_Mixed']
    # Preserve the already recorded scientific gate; no threshold or label fit.
    gate = dict(Mixed_BA_strictly_above=.5625049686633887, Top512_Mixed_strictly_above=196,
                BA_equality_roundoff_tolerance=1e-12,
                provenance='root-data-scaling-priority-20261021.json existing goal_gate',
                fixed_threshold=.5, both_metrics_required=True)
    candidates = []
    for r in results:
        if r['epoch'] == 0:
            continue
        for state in ('raw', 'EMA'):
            v = r['states'][state]['fixed_selection2048']
            if (v['Mixed_balanced_accuracy'] > gate['Mixed_BA_strictly_above'] + gate['BA_equality_roundoff_tolerance']
                    and v['selected512_Mixed'] > gate['Top512_Mixed_strictly_above']):
                candidates.append(dict(epoch=r['epoch'], state=state,
                                       Mixed_BA=v['Mixed_balanced_accuracy'],
                                       Top512_Mixed=v['selected512_Mixed']))
    return dict(status='ACTUAL_COMPLETED_EPOCH_CPU_CLASS_AND_TOP512_AUDIT', epoch=time.time(),
                config=config, completed_epochs=[r['epoch'] for r in results], pending_epochs=missing,
                epoch_results=results, posttraining_epoch_available=any(r['epoch'] > 0 for r in results),
                fixed_threshold=.5, no_model_tensor_or_GPU_load=True, no_new_predictions=True,
                class_BCE_recomputed_from_recorded_probabilities_with_clip=True,
                scientific_selection_gate=gate, qualifying_posttraining_candidates=candidates,
                untouched_confirmation_ready=not missing and bool(candidates),
                final4096_read=False, current_training_changed=False,
                independent_checkpoint_tensor_reload_in_this_metadata_audit=False)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(), allow_nan=False))

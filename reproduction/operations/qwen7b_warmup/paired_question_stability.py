"""Descriptive paired question bootstrap; no model, fitting or GPU operation.

Input rows must already have independently audited own-EM/G5 identity. The CLI
also binds their full files and audit/backend receipts. All score thresholds,
fractions, strata and resampling choices are fixed, not selected from results.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random

SEED = 9137
REPLICATES = 2000
ROLE = 'already_viewed_recipe_selection_validation'
REMOTE_ROOT = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def freeze_strata(qids):
    """Uses only unique question identities, never labels or scores."""
    if len(qids) != len(set(qids)) or len(qids) % 4:
        raise ValueError('Unique question IDs and four equal strata required')
    ordered = sorted(qids, key=lambda q: (hashlib.sha256((str(SEED) + '\x1f' + str(q)).encode()).hexdigest(), str(q)))
    strata = [ordered[i::4] for i in range(4)]
    return dict(seed=SEED, rule='sha256(seed + unit_separator + str(qid)); sorted; round_robin_four',
                labels_consulted=False, scores_consulted=False, strata=strata,
                strata_sha256=digest(strata), all_qids_sha256=digest(sorted(qids, key=str)))


def _validated(rows):
    by = {}
    for row in rows:
        qid = row['qid']; labels = row['strict_em']; tids = row['trajectory_ids']; ps = row['p_j']; q = row['predicted_mixed']
        if qid in by or len(tids) != 5 or len(set(tids)) != 5 or len(labels) != 5 or len(ps) != 5:
            raise ValueError('Unique question and five authentic trajectories required')
        if any(type(v) is not int or v not in (0, 1) for v in labels):
            raise ValueError('Own binary strict EM required')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in ps) or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError('Finite probabilities required')
        correct = wrong = 1.
        for p in ps:
            correct *= p; wrong *= 1. - p
        if abs(q - (1. - correct - wrong)) > 2e-6:
            raise ValueError('Stored label-blind G5 q formula differs')
        by[qid] = dict(qid=qid, trajectory_ids=tids, strict_em=labels, score=q, mixed=0 < sum(labels) < 5)
    return by


def metrics(rows):
    n = len(rows); positives = sum(r['mixed'] for r in rows); negatives = n - positives
    if not n or not positives or not negatives:
        raise ValueError('Both Mixed and NonMixed required for balanced accuracy')
    tp = sum(r['mixed'] and r['score'] >= .5 for r in rows)
    tn = sum(not r['mixed'] and r['score'] < .5 for r in rows)
    take = (n + 3) // 4
    selected = sorted(rows, key=lambda r: (-r['score'], str(r['qid'])))[:take]
    count = sum(r['mixed'] for r in selected)
    return dict(questions=n, mixed_questions=positives, threshold=.5, selected_questions=take,
                mixed_balanced_accuracy=.5 * (tp / positives + tn / negatives),
                mixed_recall=tp / positives, nonmixed_recall=tn / negatives,
                top_quarter_mixed_count=count, top_quarter_mixed_fraction=count / take)


def _quantile(values, p):
    values = sorted(values); pos = p * (len(values) - 1); low = int(pos); high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def _deltas(a, b):
    return {key: a[key] - b[key] for key in
            ('mixed_balanced_accuracy', 'top_quarter_mixed_count', 'top_quarter_mixed_fraction')}


def _scope(by_models, qids, scope, replicates):
    aligned = {name: [by[qid] for qid in qids] for name, by in by_models.items()}
    point = {name: metrics(rows) for name, rows in aligned.items()}
    # The exact same resampled original-question indices are used for all models.
    rng = random.Random(SEED + int(hashlib.sha256(scope.encode()).hexdigest()[:8], 16))
    samples = {name: {key: [] for key in _deltas(point['candidate'], point[name])}
               for name in ('E2', 'E4')}
    skipped = 0
    for _ in range(replicates):
        indices = [rng.randrange(len(qids)) for unused in qids]
        try:
            result = {name: metrics([rows[i] for i in indices]) for name, rows in aligned.items()}
        except ValueError:
            skipped += 1; continue
        for name in samples:
            for key, value in _deltas(result['candidate'], result[name]).items():
                samples[name][key].append(value)
    if replicates and replicates == skipped:
        raise ValueError('No replicate contains both classes')
    pairs = {}
    for name, columns in samples.items():
        pairs[name] = dict(candidate_minus_baseline=_deltas(point['candidate'], point[name]),
            percentile_95_CI={key: [_quantile(values, .025), _quantile(values, .975)]
                              for key, values in columns.items()} if replicates else {},
            paired_question_resamples=True)
    return dict(qids_sha256=digest(qids), models=point, comparisons=pairs,
                bootstrap_replicates_requested=replicates, bootstrap_replicates_used=replicates-skipped,
                single_class_replicates_skipped=skipped)


def analyse(candidate, E2, E4, replicates=REPLICATES, expected_questions=2048):
    if type(replicates) is not int or not 0 <= replicates <= REPLICATES:
        raise ValueError('Finite, predeclared bootstrap budget required')
    # Freeze identity-only membership before inspecting any probabilities/EM.
    qids = sorted([row['qid'] for row in candidate], key=str)
    frozen = freeze_strata(qids)
    if len(qids) != expected_questions:
        raise ValueError('Question count differs from the explicitly bound pool')
    by_models = {name: _validated(rows) for name, rows in [('candidate', candidate), ('E2', E2), ('E4', E4)]}
    for name in ('E2', 'E4'):
        if set(by_models[name]) != set(qids):
            raise ValueError('Paired question membership differs')
        for qid in qids:
            if any(by_models[name][qid][key] != by_models['candidate'][qid][key] for key in ('trajectory_ids', 'strict_em')):
                raise ValueError('Own trajectories/EM differ across paired models')
    scopes = {'all_questions': _scope(by_models, qids, 'all_questions', replicates)}
    for i, subset in enumerate(frozen['strata']):
        scopes['hash_stratum_%d' % i] = _scope(by_models, subset, 'hash_stratum_%d' % i, replicates)
    return dict(status='DESCRIPTIVE_PAIRED_QUESTION_STABILITY_COMPLETE', seed=SEED,
                strata=frozen, selection_fraction=.25, threshold=.5, scopes=scopes,
                confidence_interval='pointwise percentile 2.5/97.5; descriptive, not multiplicity-corrected',
                question_resampling_with_replacement=True, selection_recomputed_per_resample=True,
                parameters_fitted=False, threshold_selected=False, candidate_selected=False,
                external_final_read=False, generalization_claim=False)


def checked(ref):
    path = Path(ref['path']); h = hashlib.sha256(); size = 0
    if not path.is_absolute() or REMOTE_ROOT not in path.parents or '..' in path.parts:
        raise ValueError('CLI probability files and receipts must remain in the bound remote experiment')
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data); size += len(data)
    if size != ref['bytes'] or h.hexdigest() != ref['sha256']:
        raise ValueError('Bound input file SHA/bytes differs')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--manifest', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); manifest = json.loads(Path(args.manifest).read_text())
    # CPU-only finite worker; no libraries with model/CUDA initialization imported.
    import resource
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    if manifest['role'] != ROLE or manifest['questions'] != 2048 or manifest['bootstrap_replicates'] != REPLICATES:
        raise ValueError('Only the already-viewed fixed2048 recipe-selection pool is allowed')
    if set(manifest['predictions']) != {'candidate', 'E2', 'E4'}:
        raise ValueError('Exactly one candidate and both same-HF baseline predictions required')
    # The independent audit owns bank/own-EM/backend provenance. Bind its actual
    # file, not a human assertion, and preserve it in the returned provenance.
    checked(manifest['independent_identity_audit']); checked(manifest['same_HF_backend_receipt'])
    rows = {}
    for name, ref in manifest['predictions'].items():
        with gzip.open(checked(ref), 'rt') as stream:
            rows[name] = [json.loads(line) for line in stream]
    result = analyse(**rows); result['source_manifest'] = manifest
    result['source_manifest_sha256'] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    path = Path(args.output); temporary = path.with_name(path.name + '.tmp')
    if path.exists():
        raise ValueError('Existing descriptive audit preserved; use a separate output')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n'); temporary.replace(path)
    print(json.dumps(dict(status=result['status'], output=str(path), aggregate_only=True)))


if __name__ == '__main__':
    main()

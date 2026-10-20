"""One unseen G5 bank; four EMA checkpoints; predicted top512, then true EM.

Scoring uses Prefix5 only. The true labels are joined after the ranking and
never replace a probe selection with an oracle Mixed selection.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time


REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def mixed_probability(p):
    assert len(p) == 5 and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
    q = 1 - math.prod(p) - math.prod(1 - value for value in p)
    assert -1e-12 <= q <= 1 + 1e-12
    return max(0., min(1., q))


def select(predictions, count=512):
    assert len({row['qid'] for row in predictions}) == len(predictions)
    # This interface has no label argument or label-based fallback.
    for row in predictions:
        assert set(row) == {'qid', 'trajectory_ids', 'p_j', 'predicted_mixed'}
    return sorted(predictions, key=lambda row: (-row['predicted_mixed'], row['qid']))[:count]


def wilson(k, n):
    z, p = 1.959963984540054, k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [center - half, center + half]


def assess(predictions, truth):
    hist = [0] * 6
    for row in predictions:
        labels = truth[row['qid']]
        assert len(labels) == 5 and all(value in (0, 1) for value in labels)
        hist[sum(labels)] += 1
    n = len(predictions)
    mixed = sum(hist[1:5])
    return dict(questions=n, trajectories=n * 5, K_histogram=hist, true_mixed=mixed,
                true_mixed_fraction=mixed / n, all_wrong_fraction=hist[0] / n,
                all_correct_fraction=hist[5] / n, mixed_fraction_wilson95=wilson(mixed, n),
                trajectory_strict_EM=sum(i * value for i, value in enumerate(hist)) / (5 * n),
                mean_predicted_mixed=sum(row['predicted_mixed'] for row in predictions) / n)


def merge(base):
    config = read(base / 'config.json')
    expected = read(base / 'inputs/questions.json')['rows']
    assert config['evaluation_only'] and config['evaluation_questions'] == len(expected) == 2048
    groups, features, seen, manifests = {}, [], set(), []
    for shard in range(4):
        done = read(base / 'sampling' / ('shard-%02d' % shard) / 'complete.json')
        assert done['run_id'] == config['run_id'] and done['model_revision'] == REVISION
        assert done['questions'] == 512 and done['trajectories'] == 2560 and len(done['batches']) == 4
        for proof_path in done['batches']:
            proof_path = Path(proof_path)
            proof = read(proof_path)
            source_path = base / 'inputs' / ('shard-%02d' % shard) / (proof['batch'] + '.json')
            source = read(source_path)
            assert sha(source_path) == proof['input_sha256'] and proof['run_id'] == config['run_id']
            inputs = {row['qid']: row for row in source['rows']}
            prefix, full = proof['prefix'], proof['full']
            assert prefix['actor_version'] == full['actor_version'] == 'official7b-' + REVISION
            for stage, keys in ((prefix, ('states', 'prefix', 'actor_hidden')), (full, ('full', 'own_em'))):
                for key in keys:
                    assert sha(stage[key]['path']) == stage[key]['sha256']
            hidden = read(prefix['actor_hidden']['path'])
            assert hidden['status'] == 'LOSSLESS_NATIVE_FEATURES_COMPLETE'
            assert hidden['dimension'] == 3584 and hidden['question_max_tokens'] == 273
            assert hidden['feature_layer'] == 26 and hidden['prefix_tokens'] == 20
            for item in hidden['files'].values():
                assert Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256']
            labels = {}
            for line in Path(full['own_em']['path']).open():
                row = json.loads(line)
                assert row['trajectory_id'] not in labels and row['strict_em'] in (0, 1)
                labels[row['trajectory_id']] = row['strict_em']
            own = {}
            with gzip.open(prefix['prefix']['path'], 'rt') as handle:
                for line in handle:
                    row = json.loads(line)
                    qid, tid = row['qid'], row['trajectory_id']
                    assert qid in inputs and tid in labels and tid not in seen
                    assert row['question_hash'] == inputs[qid]['question_hash']
                    assert row['initial_prompt_token_ids'] == inputs[qid]['initial_prompt_token_ids']
                    assert row['split'] == 'selection_validation' and row['feature_layer'] == 26
                    assert row['retained_policy_length'] <= 20
                    seen.add(tid)
                    own.setdefault(qid, []).append((row['response_index'], tid, labels[tid]))
            assert set(own) == set(inputs) and set(labels) == {r[1] for values in own.values() for r in values}
            for qid, values in own.items():
                values.sort()
                assert [r[0] for r in values] == list(range(5)) and qid not in groups
                groups[qid] = dict(qid=qid, data_source='nq', trajectory_ids=[r[1] for r in values],
                                   labels=[r[2] for r in values], question_hash=inputs[qid]['question_hash'])
            features.append(prefix['actor_hidden']['path'])
            manifests.append(dict(path=str(proof_path), sha256=sha(proof_path)))
    ordered = [groups[row['qid']] for row in expected]
    assert len(groups) == 2048 and len(seen) == 10240
    qid_sha = hashlib.sha256(json.dumps([row['qid'] for row in ordered]).encode()).hexdigest()
    assert qid_sha == read(base / 'inputs/manifest.json')['ordered_qid_sha256']
    atomic(base / 'bank/groups.json', dict(selection_validation=ordered))
    atomic(base / 'bank/features.json', features)
    result = dict(status='UNSEEN_2048_G5_NATIVE_FEATURES_OWN_EM_VERIFIED', questions=2048, trajectories=10240,
                  run_id=config['run_id'], model_revision=REVISION, source_dataset_sha256=config['heldout_source_sha256'],
                  ordered_qid_sha256=qid_sha, groups_sha256=sha(base / 'bank/groups.json'),
                  features_sha256=sha(base / 'bank/features.json'), capture_prefix_tokens=20, fit_prefix_tokens=5,
                  feature_layer=26, actor_updates=0, batches=manifests, time=time.time())
    atomic(base / 'bank/complete.json', result)
    print(json.dumps(result), flush=True)


def score(base, micro):
    config, proof = read(base / 'config.json'), read(base / 'bank/complete.json')
    assert proof['status'] == 'UNSEEN_2048_G5_NATIVE_FEATURES_OWN_EM_VERIFIED'
    assert sha(base / 'bank/groups.json') == proof['groups_sha256']
    assert sha(base / 'bank/features.json') == proof['features_sha256']
    os.environ['QWEN7B_QUESTION_MAX_TOKENS'] = '273'
    import torch
    from reproduction.operations.qwen7b_warmup.model import Probe, restore_buffers
    from reproduction.operations.qwen7b_warmup.packed_features import FeatureReader
    from reproduction.operations.qwen7b_warmup.train import model_hash
    training = Path(config['training_base'])
    groups = read(base / 'bank/groups.json')['selection_validation']
    truth = {row['qid']: row['labels'] for row in groups}
    reader = FeatureReader(read(base / 'bank/features.json'))
    assert len(reader.records) == 10240
    model = Probe(config['model_dir'], 273, 5).cuda().eval().requires_grad_(False)
    results = base / 'results'
    results.mkdir(exist_ok=True)
    for epoch in range(1, 5):
        complete = results / ('epoch-%04d.json' % epoch)
        if complete.exists():
            previous = read(complete)
            assert previous['bank_sha256'] == sha(base / 'bank/complete.json')
            assert sha(previous['predictions']['path']) == previous['predictions']['sha256']
            continue
        receipt = read(training / 'probe' / ('epoch-%04d.json' % epoch))
        assert receipt['completed_epoch'] and receipt['epoch'] == epoch and receipt['steps'] == epoch * 4573
        assert receipt['actor_updates'] == 0 and receipt['config_sha256'] == config['training_config_sha256']
        assert sha(receipt['path']) == receipt['sha256']
        saved = torch.load(receipt['path'], map_location='cpu', mmap=True, weights_only=False)
        assert saved['completed_epoch'] and saved['steps'] == receipt['steps']
        assert saved['config_sha256'] == config['training_config_sha256'] and saved['actor_updates'] == 0
        model.load_state_dict(saved['ema'], strict=True)
        restore_buffers(model, saved['ema_buffers'])
        expected_ema_hash = saved['rank_states'][0]['ema_sha256']
        assert model_hash(model) == expected_ema_hash
        del saved
        started, predictions = time.monotonic(), []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for start in range(0, len(groups), micro):
                batch = groups[start:start + micro]
                data = reader.get([tid for row in batch for tid in row['trajectory_ids']], 'cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(data).reshape(len(batch), 5)
                probabilities = torch.sigmoid(logits.float()).cpu().tolist()
                for row, p in zip(batch, probabilities):
                    predictions.append(dict(qid=row['qid'], trajectory_ids=row['trajectory_ids'],
                                            p_j=p, predicted_mixed=mixed_probability(p)))
                atomic(results / 'progress.json', dict(epoch=epoch, completed_questions=len(predictions),
                       target_questions=2048, elapsed=time.monotonic() - started, time=time.time()))
        chosen = select(predictions)
        pool, selected = assess(predictions, truth), assess(chosen, truth)
        destination = results / ('epoch-%04d-predictions.jsonl.gz' % epoch)
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        with gzip.open(temporary, 'wt') as handle:
            for row in predictions:
                handle.write(json.dumps(dict(row, true_EM=truth[row['qid']])) + '\n')
        temporary.replace(destination)
        atomic(results / ('epoch-%04d-selected512.json' % epoch), dict(epoch=epoch,
               selection='predicted_q_descending_then_qid_ascending_no_EM',
               selected=[dict(row, true_EM=truth[row['qid']]) for row in chosen]))
        cutoff = chosen[-1]['predicted_mixed']
        result = dict(status='EMA_UNSEEN2048_PREDICTED_TOP512_SCORED', epoch=epoch,
                      checkpoint=receipt, weight_key='ema', ema_sha256=expected_ema_hash,
                      bank_sha256=sha(base / 'bank/complete.json'), pool=pool, selected512=selected,
                      uplift_percentage_points=100 * (selected['true_mixed_fraction'] - pool['true_mixed_fraction']),
                      cutoff_predicted_q=cutoff, total_tied_at_cutoff=sum(row['predicted_mixed'] == cutoff for row in predictions),
                      predictions=dict(path=str(destination), sha256=sha(destination), bytes=destination.stat().st_size),
                      seconds=time.monotonic() - started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      micro_groups=micro, score_prefix_tokens=5, retained_prefix_tokens=20,
                      no_actor_or_probe_optimizer_updates=True, time=time.time())
        atomic(complete, result)
        print(json.dumps(result), flush=True)
    reader.close()
    report(base)


def report(base):
    config = read(base / 'config.json')
    bank = read(base / 'bank/complete.json')
    assert sha(base / 'bank/groups.json') == bank['groups_sha256']
    assert sha(base / 'bank/features.json') == bank['features_sha256']
    groups = read(base / 'bank/groups.json')['selection_validation']
    truth = {row['qid']: row['labels'] for row in groups}
    by_id = {row['qid']: row for row in groups}
    results, checked, selected_sets = base / 'results', [], {}
    for epoch in range(1, 5):
        receipt = read(results / ('epoch-%04d.json' % epoch))
        assert receipt['epoch'] == epoch and receipt['weight_key'] == 'ema'
        assert receipt['bank_sha256'] == sha(base / 'bank/complete.json')
        assert receipt['score_prefix_tokens'] == 5 and receipt['retained_prefix_tokens'] == 20
        checkpoint = read(Path(config['training_base']) / 'probe' / ('epoch-%04d.json' % epoch))
        assert receipt['checkpoint'] == checkpoint and sha(checkpoint['path']) == checkpoint['sha256']
        assert sha(receipt['predictions']['path']) == receipt['predictions']['sha256']
        predictions = []
        with gzip.open(receipt['predictions']['path'], 'rt') as handle:
            for line in handle:
                row = json.loads(line)
                assert row.pop('true_EM') == truth[row['qid']]
                assert row['trajectory_ids'] == by_id[row['qid']]['trajectory_ids']
                assert abs(mixed_probability(row['p_j']) - row['predicted_mixed']) < 1e-12
                predictions.append(row)
        assert len(predictions) == 2048 and {row['qid'] for row in predictions} == set(truth)
        chosen = select(predictions)
        selected = read(results / ('epoch-%04d-selected512.json' % epoch))['selected']
        assert selected == [dict(row, true_EM=truth[row['qid']]) for row in chosen]
        assert assess(chosen, truth) == receipt['selected512'] and assess(predictions, truth) == receipt['pool']
        selected_sets[epoch] = {row['qid'] for row in chosen}
        checked.append(receipt)
    best = min(checked, key=lambda row: (-row['selected512']['true_mixed'], row['epoch']))
    overlaps = {'%d-%d' % (a, b): len(selected_sets[a] & selected_sets[b]) for a in range(1, 5) for b in range(a + 1, 5)}
    result = dict(status='FOUR_EMA_UNSEEN2048_TOP512_TRUE_MIXED_COMPARISON_VERIFIED',
                  results=checked, selected_question_overlaps=overlaps, recommended_epoch=best['epoch'],
                  recommended_checkpoint=best['checkpoint']['path'], recommended_weight_key='ema',
                  selection='highest_true_Mixed_count_in_predicted_top512_then_earlier_epoch',
                  dataset='independent_NQ_dev_validation_not_reserved_final_QA', time=time.time())
    atomic(results / 'comparison.json', result)
    atomic(results / 'recommended-ema.json', dict(epoch=best['epoch'], checkpoint=best['checkpoint'], weight_key='ema',
           criterion=result['selection'], validation_mixed_count=best['selected512']['true_mixed'], denominator=512))
    lines = ['# Four-checkpoint unseen NQ-dev selection validation', '',
             'Identical2048 questions ×5 authentic trajectories, excluded from warmup and all old dev/final/stage splits.',
             'Each EMA ranks predicted Mixed probability; true EM is used only afterwards. This is checkpoint selection validation.', '',
             '| Epoch | True Mixed@512 | Fraction | Full2048 Mixed | Uplift (pp) |',
             '|---|---:|---:|---:|---:|']
    for row in checked:
        lines.append('| %d | %d/512 | %.4f%% | %.4f%% | %+.4f |' % (row['epoch'], row['selected512']['true_mixed'],
            row['selected512']['true_mixed_fraction'] * 100, row['pool']['true_mixed_fraction'] * 100, row['uplift_percentage_points']))
    lines += ['', 'Recommended EMA epoch: **%d**. Exact ties prefer the earlier checkpoint.' % best['epoch'],
              'Wilson intervals are descriptive; this selection validation is not an unbiased final seven-benchmark QA result.',
              'All p_j, predicted q, same-trajectory EM and selected question identities remain on shared remote storage.']
    (results / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], recommended_epoch=best['epoch'])), flush=True)


def selftest():
    assert mixed_probability([0.] * 5) == mixed_probability([1.] * 5) == 0
    assert mixed_probability([.5] * 5) == .9375
    assert mixed_probability([0., 1., 0., 1., 0.]) == 1
    rows = [dict(qid=qid, trajectory_ids=[qid + str(i) for i in range(5)], p_j=p, predicted_mixed=mixed_probability(p))
            for qid, p in [('z', [.5] * 5), ('a', [.5] * 5), ('b', [.01] * 5)]]
    # High predicted uncertainty may be truly all-wrong. Do not replace it with
    # the truly Mixed lower-ranked question: that would leak the answer labels.
    truth = {'a': [0] * 5, 'z': [1] * 5, 'b': [0, 1, 0, 1, 0]}
    assert [r['qid'] for r in select(rows, 1)] == ['a']
    assert assess(select(rows, 1), truth)['true_mixed'] == 0
    print('MIXED_FORMULA_TIE_ORDER_AND_NO_ORACLE_FALLBACK_CHECK_PASSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('merge', 'score', 'report', 'selftest'), required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--micro-groups', type=int, default=32)
    args = parser.parse_args()
    if args.mode == 'selftest':
        selftest()
    else:
        assert args.base is not None
        if args.mode == 'merge':
            merge(args.base)
        elif args.mode == 'score':
            score(args.base, args.micro_groups)
        else:
            report(args.base)

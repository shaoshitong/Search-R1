"""Read completed periodic probe inputs, actual Adam ledgers and paired metrics.

CPU metadata analysis only: no tensor loading, model forward or training gate.
Stage separately from the immutable training repository.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_summary(scan, outcome_path, actor_version):
    index_rows = rows(Path(scan) / 'index.jsonl')
    index = {row['trajectory_id']: row for row in index_rows}
    outcomes = rows(outcome_path)
    assert len(index) == len(index_rows) == len(outcomes)
    assert {row['trajectory_id'] for row in outcomes} == set(index)
    groups = {}
    for row in outcomes:
        original = index[row['trajectory_id']]
        assert row['qid'] == original['qid'] and row['actor_version'] == actor_version
        assert row['prefix_sha256'] == original['prefix_sha256']
        assert type(row['strict_em']) is int and row['strict_em'] in (0, 1)
        groups.setdefault(str(row['qid']), []).append(row['strict_em'])
    assert all(len(group) == 5 for group in groups.values())
    histogram = Counter(sum(group) for group in groups.values())
    summary = dict(questions=len(groups), trajectories=len(outcomes),
                   correct_count_histogram={str(k): histogram[k] for k in range(6)},
                   mixed_questions=sum(histogram[k] for k in (1, 2, 3, 4)),
                   outcome_sha256=sha(outcome_path), index_sha256=sha(Path(scan) / 'index.jsonl'))
    return summary, set(groups), {row['question_hash'] for row in index_rows}


def inspect(base, outer):
    base = Path(base)
    assert outer % 20 == 0 and 20 <= outer <= 200
    root = base / 'outers' / ('outer-%06d' % outer)
    if not (root / 'complete.json').exists():
        return dict(status='PENDING_JOINT_COMPLETION', outer=outer, training_modified=False)
    joint = read(root / 'complete.json')
    metrics_path = root / 'probe-update/metrics.json'
    metrics = read(metrics_path)
    complete = read(root / 'probe-update/complete.json')
    assert sha(metrics_path) == complete['metrics_sha256']
    expected_before = 100 * (outer // 20 - 1)
    assert metrics['actual_adam_steps_per_rank'] == [expected_before + 100] * 8
    assert metrics['new_adam_steps_per_rank'] == 100 and metrics['actual_learning_rate'] == 1e-6
    ledger_facts, ledgers = [], []
    for rank in range(8):
        path = root / 'probe-update' / ('rank-%d-updates.jsonl' % rank)
        values = rows(path)
        assert len(values) == 100
        assert [row['step'] for row in values] == list(range(1, 101))
        assert [row['actual_adam_steps'] for row in values] == list(range(expected_before + 1, expected_before + 101))
        assert all(row['outer'] == outer and row['rank'] == rank
                   and row['actual_learning_rate'] == 1e-6 for row in values)
        assert sum(row['local_questions'] for row in values) == 640
        assert sum(row['local_trajectories'] for row in values) == 3200
        ledger_facts.append(dict(rank=rank, actual_writes=100, last_actual_adam=expected_before + 100,
                                 local_questions=640, local_trajectories=3200, sha256=sha(path)))
        ledgers.append(values)
    global_path = root / 'probe-update/global-loss-updates.jsonl'
    curve = rows(global_path)
    assert len(curve) == 100
    assert Counter(row['global_trajectories'] for row in curve) == {255: 80, 260: 20}
    for step, row in enumerate(curve):
        assert sum(values[step]['local_questions'] for values in ledgers) == row['global_questions']
        assert sum(values[step]['local_trajectories'] for values in ledgers) == row['global_trajectories']
    selection = read(root / 'probe-score/selection.json')
    pool = read(root / 'full-pool/complete.json')
    projected = read(root / 'full/complete.json')
    diagnostic = read(root / 'diagnostic-complete.json')
    assert pool['trajectories'] == 25600 and projected['trajectories'] == 2560
    assert projected['projected_generation_requests'] == 0
    assert projected['full_pool_complete_sha256'] == sha(root / 'full-pool/complete.json')
    assert selection['outcome_labels_used'] is False
    assert pool['actor_version'] == diagnostic['actor_version'] == metrics['actor_version'] == selection['actor_version']
    train, train_ids, train_hashes = input_summary(root / 'prefix/scan', root / 'full-pool/own-em.jsonl', pool['actor_version'])
    test_path = Path(diagnostic['scan_path'])
    test, test_ids, test_hashes = input_summary(test_path, test_path.parent.parent / 'full/own-em.jsonl', pool['actor_version'])
    assert train['questions'] == 5120 and test['questions'] == 512
    assert not train_ids.intersection(test_ids) and not train_hashes.intersection(test_hashes)
    assert metrics['train']['mixed_count'] == train['mixed_questions']
    assert metrics['test']['mixed_count'] == test['mixed_questions']
    checkpoint = metrics['checkpoint']
    head = base / 'probe' / ('head-outer-%04d.pt' % outer)
    assert head.stat().st_size == checkpoint['bytes']
    next_root = base / 'outers' / ('outer-%06d' % (outer + 1)) / 'probe-score'
    next_use = None
    if (next_root / 'complete.json').exists():
        next_selection = read(next_root / 'selection.json')
        assert next_selection['probe_before_checkpoint_sha256'] == checkpoint['sha256']
        next_use = dict(outer=outer + 1, probe_sha256=checkpoint['sha256'],
                        actor_version=next_selection['actor_version'], selection_sha256=sha(next_root / 'selection.json'))
    return dict(status='PERIODIC_NATIVE_INPUTS_ADAM_METRICS_READBACK', outer=outer,
                actor_actual_adam_per_rank=joint['actor']['actual_adam_steps_per_rank'],
                probe_actual_adam_per_rank=metrics['actual_adam_steps_per_rank'], rank_ledgers=ledger_facts,
                global_batch_counts={'255': 80, '260': 20}, global_loss_ledger_sha256=sha(global_path),
                train_inputs=train, test_inputs=test, question_id_overlap=0, question_hash_overlap=0,
                sampling_actor_version=pool['actor_version'], preselection_probe_sha256=selection['probe_before_checkpoint_sha256'],
                producer_checkpoint=checkpoint, next_probe_use=next_use,
                before_update=metrics['before_update'], after_update={'train': metrics['train'], 'test': metrics['test']},
                train_loop_seconds=metrics['train_loop_seconds'], phase_timings=joint['phase_timings'],
                selected_mixed_fraction=projected['actual_selected_mixed_fraction'],
                metrics_sha256=sha(metrics_path), joint_sha256=sha(root / 'complete.json'),
                training_modified=False, tensor_loaded=False, gpu_forward_performed=False,
                independent_model_sha_readback=False, checkpoint_semantic_reload_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--outer', required=True, type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = inspect(args.base, args.outer)
    path = Path(args.output)
    assert Path(args.base).resolve() in path.resolve().parents
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(result, stream, sort_keys=True)
    print(json.dumps(dict(status=result['status'], output=str(path), sha256=sha(path))))


if __name__ == '__main__':
    main()

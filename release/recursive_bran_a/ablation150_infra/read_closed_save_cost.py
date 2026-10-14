"""Read existing closed checkpoint timing receipts; never load payload tensors.

Standalone stdlib CLI/stdin reader. stdout is small aggregate/provenance JSON;
no Node writes, GPU/HTTP interfaces, RNG, locks, signals or training changes.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import time

BASE = Path('/dataset/shaoshitong/recursive-ablation150-20261014')
A_ORIGINAL = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
A_RECOVERY = A_ORIGINAL / 'recovery-from100-20261013'
MILESTONES = (5, 50, 100, 150)
SCIENCE = '233b0dc95de3f7b797d5b424d3b7dcaafca5328f'
A_CONFIGS = {
    A_ORIGINAL: ('1c806d15752d44a7ac99687f0737d7e3',
        '653193a4eb06bd28667f78ac14a66a2d1c97200ca8199b8521dcc2d3de35b319',
        '7966a1e76d6abddb4349d57c4727d69fe62fb16d'),
    A_RECOVERY: ('d97047b5902642c883d24aa908945cda',
        '43e28cc7ecc0d38a93324bf480aafb23a0c1ecdb979e9ea587d0358ae095233d',
        'a1ec638575268747468b1a14ff8fd5f858ca0e1a'),
}
NEW_GROUPS = {'C1-root-only', 'C2-random-final', 'budget-5', 'budget-20',
              'position-early', 'position-late', 'generation-2'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Invalid recorded save duration')
    return float(value)


def signature(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def raw_bytes(path, parent, limit=4 * 1024 * 1024):
    require(path.is_absolute() and not path.is_symlink()
            and path.resolve().is_relative_to(parent.resolve()), 'Unexpected checkpoint evidence path')
    before = path.stat()
    require(before.st_size <= limit, 'Only small JSON checkpoint evidence is allowed')
    raw = path.read_bytes()
    after = path.stat()
    require(signature(before) == signature(after) and len(raw) == before.st_size,
            'Closed small evidence changed during read')
    return raw, dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def small(path, parent):
    raw, proof = raw_bytes(path, parent)
    return json.loads(raw), proof


def configuration(root, expected=None):
    receipt, proof = small(root / 'receipts/configuration.json', root)
    if expected:
        require((receipt.get('run_id'), receipt.get('config_sha256'), receipt.get('source_commit')) == expected,
                'Retained A segment identity differs')
    else:
        require(receipt.get('group') == root.name and receipt.get('source_commit') == SCIENCE
                and receipt.get('actual_update_limit') == 150
                and receipt.get('evaluation_steps') == list(MILESTONES), 'Frozen ablation identity differs')
    config = Path(receipt['config'])
    _, config_proof = raw_bytes(config, root)
    require(config_proof['sha256'] == receipt['config_sha256'], 'Resolved configuration SHA differs')
    return receipt, dict(receipt=proof, resolved_configuration=config_proof)


def worker_save_receipt(path, root, receipt, outer, expected_phase=None):
    value, proof = small(path, root)
    identity = value['identity']
    require(value.get('status') == 'COMPLETE_SAME_WORLD_STATE_SAVED'
            and identity.get('completed_step') == outer
            and identity.get('config_sha256') == receipt['config_sha256']
            and identity.get('code_commit') == receipt['source_commit'], 'Closed checkpoint identity differs')
    if expected_phase:
        require(value.get('boundary_phase') == expected_phase, 'Checkpoint evaluation phase differs')
    elif value.get('boundary_phase') is not None:
        raise ValueError('Pre/rolling checkpoint unexpectedly has a post-evaluation phase')
    workers = value['worker_states']
    require(len(workers) == 8 and {v['identity']['rank'] for v in workers} == set(range(8)),
            'Checkpoint lacks actual eight-rank coverage')
    durations, verifies = {}, {}
    for worker in workers:
        native = worker['identity']
        require(all(native.get(key) == item for key, item in identity.items())
                and native.get('world_size') == 8 and worker.get('full_worker_state_finite') is True,
                'Worker save receipt identity/finite scope differs')
        rank = str(native['rank'])
        durations[rank] = seconds(worker.get('checkpoint_and_reload_seconds'))
        verifies[rank] = worker.get('exact_same_world_reload_verified')
        require(type(verifies[rank]) is bool, 'Worker reload scope missing')
    vals = list(durations.values())
    return dict(outer=outer, manifest=proof, phase=expected_phase or 'pre-evaluation-or-native-boundary',
        worker_checkpoint_and_reload_seconds_by_rank=durations,
        worker_seconds_min=min(vals), worker_seconds_max=max(vals),
        worker_seconds_sum_overlapping=sum(vals), worker_reload_verified_by_rank=verifies,
        coordinator_save_wall_seconds=None, HF_export_wall_seconds=None,
        scope='Eight worker durations are concurrent and include snapshot/logical digest/write/hash; '
              'verified workers also include actual reload. min/max/sum are not coordinator wall or GPU billing.')


def ablation(group, milestone):
    root = BASE / group
    receipt, config_proof = configuration(root)
    records, sources = {}, []
    for output in sorted((root / 'output').glob('attempt-*')):
        path = output / 'completed-outers.jsonl'
        if not path.exists():
            continue
        # Read only the closed records in a bounded append-only journal. A later
        # append does not change any accepted closed record included below.
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
                'Completion journal escapes this group')
        size = path.stat().st_size
        require(size <= 16 * 1024 * 1024, 'Completion journal exceeds small evidence limit')
        with path.open('rb') as handle:
            raw = handle.read(size)
        complete = raw[:raw.rfind(b'\n') + 1]
        sources.append(dict(path=str(path), captured_prefix_bytes=len(complete),
                            captured_prefix_sha256=hashlib.sha256(complete).hexdigest()))
        for line in complete.splitlines():
            row = json.loads(line)
            outer = row.get('completed_outer')
            if type(outer) is not int or not 1 <= outer <= milestone:
                continue
            require(row.get('run_id') == receipt['run_id']
                    and row.get('config_sha256') == receipt['config_sha256']
                    and row.get('source_commit') == receipt['source_commit']
                    and row.get('group') == group and row.get('next_outer') == outer + 1,
                    'Closed completion journal identity/cursor differs')
            if outer in records:
                require(records[outer] == row, 'Conflicting closed save records')
            records[outer] = row
    require(set(records) == set(range(1, milestone + 1)), 'Closed1..milestone completion coverage missing')
    rolling, keypoints = {}, {}
    for outer, record in sorted(records.items()):
        checkpoint = record['checkpoint']
        if outer in MILESTONES:
            require(checkpoint.get('native_only') is False, 'Formal keypoint is not scheduled native state')
            path = root / 'checkpoints/training_state' / ('global_step_' + str(outer))
            pre = worker_save_receipt(path / 'manifest.json', root, receipt, outer)
            post = worker_save_receipt(path / 'post-evaluation/manifest.json', root, receipt, outer, 'post-evaluation')
            require(checkpoint.get('path') == post['manifest']['path']
                    and checkpoint.get('sha256') == post['manifest']['sha256'],
                    'Closed formal post-evaluation receipt SHA differs')
            keypoints[str(outer)] = dict(pre=pre, post=post,
                containing_native_testing_wall_seconds=record['metrics'].get('timing_s/testing'),
                protected_eval_save_checkpoint_noop_seconds=record['metrics'].get('timing_s/save_checkpoint'),
                scope='HF export, pre native save, evaluation and post native save are inside '
                      'native timing_s/testing and timing_s/step; pure HF/eval/coordinator-save split is unrecorded.')
        else:
            expected = root / 'checkpoints/native-rolling' / ('global_step_' + str(outer)) / 'manifest.json'
            require(checkpoint.get('native_only') is True and checkpoint.get('path') == str(expected)
                    and isinstance(checkpoint.get('sha256'), str) and len(checkpoint['sha256']) == 64,
                    'Closed rolling receipt is not its canonical native-only checkpoint')
            # Later rolling payload/manifests may have been normally reclaimed.
            # The durable completed record retains its actual driver timer.
            rolling[str(outer)] = dict(save_seconds=seconds(checkpoint.get('save_seconds')),
                recorded_manifest_sha256=checkpoint['sha256'],
                source='completed-outers checkpoint.save_seconds', current_payload_read=False)
    return dict(run_id=receipt['run_id'], config_sha256=receipt['config_sha256'],
        scientific_source_commit=receipt['source_commit'], configuration=config_proof,
        closed_completion_journals=sources, closed_prefix_outer=milestone,
        rolling_coordinator_save_seconds_by_outer=rolling,
        rolling_save_seconds_sum_outside_native_step=sum(v['save_seconds'] for v in rolling.values()),
        rolling_save_coverage_outers=len(rolling), expected_rolling_outers=milestone-len(keypoints),
        rolling_scope='Actual driver timer includes parallel eight-rank save RPC, driver snapshot/manifest and '
                      'older rolling reclamation; it runs after native logger.log, outside recorded native step.',
        keypoint_worker_save_receipts=keypoints,
        status='COMPLETE_RECORDED_ROLLING_PREFIX_WITH_KEYPOINT_TIMING_LIMITS')


def retained_a(milestone):
    segments, keypoints = {}, {}
    for root, expected in A_CONFIGS.items():
        if root == A_RECOVERY and milestone <= 100:
            continue
        receipt, proof = configuration(root, expected)
        segments[str(root)] = dict(identity=dict(run_id=receipt['run_id'], config_sha256=receipt['config_sha256'],
            source_commit=receipt['source_commit']), configuration=proof)
    for outer in MILESTONES:
        if outer > milestone:
            continue
        root = A_ORIGINAL if outer <= 100 else A_RECOVERY
        receipt, _ = configuration(root, A_CONFIGS[root])
        directory = root / 'checkpoints/training_state' / ('global_step_' + str(outer))
        keypoints[str(outer)] = dict(
            pre=worker_save_receipt(directory / 'manifest.json', root, receipt, outer),
            post=worker_save_receipt(directory / 'post-evaluation/manifest.json', root, receipt, outer, 'post-evaluation'))
    return dict(status='COMPLETE_RETAINED_A_KEYPOINT_WORKER_TIMINGS_WITH_WALL_LIMITS',
        retained_ancestry='Original1..100 plus actual saved100 continuation101..150; discarded original101..118/partial119 excluded.',
        segments=segments, closed_prefix_outer=milestone, keypoint_worker_save_receipts=keypoints,
        rolling_coordinator_save_seconds_sum=None,
        scope='A native manifests record each concurrent worker save/reload duration. No A coordinator '
              'rolling-save wall receipt is identified; this is keypoint worker coverage, not a complete save-wall prefix.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', default='C1-root-only', choices=sorted(NEW_GROUPS | {'A-original'}))
    parser.add_argument('--milestone', type=int, required=True, choices=MILESTONES)
    args = parser.parse_args()
    result = dict(protocol='ablation150_closed_save_cost_readonly_v1', time=time.time(), group=args.group,
        milestone=args.milestone, training_action='NONE', Node_writes=False,
        exact_HF_export_wall_seconds=None, exact_keypoint_coordinator_save_seconds=None,
        measured_GPU_active_seconds=None, allocator_billed_GPU_seconds=None,
        end_to_end_wall_reconstruction_performed=False)
    try:
        result.update(retained_a(args.milestone) if args.group == 'A-original' else ablation(args.group, args.milestone))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status='INCOMPLETE', reason=type(error).__name__ + ': ' + str(error),
                      manufactured_save_total=None)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 2 if result['status'] == 'INCOMPLETE' else 0


if __name__ == '__main__':
    raise SystemExit(main())

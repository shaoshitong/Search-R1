"""Import a CPU-qualified native joint state plus one reserved prefix phase.

The saved joint is the optimizer/RNG boundary.  A later candidate reservation
is an immutable sampling event, not another completed outer.  The new pool
starts at the joint snapshot; ``pool.prepare`` then applies that one already
sealed reservation.  Only qualified prefix/score files and complete individual
continuation receipts are linked.  Failed continuation fragments stay at their
original paths.  This module never loads a model or changes the source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .actor import read, sha, restore_metadata
from .contract import CANDIDATE_QUESTIONS, WORLD
from .periodic import probe_boundary
from .pool import state_sha
from .pending_probe_resume import SCIENCE_KEYS
from reproduction.prefix20_online.launch import once

ADAPTER = 'native_a800_joint_v1'
STATUS = 'NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS'
PHASE_MARKERS = {'prefix': 'complete.json', 'probe-score': 'manifest.json'}
SCIENCE = SCIENCE_KEYS + (
    'actor_loss_reduction', 'actor_loss_virtual_world', 'common_manifest_sha256',
    'source_question_pool_sha256', 'hidden_capture_contract',
    'independent_encoder_forward', 'validation_questions_sha256',
    'final_questions_sha256', 'grounding_manifest_sha256',
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _target_binding(c):
    """Bind science/identity, allowing CPU prepare to finalize idle GPU UUIDs."""
    return _digest(dict(science={key: c.get(key) for key in SCIENCE},
        base=c['base'], execution_id=c['execution_id'], source_commit=c['source_commit'],
        resume_joint=c['resume_joint'], continue_reuse=c.get('continue_reuse'),
        rollout_enforce_eager=c.get('rollout_enforce_eager')))


def _qualified(config):
    spec = config['resume_joint']; source = Path(spec['source_base'])
    if (spec.get('adapter') != ADAPTER or config.get('resume_pending_probe')
            or config.get('resume_import') or config.get('rollout_enforce_eager') is not True):
        raise ValueError('Use only the explicit native joint/prefix recovery adapter with eager rollout')
    for key, file_key in (('source_config_sha256', source/'preparation/launch-config.json'),
                          ('terminal_proof_sha256', spec['terminal_proof']),
                          ('checkpoint_cpu_proof_sha256', spec['checkpoint_cpu_proof'])):
        if sha(file_key) != spec[key]:
            raise ValueError('Native recovery source, terminal or full-SHA CPU qualification changed')
    q = read(spec['checkpoint_cpu_proof']); n = spec['completed_outer']; pending = n+1
    boundary = probe_boundary(n)
    expected = dict(schema_version=1, adapter=ADAPTER, status=STATUS,
        run_id=config['run_id'], source_base=str(source),
        source_execution_id=spec['source_execution_id'], source_commit=spec['source_commit'],
        source_config_sha256=spec['source_config_sha256'], outer=n,
        actor_adam_per_rank=[20*n]*WORLD, scheduler_completed_outer=n,
        probe_adam_per_rank=[boundary['actual_adam_per_rank']]*WORLD,
        probe_last_update_outer=boundary['last_update_outer'],
        pool_next_outer=pending, pool_consumed_candidates=CANDIDATE_QUESTIONS*n,
        pending_outer=pending, current_pool_next_outer=pending+1,
        current_pool_consumed_candidates=CANDIDATE_QUESTIONS*pending,
        source_joint_sha256=spec['source_joint_sha256'], owned_births_all_gone=True,
        terminal_proof_sha256=spec['terminal_proof_sha256'])
    if (type(n) is not int or not 1 <= n < 200 or spec.get('inherited_prefix_outer') != pending
            or any(q.get(key) != value for key, value in expected.items())):
        raise ValueError('Native CPU proof does not seal this exact joint and pending prefix boundary')
    rows = q.get('verified_files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Native CPU proof requires a nonempty full-SHA file seal')
    seals = {}
    for row in rows:
        path = Path(row['path'])
        if (str(path) in seals or path.is_symlink() or not path.is_file()
                or path.resolve(strict=True) != path or not str(path).startswith('/data/')
                or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Qualified files must be unique, canonical, actual remote files')
        st = path.stat()
        if ((st.st_ino, st.st_size, st.st_mtime_ns) != (row['inode'], row['bytes'], row['mtime_ns'])
                or ('dev' in row and st.st_dev != row['dev'])
                or ('device' in row and st.st_dev != row['device'])):
            raise ValueError('Qualified state changed after its full-SHA CPU readback')
        seals[str(path)] = row
    required = ['preparation/launch-config.json', 'latest-joint.json', 'actor/latest.json',
        'probe/latest.json', 'probe/latest.pt', 'probe/initialization-complete.json',
        'question-pool/binding.json', 'question-pool/state.json']
    root = source/'outers'/('outer-%06d'%pending)
    paths = {str(source/name) for name in required}
    paths |= {str(root/name) for name in ('pool-reservation.json', 'selection.json',
        'questions.json', 'actor-binding.json', 'prefix/complete.json',
        'probe-score/manifest.json', 'probe-score/selection.json')}
    paths |= {str(source/'outers'/('outer-%06d'%outer)/'complete.json')
              for outer in range(1, n+1)}
    for outer in (5, 50, 100, 150):
        p = source/'outers'/('outer-%06d'%outer)/'validation/complete.json'
        if outer <= n and p.is_file():
            paths.add(str(p))
    if not paths <= seals.keys():
        raise ValueError('CPU proof omits joint state, history or the already accepted prefix/score event')
    return spec, source, q, seals


def _metadata(path, seals):
    path = Path(path)
    if str(path) not in seals or sha(path) != seals[str(path)]['sha256']:
        raise ValueError('Actual recovery metadata differs from its CPU-qualified full SHA')
    return read(path)


def _boundary(base, c, *, imported):
    spec, source, q, seals = _qualified(c)
    base = Path(base); n = spec['completed_outer']; pending = n+1
    if (source == base or source.parent != base.parent or source.resolve(strict=True) != source
            or not source.name.startswith('search-g1-prefix-selection200-')):
        raise ValueError('Native recovery requires a distinct canonical namespace for the same experiment')
    original = _metadata(source/'preparation/launch-config.json', seals)
    if (c['run_id'] != original['run_id'] or c['source_commit'] == original['source_commit']
            or c['execution_id'] == original['execution_id']
            or spec['source_execution_id'] != original['execution_id']
            or spec['source_commit'] != original['source_commit']
            or any(c.get(key) != original.get(key) for key in SCIENCE)):
        raise ValueError('Native recovery must preserve science in a new source and execution')
    parent_joint = _metadata(source/'latest-joint.json', seals)
    if (sha(source/'latest-joint.json') != spec['source_joint_sha256']
            or sha(source/'outers'/('outer-%06d'%n)/'complete.json') != spec['source_joint_sha256']):
        raise ValueError('The explicit complete parent joint differs from its sealed history')
    actor = _metadata(source/'actor/latest.json', seals)
    probe = _metadata(source/'probe/latest.json', seals)
    pool = _metadata(parent_joint['pool_snapshot'], seals)
    current = _metadata(source/'question-pool/state.json', seals)
    old_root = source/'outers'/('outer-%06d'%pending)
    reservation = _metadata(old_root/'pool-reservation.json', seals)
    b = probe_boundary(n)
    if (parent_joint['run_id'] != c['run_id'] or parent_joint['outer'] != n
            or parent_joint['source_commit'] != spec['source_commit']
            or parent_joint['actor'] != actor or parent_joint['probe'] != probe
            or probe['run_id'] != c['run_id'] or probe['completed_outer'] != b['last_update_outer']
            or probe['actual_adam_steps_per_rank'] != [b['actual_adam_per_rank']]*WORLD
            or pool['next_outer'] != pending or pool['consumed_candidates'] != CANDIDATE_QUESTIONS*n
            or pool['cycle']*len(pool['order'])+pool['cursor'] != CANDIDATE_QUESTIONS*n
            or current['next_outer'] != pending+1
            or current['consumed_candidates'] != CANDIDATE_QUESTIONS*pending
            or current['cycle']*len(current['order'])+current['cursor'] != CANDIDATE_QUESTIONS*pending
            or reservation['outer'] != pending or reservation['after'] != current
            or reservation['before_sha256'] != state_sha(pool)
            or reservation['after_sha256'] != state_sha(current)
            or len(reservation['indices']) != CANDIDATE_QUESTIONS
            or len(set(reservation['indices'])) != CANDIDATE_QUESTIONS
            or (source/'probe/latest.pt').stat().st_ino != Path(parent_joint['probe_snapshot']).stat().st_ino
            or (old_root/'complete.json').exists() or (old_root/'full/complete.json').exists()
            or list(old_root.glob('actor-attempt-*-config.json'))):
        raise ValueError('Require complete joint plus exactly one reserved, not actor-updated, sampling event')
    restore_metadata(c, actor, pending)
    actor_binding = _metadata(old_root/'actor-binding.json', seals)
    expected_binding = dict(run_id=c['run_id'], outer=pending, actor_model_dir=actor['hf_model_dir'],
        actor_version=actor['actor_version'],
        actor_model_files_sha256={name: row['sha256'] for name,row in actor['hf_files'].items()})
    if actor_binding != expected_binding:
        raise ValueError('Accepted prefix is not bound to the immediately previous complete actor')
    for phase, marker in PHASE_MARKERS.items():
        path = old_root/phase/marker; value = _metadata(path, seals)
        if (q['phase_hashes'].get(phase) != sha(path) or value['run_id'] != c['run_id']
                or value['outer'] != pending or value['actor_version'] != actor['actor_version']):
            raise ValueError('Accepted prefix/score phase identity or sealed metadata differs')
    prefix = read(old_root/'prefix/complete.json'); score = read(old_root/'probe-score/manifest.json')
    selection = _metadata(old_root/'probe-score/selection.json', seals)
    if (prefix['questions'] != CANDIDATE_QUESTIONS or prefix['trajectories'] != CANDIDATE_QUESTIONS*5
            or prefix['config_sha256'] != spec['source_config_sha256']
            or score['count'] != CANDIDATE_QUESTIONS*5
            or score['probe_before_checkpoint_sha256'] != probe['sha256']
            or score['prefix_scan_sha256'] != prefix['scan_sha256']
            or score['selection_sha256'] != sha(old_root/'probe-score/selection.json')
            or selection['run_id'] != c['run_id'] or selection['outer'] != pending
            or selection['actor_version'] != actor['actor_version']
            or selection['selected_questions'] != 512 or selection['outcome_labels_used'] is not False):
        raise ValueError('Prefix/score/selection does not preserve the accepted label-free sampling event')
    whitelist = q.get('continuation_rank_whitelist')
    if not isinstance(whitelist, list):
        raise ValueError('An explicit complete continuation-rank whitelist is required, even when empty')
    expected_reuse = dict(outer=pending, source_base=str(source),
        source_config_sha256=spec['source_config_sha256'],
        imported_receipts_sha256={str(row['rank']): row['receipt_sha256'] for row in whitelist})
    if (c.get('continue_reuse') != expected_reuse
            or len({row['rank'] for row in whitelist}) != len(whitelist)
            or any(type(row['rank']) is not int or not 0 <= row['rank'] < WORLD for row in whitelist)):
        raise ValueError('Continuation reuse must match only the fully qualified completed-rank whitelist')
    for row in whitelist:
        path = old_root/'full'/('rank-%d-complete.json'%row['rank'])
        if (row['receipt_path'] != str(path) or row['receipt_sha256'] != sha(path)
                or row['config_sha256'] != spec['source_config_sha256']):
            raise ValueError('Qualified continuation rank receipt changed')
        _metadata(path, seals)
    if imported:
        if (read(base/'latest-joint.json') != parent_joint or read(base/'actor/latest.json') != actor
                or read(base/'probe/latest.json') != probe or read(base/'question-pool/state.json') != pool
                or (base/'probe/latest.pt').stat().st_ino != (source/'probe/latest.pt').stat().st_ino):
            raise ValueError('Imported joint actor/probe/pool state changed before CUDA admission')
    return spec, source, q, seals, parent_joint, actor, probe, pool


def validate_imported_boundary(base, config):
    base = Path(base)
    spec, source, q, seals, joint, actor, probe, pool = _boundary(base, config, imported=True)
    receipt = read(base/'preparation/resume-joint-receipt.json')
    if (receipt['status'] != 'NATIVE_A800_JOINT_IMPORTED'
            or receipt['run_id'] != config['run_id'] or receipt['execution_id'] != config['execution_id']
            or receipt['source_commit'] != config['source_commit']
            or receipt['target_binding_sha256'] != _target_binding(config)
            or receipt['completed_outer'] != spec['completed_outer']
            or receipt['joint_sha256'] != sha(base/'latest-joint.json')
            or receipt['qualification_proof_sha256'] != spec['checkpoint_cpu_proof_sha256']
            or receipt['probe_inode'] != (base/'probe/latest.pt').stat().st_ino):
        raise ValueError('Native joint import receipt differs from the actual qualified state')
    # Prelaunch validation consumes every imported alias, including historical
    # source provenance.  Tensor bytes were fully read by the CPU qualifier;
    # metadata is cheap to rehash and immutable hardlinks retain the same inode.
    for row in receipt['assets']:
        path = Path(row['target']); src = Path(row['source'])
        if ((path.stat().st_dev, path.stat().st_ino) != (src.stat().st_dev, src.stat().st_ino)
                or (path.suffix == '.json' and sha(path) != row['sha256'])):
            raise ValueError('Imported alias no longer matches its sealed immutable source')
    return spec['completed_outer']


def inherited_phase(base, root, phase, c):
    """Only the explicitly sealed next outer may skip prefix/score execution."""
    spec = c['resume_joint']; pending = spec['inherited_prefix_outer']
    if root != Path(base)/'outers'/('outer-%06d'%pending):
        return False
    receipt = read(Path(base)/'preparation/resume-import-receipt.json')
    marker = PHASE_MARKERS[phase]; path = root/phase/marker
    if (receipt['adapter'] != ADAPTER or receipt['run_id'] != c['run_id']
            or receipt['execution_id'] != c['execution_id'] or receipt['outer'] != pending
            or receipt['source_base'] != spec['source_base']
            or receipt['qualification_proof_sha256'] != spec['checkpoint_cpu_proof_sha256']
            or receipt['phase_artifacts'][phase] != phase+'/'+marker
            or receipt['phase_hashes'][phase] != sha(path)):
        raise ValueError('Inherited native phase is not the explicitly qualified original sampling event')
    value = read(path)
    if (value['run_id'] != c['run_id'] or value['outer'] != pending
            or value['actor_version'] != read(root/'actor-binding.json')['actor_version']):
        raise ValueError('Inherited native phase run/outer/actor changed')
    return True


def inherited_actor_binding(base, root, c, actual):
    spec = c['resume_joint']; pending = spec['inherited_prefix_outer']
    if root != Path(base)/'outers'/('outer-%06d'%pending):
        return False
    receipt = read(Path(base)/'preparation/resume-import-receipt.json')
    return (receipt['adapter'] == ADAPTER and receipt['run_id'] == c['run_id']
        and receipt['execution_id'] == c['execution_id']
        and receipt['qualification_proof_sha256'] == spec['checkpoint_cpu_proof_sha256']
        and receipt['actor_binding_sha256'] == sha(root/'actor-binding.json')
        and read(root/'actor-binding.json') == actual)


def import_complete(base):
    base = Path(base); c = read(base/'preparation/launch-config.json')
    receipt = base/'preparation/resume-joint-receipt.json'
    intent = base/'preparation/resume-joint-intent.json'
    if receipt.exists():
        validate_imported_boundary(base, c)
        return dict(status='EXISTING_JOINT_IMPORT_NO_REPLAY', receipt=read(receipt))
    if intent.exists():
        raise RuntimeError('Native joint import intent exists: reconcile partial import, never replay')
    spec, source, q, seals, joint, actor, probe, pool = _boundary(base, c, imported=False)
    n = spec['completed_outer']; pending = n+1
    old_root = source/'outers'/('outer-%06d'%pending)
    root = base/'outers'/old_root.name
    once(intent, dict(adapter=ADAPTER, run_id=c['run_id'], execution_id=c['execution_id'],
        completed_outer=n, source_base=str(source), started_unix=time.time(),
        qualification_proof_sha256=spec['checkpoint_cpu_proof_sha256']))
    assets, history = [], []

    def link(src, dst):
        src, dst = Path(src), Path(dst)
        if str(src) not in seals:
            raise ValueError('Only explicitly full-SHA qualified files may be imported')
        row = seals[str(src)]
        if src.suffix == '.json':
            _metadata(src, seals)
        dst.parent.mkdir(parents=True, exist_ok=True); os.link(src, dst)
        item = dict(source=str(src), target=str(dst), bytes=row['bytes'], inode=row['inode'],
                    dev=src.stat().st_dev, sha256=row['sha256'])
        assets.append(item)
        return item

    for name in ('actor/latest.json', 'probe/initialization-complete.json', 'probe/latest.json',
                 'probe/latest.pt', 'question-pool/binding.json', 'latest-joint.json'):
        link(source/name, base/name)
    link(joint['pool_snapshot'], base/'question-pool/state.json')
    for outer in range(1, n+1):
        old = source/'outers'/('outer-%06d'%outer)
        completed = _metadata(old/'complete.json', seals); b = probe_boundary(outer)
        if (completed['run_id'] != c['run_id'] or completed['outer'] != outer
                or not re.fullmatch('[0-9a-f]{40}', completed['source_commit'])
                or completed['actor']['completed_outer'] != outer
                or completed['actor']['actual_adam_steps_per_rank'] != [20*outer]*WORLD
                or completed['actor']['scheduler_completed_outer'] != outer
                or completed['probe']['completed_outer'] != b['last_update_outer']
                or completed['probe']['actual_adam_steps_per_rank'] != [b['actual_adam_per_rank']]*WORLD):
            raise ValueError('Historical source/run or actual four-rank scientific counters differ')
        for name in ('complete.json', 'probe-frozen.json', 'probe-update/metrics.json',
                     'validation/complete.json'):
            path = old/name
            if path.is_file():
                item = link(path, base/'outers'/old.name/name)
                history.append(dict(item, run_id=c['run_id'], run=c['run_id'], outer=outer,
                    source_commit=completed['source_commit'], role=(
                        'inherited_existing_qa_metadata' if name.startswith('validation/')
                        else 'inherited_complete_joint' if name == 'complete.json'
                        else 'inherited_existing_probe_metadata')))
    for name in ('pool-reservation.json', 'selection.json', 'questions.json', 'actor-binding.json'):
        link(old_root/name, root/name)
    for path in sorted(seals):
        src = Path(path)
        for phase in PHASE_MARKERS:
            phase_root = old_root/phase
            if src.is_relative_to(phase_root):
                link(src, root/phase/src.relative_to(phase_root))
                break
    for row in q['continuation_rank_whitelist']:
        link(row['receipt_path'], root/'full'/('rank-%d-complete.json'%row['rank']))
    imported = dict(adapter=ADAPTER, run_id=c['run_id'], execution_id=c['execution_id'],
        source_base=str(source), source_execution_id=spec['source_execution_id'],
        parent_source_commit=spec['source_commit'], source_config_sha256=spec['source_config_sha256'],
        outer=pending, qualification_proof_sha256=spec['checkpoint_cpu_proof_sha256'],
        phase_hashes=q['phase_hashes'], phase_artifacts={k:k+'/'+v for k,v in PHASE_MARKERS.items()},
        actor_binding_sha256=sha(root/'actor-binding.json'),
        continuation_rank_whitelist=q['continuation_rank_whitelist'],
        history_provenance=history, source_modified=False, raw_partial_fragments_imported=False)
    once(base/'preparation/resume-import-receipt.json', imported)
    result = dict(status='NATIVE_A800_JOINT_IMPORTED', adapter=ADAPTER, run_id=c['run_id'],
        execution_id=c['execution_id'], source_commit=c['source_commit'],
        source_base=str(source), source_execution_id=spec['source_execution_id'],
        parent_source_commit=spec['source_commit'], completed_outer=n, next_outer=pending,
        inherited_prefix_outer=pending, joint_sha256=sha(base/'latest-joint.json'),
        probe_inode=(base/'probe/latest.pt').stat().st_ino,
        qualification_proof_sha256=spec['checkpoint_cpu_proof_sha256'],
        actor_adam_per_rank=actor['actual_adam_steps_per_rank'],
        probe_adam_per_rank=probe['actual_adam_steps_per_rank'], scheduler_completed_outer=n,
        candidate_cursor=pool['cursor'], candidate_cycle=pool['cycle'],
        candidate_consumed=CANDIDATE_QUESTIONS*n,
        target_binding_sha256=_target_binding(c), assets=assets,
        history_provenance=history, inherited_metadata_files=history,
        full_state_import=True, cuda_restore_verified=False, optimizer_reset=False,
        probe_initialized_again=False, candidates_to_resample=0, prefix_scores_to_recompute=0,
        source_modified=False, large_assets_transferred_to_Mac=False)
    once(receipt, result)
    validate_imported_boundary(base, c)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    print(json.dumps(import_complete(args.base), allow_nan=False))


if __name__ == '__main__':
    main()

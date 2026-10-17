"""Archive a wholly skipped actor attempt before fresh same-question sampling.

No actor/model/optimizer files are moved. This operation preserves the reserved
question pool and all rejected generated artifacts; it never claims rollback.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid

from .actor import canonical, restore_metadata, _state_binding
from .contract import CANDIDATE_QUESTIONS, WORLD
from .storage import owned, validate_base

GENERATED = ('prefix', 'probe-score', 'full', 'full-pool', 'full-pool-config.json')
PRESERVED = ('questions.json', 'selection.json', 'pool-reservation.json', 'actor-binding.json')
OPTIONAL_PRESERVED = ('diagnostic-complete.json',)


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_bytes(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode): raise ValueError('Read only an actual regular metadata file')
        with os.fdopen(os.dup(descriptor), 'rb') as stream: raw = stream.read()
        if _signature(os.fstat(descriptor)) != _signature(before):
            raise ValueError('Metadata changed while binding the rejected attempt')
        return raw
    finally:
        os.close(descriptor)


def _read(path):
    return json.loads(_file_bytes(path))


def _sha(path):
    return hashlib.sha256(_file_bytes(path)).hexdigest()


def _write_at(directory, name, value):
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.fsync(directory)


def _snapshot_at(directory, name):
    """No-follow traversal; hash JSON metadata and retain only payload stat data."""
    result = []
    def visit(parent, child, relative):
        before = os.stat(child, dir_fd=parent, follow_symlinks=False)
        if not (stat.S_ISDIR(before.st_mode) or stat.S_ISREG(before.st_mode)):
            raise ValueError('Generated artifacts contain a symlink or non-regular asset')
        row = dict(relative_path=relative, kind='directory' if stat.S_ISDIR(before.st_mode) else 'file',
                   signature=list(_signature(before)), bytes=before.st_size)
        result.append(row)
        if stat.S_ISDIR(before.st_mode):
            descriptor = os.open(child, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                if _signature(os.fstat(descriptor)) != _signature(before): raise ValueError('Generated directory changed')
                for entry in sorted(os.listdir(descriptor)):
                    visit(descriptor, entry, relative + '/' + entry)
                if _signature(os.fstat(descriptor)) != _signature(before): raise ValueError('Generated directory is active')
            finally:
                os.close(descriptor)
        elif relative.endswith('.json'):
            descriptor = os.open(child, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                if _signature(os.fstat(descriptor)) != _signature(before): raise ValueError('Generated metadata changed')
                digest = hashlib.sha256()
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk: break
                    digest.update(chunk)
                if _signature(os.fstat(descriptor)) != _signature(before): raise ValueError('Generated metadata is active')
                row['contents_sha256'] = digest.hexdigest()
            finally:
                os.close(descriptor)
    visit(directory, name, name)
    return dict(metadata_sha256=hashlib.sha256(canonical(result)).hexdigest(), entries=result,
                metadata_hash_definition='canonical relative paths, inode/stat and JSON contents SHA256; payload bytes not read')


def _checked_zero_result(base, config, outer, root, attempt_id):
    latest_path = owned(base, base / 'actor/latest.json')
    latest = _read(latest_path) if latest_path.exists() else None
    if outer == 1:
        if latest is not None: raise ValueError('Fresh outer1 already has an actor state; never replay it')
        prior, actor_version = [0] * WORLD, config['run_id'] + '-official0'
    else:
        if latest is None: raise ValueError('Missing immediately preceding complete actor')
        restore_metadata(config, latest, outer)
        prior, actor_version = latest['actual_adam_steps_per_rank'], latest['actor_version']
    path = owned(base, base / 'actor/outers' / ('outer-%04d-attempt-%s' % (outer, attempt_id)) / 'guard-result.json')
    result = _read(path)
    def checked(row, expected):
        integer_fields = ('opportunities_processed', 'optimizer_writes_this_attempt', 'skipped_update_opportunities',
            'optimizer_writes_before', 'optimizer_writes_after', 'scheduler_advances', 'scheduler_completed_outer',
            'written_with_post_guard')
        return (all(type(row.get(field)) is int for field in integer_fields) and row.get('status') == 'recoverable_guard' and row.get('complete_outer') is False and
            row.get('full_budget_complete') is False and row.get('opportunities_processed') == 20 and
            row.get('opportunity_statuses') == ['skipped_guard'] * 20 and
            row.get('optimizer_writes_this_attempt') == 0 and row.get('skipped_update_opportunities') == 20 and
            row.get('optimizer_writes_before') == expected and row.get('optimizer_writes_after') == expected and
            row.get('scheduler_advances') == 0 and row.get('scheduler_completed_outer') == outer - 1 and
            row.get('written_with_post_guard') == 0 and row.get('automatic_rollback_performed') is False)
    ranks = result.get('ranks')
    if (result.get('run_id') != config['run_id'] or result.get('outer') != outer or result.get('attempt_id') != attempt_id or
            result.get('actor_version') != actor_version or result.get('binding') != _state_binding(config) or
            not checked(result, prior[0]) or
            result.get('cached_rollout_path') != str(root / 'full/complete.json') or
            not isinstance(ranks, list) or len(ranks) != WORLD or
            any(type(row.get('rank')) is not int for row in ranks) or
            {row.get('rank') for row in ranks} != set(range(WORLD))):
        raise ValueError('Require the actual four-rank zero-write20-opportunity guarded disposition')
    for row in ranks:
        rank = row['rank']
        if (not checked(row, prior[rank]) or row.get('run_id') != config['run_id'] or row.get('outer') != outer or
                row.get('attempt_id') != attempt_id or row.get('new_adam_steps') != 0 or
                row.get('actual_adam_steps') != prior[rank] or row.get('opportunities') != 20 or
                (result.get('grounding_state') is not None and (row.get('grounding_state') != result['grounding_state'] or
                    row.get('grounding_probe_version') != result['grounding_probe_version']))):
            raise ValueError('A rank retained a write or has a different actual attempt/counter')
    if config.get('grounding_state') is not None and (result.get('grounding_state') != config['grounding_state'] or
            result.get('grounding_probe_version') != config['grounding_probe_version']):
        raise ValueError('Guarded disposition adopted another reward evaluator')
    if re.fullmatch('[0-9a-f]{64}', result.get('cached_rollout_sha256', '')) is None:
        raise ValueError('Guarded disposition must retain its exact native rollout metadata SHA')
    actor_binding = _read(owned(base, root / 'actor-binding.json'))
    if (actor_binding.get('run_id') != config['run_id'] or actor_binding.get('outer') != outer or
            actor_binding.get('actor_version') != actor_version):
        raise ValueError('Fresh sampling must keep its actually retained actor binding')
    return path, prior, actor_version, result['cached_rollout_sha256']


def archive_zero_write_attempt(base, config, outer, root, attempt_id):
    """After owned workers exit, retain rejected artifacts and return a fresh sample ID.

    Caller injects ``sampling_attempt_id`` into a NEW rollout config, reruns
    prefix/score/continue on the retained2560 questions, then uses a new actor
    attempt ID. It must not call pool.prepare again or replay the bad rollout.
    An interrupted archive is retained with its intent and requires reconciliation.
    """
    base, disk = validate_base(base); root = Path(root)
    if (type(outer) is not int or not 1 <= outer <= 200 or
            root != base / 'outers' / ('outer-%06d' % outer) or
            not isinstance(attempt_id, str) or re.fullmatch('[0-9a-f]{32}', attempt_id) is None or
            config.get('base') != str(base) or config.get('run_id') != disk['run_id'] or
            config.get('source_commit') != disk['source_commit']):
        raise ValueError('Rejected-attempt archive must bind the exact canonical owned run/outer')
    owned(base, root)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    lock = None
    try:
        lock = os.open('guard-retry.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = _read(owned(base, base / 'logs/process-ledger.json'))
        if ledger.get('run_id') != config['run_id'] or any(
                name.startswith('outer-%06d-' % outer) and row.get('state') != 'STOPPED'
                for name, row in ledger.get('jobs', {}).items()):
            raise ValueError('Archive only after every owned current-outer phase has exited')
        guard_path, prior, actor_version, rollout_sha = _checked_zero_result(base, config, outer, root, attempt_id)
        preserved_names = PRESERVED + tuple(name for name in OPTIONAL_PRESERVED if (root / name).exists())
        preserved = {name: dict(path=str(owned(base, root / name)), sha256=_sha(root / name)) for name in preserved_names}
        diagnostic = owned(base, root / 'diagnostic')
        preserved_diagnostic = _snapshot_at(root_fd, 'diagnostic') if diagnostic.exists() else None
        questions = _read(root / 'questions.json'); selection = _read(root / 'selection.json')
        reservation = _read(root / 'pool-reservation.json')
        if (questions.get('run_id') != config['run_id'] or questions.get('outer') != outer or
                questions.get('questions') != CANDIDATE_QUESTIONS or len(questions.get('rows', [])) != CANDIDATE_QUESTIONS or
                len({row['qid'] for row in questions['rows']}) != CANDIDATE_QUESTIONS or
                questions.get('selection_sha256') != preserved['selection.json']['sha256'] or
                selection.get('run_id') != config['run_id'] or selection.get('outer') != outer or
                selection.get('questions') != CANDIDATE_QUESTIONS or
                len(selection.get('rows', [])) != CANDIDATE_QUESTIONS or
                {row['qid'] for row in selection['rows']} != {row['qid'] for row in questions['rows']} or
                reservation.get('outer') != outer or
                len(reservation.get('indices', [])) != CANDIDATE_QUESTIONS or
                len(set(reservation['indices'])) != CANDIDATE_QUESTIONS or
                reservation['indices'] != [row['pool_index'] for row in selection['rows']]):
            raise ValueError('Keep the exact already-reserved2560 distinct question identities')
        guard_sha = _sha(guard_path)
        archive_parent = owned(base, root / 'guard-rejected-attempts')
        if not archive_parent.exists(): os.mkdir('guard-rejected-attempts', dir_fd=root_fd)
        parent_fd = os.open('guard-rejected-attempts', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            name = 'attempt-' + attempt_id; archive = owned(base, archive_parent / name)
            completed = archive / 'ABORTED_ATTEMPT.json'
            def verify_rollout(path):
                full = _read(owned(base, path))
                if (full.get('run_id') != config['run_id'] or full.get('outer') != outer or
                        full.get('actor_version') != actor_version or _sha(path) != rollout_sha):
                    raise ValueError('Guarded disposition does not bind the actual rejected native rollout')
            if archive.exists():
                existing = _read(owned(base, completed)) if completed.exists() else None
                if (existing is not None and existing.get('guard_result_sha256') == guard_sha and
                        existing.get('preserved') == preserved and existing.get('preserved_diagnostic') == preserved_diagnostic and
                        existing.get('run_id') == config['run_id'] and
                        existing.get('outer') == outer and existing.get('attempt_id') == attempt_id):
                    verify_rollout(archive / 'full/complete.json')
                    return dict(existing, existing_archive_reused=True)
                raise RuntimeError('Existing interrupted/rejected archive requires reconciliation; no replay')
            verify_rollout(root / 'full/complete.json')
            plans = []
            for item in GENERATED:
                try: os.stat(item, dir_fd=root_fd, follow_symlinks=False)
                except FileNotFoundError: continue
                snapshot = _snapshot_at(root_fd, item)
                plans.append(dict(name=item, original_path=str(root / item), archived_path=str(archive / item), **snapshot))
            if not {'prefix', 'probe-score', 'full'}.issubset({row['name'] for row in plans}):
                raise ValueError('The rejected native prefix/score/full outputs must all exist before archival')
            sampling_attempt_id = uuid.uuid4().hex
            receipt = dict(schema_version=1, event='ABORTED_ATTEMPT', status='ZERO_WRITE_GUARD_REJECTED_RESAMPLE_REQUIRED',
                run_id=config['run_id'], source_commit=config['source_commit'], outer=outer, attempt_id=attempt_id,
                guard_result_path=str(guard_path), guard_result_sha256=guard_sha, preserved=preserved,
                preserved_diagnostic=preserved_diagnostic,
                diagnostic_retained_for_reuse=preserved_diagnostic is not None and 'diagnostic-complete.json' in preserved,
                diagnostic_evaluation_performed=False,
                actual_adam_steps_per_rank=prior, new_adam_steps_per_rank=[0]*WORLD, opportunities_processed=20,
                complete_outer=False, scheduler_completed_outer=outer - 1, scheduler_advances=0,
                logical_count_advanced=False, automatic_rollback_performed=False, actor_version=actor_version,
                reserved_questions=CANDIDATE_QUESTIONS, same_reserved_questions=True, fresh_current_actor_sampling_required=True,
                replay_rejected_rollout=False, sampling_attempt_id=sampling_attempt_id,
                new_sampling_attempt_id=sampling_attempt_id, optimizer_writes_this_attempt=0,
                archive_path=str(archive), receipt_path=str(completed),
                archived_items=plans, created_unix=time.time())
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            archive_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            try:
                _write_at(archive_fd, 'intent.json', receipt)
                for plan in plans:
                    if _snapshot_at(root_fd, plan['name']) != {key: plan[key] for key in
                            ('metadata_sha256', 'entries', 'metadata_hash_definition')}:
                        raise ValueError('Generated artifacts changed after the rejected-attempt intent')
                    os.rename(plan['name'], plan['name'], src_dir_fd=root_fd, dst_dir_fd=archive_fd)
                    os.fsync(root_fd); os.fsync(archive_fd)
                if any(_sha(root / name) != item['sha256'] for name, item in preserved.items()):
                    raise ValueError('Reserved question/selection/actor-binding or retained diagnostic receipt changed during archive')
                if preserved_diagnostic is not None and _snapshot_at(root_fd, 'diagnostic') != preserved_diagnostic:
                    raise ValueError('Retained same-actor diagnostic changed during archive')
                _write_at(archive_fd, 'ABORTED_ATTEMPT.json', receipt)
                return receipt
            finally:
                os.close(archive_fd)
        finally:
            os.close(parent_fd)
    finally:
        if lock is not None: os.close(lock)
        os.close(root_fd)

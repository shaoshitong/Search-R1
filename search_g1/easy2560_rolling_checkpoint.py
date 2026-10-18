"""Durable per-outer Easy checkpoints and bounded, run-local retention.

The trainer writes the native HF/model, rank-local Adam/scheduler/RNG, driver
and post-callback RNG boundary first. This module does not reload models or
change training. It hashes/fsyncs those actual files, publishes an atomic latest
pointer, and only then retires older checkpoints sealed by this manager. A new
save temporarily coexists with the preceding recoverable save.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
import uuid


MILESTONES = (5, 50, 100, 150, 200)
MANAGER = 'easy2560_per_outer_full_state_v1'
READY = 'EASY2560_ROLLING_CHECKPOINT_READY'


def _need(condition, message):
    if not condition:
        raise ValueError('EASY2560_ROLLING_' + message)


def _json(path):
    return json.loads(Path(path).read_bytes())


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path, value):
    """Replace only after a fully flushed temporary document exists."""
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with temporary.open('xb') as stream:
            stream.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_name(value):
    path = PurePosixPath(value)
    _need(isinstance(value, str) and value and not path.is_absolute() and '..' not in path.parts
          and str(path) == value, 'UNSAFE_RELATIVE_FILE')
    return value


def _within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _context(base):
    original = Path(base)
    _need(not original.is_symlink(), 'SYMLINK_BASE')
    base = original.resolve()
    run = _json(base / 'logs/run-manifest.json')
    _need(run.get('easy2560_selection') is True and run.get('training_world_size') == 4,
          'NATIVE4_EASY_RUN_REQUIRED')
    root = Path(run.get('checkpoint_dir', base / 'checkpoints/search-g1-common8'))
    _need(root.is_absolute() and not root.is_symlink(), 'UNSAFE_CHECKPOINT_ROOT')
    actor = root / 'actor'
    _need(_within(actor.resolve(), base), 'CHECKPOINT_ROOT_OUTSIDE_BASE')
    metadata = base / 'preparation/rolling-checkpoints'
    _need(not metadata.is_symlink() and _within(metadata.resolve(), base), 'UNSAFE_METADATA_ROOT')
    metadata.mkdir(parents=True, exist_ok=True)
    return base, run, actor.resolve(), metadata


@contextlib.contextmanager
def _locked(metadata):
    with (metadata / 'manager.lock').open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def _files(directory):
    """A closed regular-file tree: no traversal of directory/file symlinks."""
    _need(directory.is_dir() and not directory.is_symlink(), 'UNSAFE_DIRECTORY')
    result = {}
    for folder, directories, files in os.walk(str(directory), followlinks=False):
        parent = Path(folder)
        for name in directories:
            _need(not (parent / name).is_symlink(), 'SYMLINK_DIRECTORY')
        for name in files:
            path = parent / name
            _need(path.is_file() and not path.is_symlink(), 'NONREGULAR_OR_SYMLINK_FILE')
            result[str(path.relative_to(directory))] = path
    return result


def _record(path, *, fsync=False):
    path = Path(path)
    _need(path.is_file() and not path.is_symlink(), 'UNSAFE_FILE')
    before = path.stat()
    with path.open('rb') as stream:
        _need(_identity(os.fstat(stream.fileno())) == _identity(before), 'FILE_REPLACED_BEFORE_HASH')
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
        if fsync:
            os.fsync(stream.fileno())
        after_fd = os.fstat(stream.fileno())
    after = path.stat()
    _need(_identity(before) == _identity(after_fd) == _identity(after), 'FILE_CHANGED_DURING_HASH')
    return dict(path=str(path), device=after.st_dev, inode=after.st_ino, bytes=after.st_size,
                mtime_ns=after.st_mtime_ns, sha256=value.hexdigest())


def _check_record(row, path, *, full_sha):
    _need(path.is_file() and not path.is_symlink() and str(path) == row['path'], 'FILE_MISSING_OR_REBOUND')
    stat = path.stat()
    # Shared filesystems can expose a different client-local st_dev on the
    # login and GPU nodes. It is valid for same-host before/after reads above,
    # not for a durable cross-node recovery manifest.
    _need((stat.st_ino, stat.st_size, stat.st_mtime_ns) ==
          (row['inode'], row['bytes'], row['mtime_ns']),
          'FILE_IDENTITY_CHANGED')
    if full_sha:
        _need(_sha(path) == row['sha256'], 'FILE_SHA_CHANGED')
        _need(_identity(path.stat()) == _identity(stat), 'FILE_CHANGED_DURING_VERIFY')


def _proof(proof, expected, records):
    _need(proof['path'] == str(expected), 'STATE_PATH_DIFFERS')
    row = records[str(expected)]
    _need(proof['sha256'] == row['sha256'], 'STATE_SHA_DIFFERS')
    if 'bytes' in proof:
        _need(proof['bytes'] == row['bytes'], 'STATE_SIZE_DIFFERS')


def _validate_producer(checkpoint, step, run, records):
    complete = _json(checkpoint / 'complete.json')
    _need(complete.get('run_id') == run['run_id'] and complete.get('step') == step
          and complete.get('finite_all_saved_tensors') is True, 'PRODUCER_NOT_COMPLETE')
    if 'checkpoint' in complete:
        _need(complete['checkpoint'] == str(checkpoint), 'PRODUCER_PATH_DIFFERS')
    hashes = complete['files_sha256']
    required = {'model.safetensors.index.json', 'training_state/manifest.json', 'training_state/driver.pt'} | {
        'training_state/rank-%d.pt' % rank for rank in range(4)}
    index = _json(checkpoint / 'model.safetensors.index.json')['weight_map']
    _need(index and isinstance(index, dict), 'MODEL_INDEX_EMPTY')
    required.update(_relative_name(name) for name in index.values())
    _need(required <= set(hashes), 'FULL_MODEL_ADAM_RNG_DRIVER_REQUIRED')
    for name, sha in hashes.items():
        path = checkpoint / _relative_name(name)
        _need(str(path) in records and records[str(path)]['sha256'] == sha, 'PRODUCER_SHA_DIFFERS')
    state = _json(checkpoint / 'training_state/manifest.json')
    _need(state.get('completed_outer') == step and state.get('question_probe') is None,
          'STATE_OUTER_OR_SELECTION_HEAD_DIFFERS')
    workers = state['worker_files']
    _need(len(workers) == 4 and sorted(row['rank'] for row in workers) == list(range(4)), 'FOUR_RANKS_REQUIRED')
    for row in workers:
        _need(row.get('run_id', run['run_id']) == run['run_id'], 'WORKER_RUN_DIFFERS')
        _proof(row, checkpoint / ('training_state/rank-%d.pt' % row['rank']), records)
    _proof(state['driver_file'], checkpoint / 'training_state/driver.pt', records)
    return complete


def _validate_boundary(path, directory, step, run, records):
    _need(path == directory / 'manifest.json', 'BOUNDARY_PATH_DIFFERS')
    value = _json(path)
    _need(value.get('run_id') == run['run_id'] and value.get('completed_outer') == step,
          'BOUNDARY_RUN_OR_OUTER_DIFFERS')
    _need(value.get('callbacks_complete') is True and value.get('reward_refresh_pending') is False,
          'POST_CALLBACK_BOUNDARY_REQUIRED')
    workers = value['worker_files']
    _need(len(workers) == 4 and sorted(row['rank'] for row in workers) == list(range(4)), 'BOUNDARY_FOUR_RANKS_REQUIRED')
    for row in workers:
        _proof(row, directory / ('rank-%d.pt' % row['rank']), records)
    _proof(value['driver_file'], directory / 'driver.pt', records)
    return value


def verify_checkpoint_manifest(base, manifest_path, *, expected_run_id=None, full_sha=True):
    """Verify sealed file bytes/identities without Torch, CUDA or model reload."""
    base, run, actor, metadata = _context(base)
    manifest_path = Path(manifest_path)
    _need(not manifest_path.is_symlink(), 'SYMLINK_MANIFEST')
    value = _json(manifest_path)
    step = value['step']
    _need(type(step) is int and 1 <= step <= 200, 'INVALID_OUTER')
    directory = metadata / ('global_step_%d' % step)
    checkpoint = actor / ('global_step_%d' % step)
    _need(manifest_path == directory / 'full-manifest.json' and value['manager'] == MANAGER
          and value['status'] == READY and value['base'] == str(base)
          and value['checkpoint'] == str(checkpoint) and value['run_id'] == run['run_id'], 'MANIFEST_SCOPE_DIFFERS')
    _need(expected_run_id in (None, run['run_id']), 'EXPECTED_RUN_DIFFERS')
    _need(value['source_commit'] == run['source_commit'], 'SOURCE_COMMIT_DIFFERS')
    paths = _files(checkpoint)
    _need(set(paths) == set(value['files']), 'CHECKPOINT_MEMBERS_CHANGED')
    for name, row in value['files'].items():
        _check_record(row, paths[name], full_sha=full_sha)
    records = {row['path']: row for row in value['files'].values()}
    _validate_producer(checkpoint, step, run, records)
    if value['boundary_manifest'] is not None:
        boundary_dir = directory / 'boundary'
        paths = _files(boundary_dir)
        _need(set(paths) == set(value['boundary_files']), 'BOUNDARY_MEMBERS_CHANGED')
        for name, row in value['boundary_files'].items():
            _check_record(row, paths[name], full_sha=full_sha)
            records[row['path']] = row
        boundary_path = boundary_dir / 'manifest.json'
        _need(value['boundary_manifest'] == value['boundary_files']['manifest.json'], 'BOUNDARY_MANIFEST_DIFFERS')
        _need(value['boundary'] == _validate_boundary(boundary_path, boundary_dir, step, run, records),
              'BOUNDARY_DOCUMENT_DIFFERS')
    return value


def _latest(base, metadata, *, verify):
    path = metadata / 'latest.json'
    if not path.exists():
        return None
    _need(not path.is_symlink(), 'SYMLINK_LATEST')
    value = _json(path)
    manifest = Path(value['manifest_path'])
    _need(value.get('status') == READY and value.get('manager') == MANAGER, 'LATEST_NOT_READY')
    _need(_sha(manifest) == value['manifest_sha256'], 'LATEST_MANIFEST_SHA_DIFFERS')
    sealed = verify_checkpoint_manifest(base, manifest, expected_run_id=value['run_id'], full_sha=verify)
    for key in ('base', 'step', 'checkpoint', 'run_id', 'source_commit'):
        _need(value[key] == sealed[key], 'LATEST_BINDING_DIFFERS_' + key)
    return value


def latest_checkpoint(base, *, verify=True):
    """Return the last fully published checkpoint, or None before the first save."""
    base, _, _, metadata = _context(base)
    return _latest(base, metadata, verify=verify)


def _retire_files(directory, rows):
    """Unlink only exactly the files sealed by this manager; never broad rmtree."""
    actual = _files(directory)
    _need(set(actual) == set(rows), 'RETIRE_MEMBERS_CHANGED')
    for name, row in rows.items():
        _check_record(row, actual[name], full_sha=False)
    for name in sorted(actual):
        actual[name].unlink()
    for folder, _, _ in os.walk(str(directory), topdown=False):
        Path(folder).rmdir()
    _fsync_directory(directory.parent)


def _prune(base, run, actor, metadata, latest, *, milestones):
    # publish_checkpoint just full-hashed/fsynced these exact identities. An
    # independent prune_checkpoints call performs a fresh full SHA first.
    verify_checkpoint_manifest(base, latest['manifest_path'], expected_run_id=run['run_id'], full_sha=False)
    retired, errors = [], []
    for directory in sorted(metadata.glob('global_step_*')):
        if directory.is_symlink() or not directory.is_dir():
            continue
        path = directory / 'full-manifest.json'
        if not path.is_file() or path.is_symlink():
            continue
        try:
            sealed = _json(path)
            step = sealed['step']
            if type(step) is not int or step >= latest['step'] or step in milestones:
                continue
            expected_cp = actor / ('global_step_%d' % step)
            _need(directory == metadata / ('global_step_%d' % step) and sealed['manager'] == MANAGER
                  and sealed['status'] == READY and sealed['base'] == str(base) and sealed['run_id'] == run['run_id']
                  and sealed['checkpoint'] == str(expected_cp), 'RETIRE_SCOPE_DIFFERS')
            if not expected_cp.exists():
                continue
            _retire_files(expected_cp, sealed['files'])
            # Leave the small sealed audit/manifest. Only checkpoint-owned .pt
            # boundary payloads are retired; logs, raw traces and selections are
            # outside these two explicitly scoped trees.
            boundary = directory / 'boundary'
            for name, row in sealed.get('boundary_files', {}).items():
                if Path(name).suffix != '.pt':
                    continue
                target = boundary / _relative_name(name)
                _need(_within(target, boundary), 'RETIRE_BOUNDARY_OUTSIDE_SCOPE')
                _check_record(row, target, full_sha=False)
                target.unlink()
            if boundary.exists():
                _fsync_directory(boundary)
            receipt = dict(manager=MANAGER, base=str(base), run_id=run['run_id'], retired_step=step,
                           protected_latest_step=latest['step'], retired_unix=time.time(),
                           checkpoint=str(expected_cp), latest_manifest_sha256=latest['manifest_sha256'])
            _atomic_json(directory / 'retired.json', receipt)
            retired.append(step)
        except (OSError, ValueError, KeyError, TypeError) as error:
            # Retention failure leaves the newly published recovery point usable.
            errors.append(dict(directory=str(directory), error=type(error).__name__ + ': ' + str(error)))
    return dict(retired_steps=retired, retention_errors=errors)


def prune_checkpoints(base, *, milestone_steps=MILESTONES):
    """Retry run-local retention, first independently rehashing the latest save."""
    base, run, actor, metadata = _context(base)
    with _locked(metadata):
        latest = _latest(base, metadata, verify=True)
        if latest is None:
            return dict(retired_steps=[], retention_errors=[])
        return _prune(base, run, actor, metadata, latest, milestones=set(milestone_steps) | set(MILESTONES))


def publish_checkpoint(base, checkpoint, *, boundary_manifest=None, run_id=None, source_commit=None,
                       milestone_steps=MILESTONES, prune=True):
    """Seal a completed full save; atomically advance latest, then retire old.

    Call only after all ranks have finished saving and scheduled evaluation/
    grounding callbacks have completed. boundary_manifest binds the actual
    post-callback driver and four rank RNG snapshots; native checkpoint bytes
    stay immutable. A failed save/seal never retires the previous checkpoint.
    If latest's directory fsync fails after replace, its pointer may read old or
    new after recovery; both complete checkpoints remain because pruning has
    not started.
    """
    base, run, actor, metadata = _context(base)
    _need(run_id in (None, run['run_id']) and source_commit in (None, run['source_commit']), 'RUN_BINDING_DIFFERS')
    checkpoint = Path(checkpoint)
    _need(not checkpoint.is_symlink() and checkpoint.parent == actor, 'CHECKPOINT_OUTSIDE_ACTOR_ROOT')
    name = checkpoint.name
    _need(name.startswith('global_step_') and name[len('global_step_'):].isdigit(), 'INVALID_CHECKPOINT_NAME')
    step = int(name[len('global_step_'):])
    _need(type(step) is int and 1 <= step <= 200 and name == 'global_step_%d' % step, 'INVALID_OUTER')
    directory = metadata / name
    directory.mkdir(exist_ok=True)
    _need(not directory.is_symlink(), 'SYMLINK_METADATA_DIRECTORY')
    manifest_path = directory / 'full-manifest.json'
    with _locked(metadata):
        old_path = metadata / 'latest.json'
        old = _json(old_path) if old_path.exists() else None
        if old is not None:
            _need(not old_path.is_symlink() and old.get('manager') == MANAGER
                  and old.get('base') == str(base) and old.get('run_id') == run['run_id']
                  and type(old.get('step')) is int, 'EXISTING_LATEST_SCOPE_DIFFERS')
        _need(old is None or old['step'] <= step, 'LATEST_MUST_NOT_MOVE_BACKWARDS')
        if manifest_path.exists():
            sealed = verify_checkpoint_manifest(base, manifest_path, full_sha=True)
            _need((boundary_manifest is None and sealed['boundary_manifest'] is None) or
                  (boundary_manifest is not None and sealed['boundary_manifest']['path'] == str(Path(boundary_manifest))),
                  'REPUBLISH_BOUNDARY_DIFFERS')
        else:
            files = {name: _record(path, fsync=True) for name, path in sorted(_files(checkpoint).items())}
            records = {row['path']: row for row in files.values()}
            _validate_producer(checkpoint, step, run, records)
            boundary_files, boundary = {}, None
            if boundary_manifest is not None:
                boundary_path = Path(boundary_manifest)
                boundary_dir = directory / 'boundary'
                boundary_files = {name: _record(path, fsync=True)
                                  for name, path in sorted(_files(boundary_dir).items())}
                records.update({row['path']: row for row in boundary_files.values()})
                boundary = _validate_boundary(boundary_path, boundary_dir, step, run, records)
                _fsync_directory(boundary_dir)
            for folder, _, _ in os.walk(str(checkpoint), topdown=False):
                _fsync_directory(folder)
            _fsync_directory(checkpoint.parent)
            sealed = dict(schema_version=1, manager=MANAGER, status=READY, base=str(base),
                run_id=run['run_id'], source_commit=run['source_commit'], checkpoint=str(checkpoint), step=step,
                files=files, boundary_files=boundary_files, boundary=boundary,
                boundary_manifest=boundary_files.get('manifest.json'), sealed_unix=time.time(),
                total_bytes=sum(row['bytes'] for row in list(files.values()) + list(boundary_files.values())),
                actual_cuda_restore_accepted=False,
                scope='fsynced full-file SHA preservation; actual native4 restore acceptance is separate')
            _atomic_json(manifest_path, sealed)
        # Identity readback makes this the same completed/fsynced file set that
        # was full-hashed above, without reading 38 GB a second time per outer.
        verify_checkpoint_manifest(base, manifest_path, full_sha=False)
        pointer = {key: sealed[key] for key in ('manager', 'status', 'base', 'run_id', 'source_commit', 'checkpoint', 'step')}
        pointer.update(manifest_path=str(manifest_path), manifest_sha256=_sha(manifest_path), published_unix=time.time())
        _atomic_json(metadata / 'latest.json', pointer)
        result = dict(latest=pointer, checkpoint_file_count=len(sealed['files']),
                      boundary_file_count=len(sealed['boundary_files']), total_bytes=sealed['total_bytes'],
                      retired_steps=[], retention_errors=[])
        if prune:
            result.update(_prune(base, run, actor, metadata, pointer,
                                 milestones=set(milestone_steps) | set(MILESTONES)))
        return result

"""Copy final200 into a separate acceptance target; no GPU/model operations.

All later tensor loads are bound to freshly copied regular target files. Source
paths remain audit metadata only. The caller must admit this one target against
the existing 2TB storage reservation before invoking this component.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


ROLE = 'prefix20_final200_restore_only'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def regular_under(path, root):
    path, root = Path(path), Path(root)
    if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(root):
        raise ValueError('RESTORE_ONLY_PATH_OUTSIDE_TARGET_OR_SYMLINK')
    if not path.is_file() or path.is_symlink():
        raise ValueError('RESTORE_ONLY_REGULAR_FILE_REQUIRED')
    return path


def record(path, **extra):
    path = Path(path)
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('RESTORE_ONLY_FILE_CHANGED_DURING_SHA')
    return dict(path=str(path), inode=after.st_ino, bytes=after.st_size,
                mtime_ns=after.st_mtime_ns, full_read_sha256=digest, **extra)


def publish(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mkdir_durable(path):
    path = Path(path)
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir()
        descriptor = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def copy_verified(source, destination, digest, size=None):
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('RESTORE_ONLY_TARGET_MEMBER_EXISTS')
    before = source.stat()
    if not source.is_file() or (size is not None and before.st_size != size) or sha(source) != digest:
        raise ValueError('RESTORE_ONLY_SOURCE_SHA_OR_SIZE_DIFFERS')
    mkdir_durable(destination.parent)
    if destination.parent.resolve() != destination.parent:
        raise ValueError('RESTORE_ONLY_TARGET_PARENT_IS_SYMLINK')
    with source.open('rb') as inp, destination.open('xb') as out:
        shutil.copyfileobj(inp, out, 8 * 1024**2)
        out.flush()
        os.fsync(out.fileno())
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('RESTORE_ONLY_SOURCE_CHANGED_DURING_COPY')
    result = record(destination)
    if result['full_read_sha256'] != digest or result['bytes'] != before.st_size:
        raise ValueError('RESTORE_ONLY_TARGET_FRESH_SHA_DIFFERS')
    if (destination.stat().st_dev, destination.stat().st_ino) == (before.st_dev, before.st_ino):
        raise ValueError('RESTORE_ONLY_TARGET_IS_SOURCE_HARDLINK')
    result['copied_from'] = dict(path=str(source), sha256=digest, inode=before.st_ino, device=before.st_dev)
    return result


def verify_target_copy(base, receipt_path):
    """Freshly check actual copied bytes and reject any original load pointer."""
    base = Path(base).resolve()
    receipt_path = regular_under(Path(receipt_path), base)
    proof = json.loads(receipt_path.read_text())
    if proof.get('role') != ROLE or proof['target_base'] != str(base):
        raise ValueError('RESTORE_ONLY_COPY_RECEIPT_SCOPE_DIFFERS')
    if proof['step'] != 200 or proof['world_size'] != 4:
        raise ValueError('RESTORE_ONLY_FINAL200_NATIVE4_REQUIRED')
    seen = set()
    for row in proof['files']:
        path = regular_under(Path(row['path']), base)
        if str(path) in seen:
            raise ValueError('RESTORE_ONLY_COPY_MANIFEST_REPEATS')
        seen.add(str(path))
        current = record(path)
        for key in ('inode', 'bytes', 'mtime_ns', 'full_read_sha256'):
            if current[key] != row[key]:
                raise ValueError('RESTORE_ONLY_COPIED_MEMBER_CHANGED')
        origin = row.get('copied_from')
        if origin and (path.stat().st_dev, path.stat().st_ino) == (origin['device'], origin['inode']):
            raise ValueError('RESTORE_ONLY_ORIGINAL_OBJECT_REUSED')
    pair_path = regular_under(Path(proof['target_pair_receipt']), base)
    pair = json.loads(pair_path.read_text())
    groups = pair['files'] + pair['boundary_worker_files'] + [pair['boundary_driver'],
        pair['source_run_manifest'], pair['reward_probe_manifest']] + pair['reward_files']
    history = pair.get('historical_artifacts', {})
    groups += history.get('checkpoints', []) + list(history.get('development', {}).values()) + history.get('reward_archives', [])
    for row in groups:
        if str(regular_under(Path(row['path']), base)) not in seen:
            raise ValueError('RESTORE_ONLY_UNSEALED_PAIR_MEMBER')
    live = json.loads(Path(pair['reward_probe_manifest']['path']).read_text())
    for key in ('commit', 'boundary'):
        if str(regular_under(Path(live[key]), base)) not in seen:
            raise ValueError('RESTORE_ONLY_UNSEALED_REWARD_DEPENDENCY')
    encoder = Path(live['encoder'])
    if not encoder.is_relative_to(base) or encoder.resolve() != encoder:
        raise ValueError('RESTORE_ONLY_REWARD_ENCODER_ESCAPES_TARGET')
    if pair['source_base'] != str(base) or not Path(pair['checkpoint']).is_relative_to(base):
        raise ValueError('RESTORE_ONLY_PAIR_STILL_POINTS_TO_ORIGINAL')
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    paths = [run['initial_model'], run['reference_model'], run['full_state_restore']['source_checkpoint'],
             run['full_state_restore']['source_receipt']]
    paths += [run['question_probe'][name] for name in ('model_dir', 'head_safetensors', 'probe_config', 'eligible_pool_path')]
    if any(not Path(value).is_relative_to(base) or Path(value).resolve() != Path(value) for value in paths):
        raise ValueError('RESTORE_ONLY_RUNTIME_LOAD_PATH_ESCAPES_TARGET')
    for name in run.get('initial_model_sha256', {}):
        if str(regular_under(Path(run['initial_model']) / name, base)) not in seen:
            raise ValueError('RESTORE_ONLY_UNSEALED_OFFICIAL_MODEL_MEMBER')
    return proof


def prepare(base, source_receipt, commit):
    """Use a published/bootstrap-only target, then copy all full-state inputs.

    Target source/repo/env may already exist from control.bootstrap(). Canonical
    data symlinks created by that bootstrap are replaced only in this new target.
    No source model/checkpoint/receipt is edited, linked, or deleted.
    """
    from reproduction.prefix20_joint4.control import own_base, bind_repo, ASSETS
    from reproduction.common8.source_files import runtime_files
    base = own_base(base)
    if not base.name.startswith('qprobe-teacher-warmup200-prefix20-restore200-'):
        raise ValueError('RESTORE_ONLY_DEDICATED_TARGET_REQUIRED')
    if (base / 'logs/run-manifest.json').exists() or (base / 'preparation/restore-copy-claim.json').exists():
        raise ValueError('RESTORE_ONLY_TARGET_ALREADY_ATTEMPTED_INSPECT')
    from reproduction.prefix20_joint4.restore_controller import validate_copy_permit
    validate_copy_permit(base, source_receipt, commit)
    from search_g1.qprobe_checkpoint import read_source
    bind_repo(base, commit)
    path, original, checkpoint, origin, parent = read_source(source_receipt)
    if original['step'] != 200 or original['world_size'] != 4 or origin == base:
        raise ValueError('RESTORE_ONLY_FINAL200_DISTINCT_TARGET_REQUIRED')
    if parent['question_probe'].get('probe_architecture') != 'qwen2_prefix20_trajectory_em_v1':
        raise ValueError('RESTORE_ONLY_SCHEMA3_PREFIX20_REQUIRED')
    publish(base / 'preparation/restore-copy-claim.json', dict(role=ROLE, original_receipt=str(path),
            original_receipt_sha256=sha(path), target_base=str(base), checked_unix=time.time()))
    copied, mapping = [], {}
    def member(row, destination, *, reuse=True):
        old = str(Path(row['path']).resolve())
        if reuse and old in mapping:
            new = record(mapping[old])
        else:
            new = copy_verified(old, destination, row['full_read_sha256'], row['bytes'])
            copied.append(new)
            mapping[old] = Path(new['path'])
        return dict(row, **{key:new[key] for key in ('path', 'inode', 'bytes', 'mtime_ns', 'full_read_sha256')})
    pair = copy.deepcopy(original)
    cp = base / 'checkpoints/search-g1-common8/actor/global_step_200'
    pair['files'] = [member(row, cp / row['relative_path']) for row in original['files']]
    boundary = base / 'artifacts/question-probe/completed-pairs/global_step_200'
    pair['boundary_worker_files'] = [member(row, boundary / 'boundary-worker-rng' / ('rank-%d.pt' % row['rank']))
                                     for row in original['boundary_worker_files']]
    pair['boundary_driver'] = member(original['boundary_driver'], boundary / 'boundary-driver.pt')
    pair['source_run_manifest'] = member(original['source_run_manifest'], boundary / 'source-run-manifest.json')
    old_live = json.loads(Path(original['reward_probe_manifest']['path']).read_text())
    encoder = Path(old_live['encoder']).resolve()
    reward = []
    for row in original['reward_files']:
        old = Path(row['path']).resolve()
        if old.is_relative_to(encoder):
            destination = base / 'assets/reward-encoder' / old.relative_to(encoder)
        else:
            destination = base / 'assets/reward-heads' / old.name
        reward.append(member(row, destination, reuse=False))
    pair['reward_files'] = reward
    live = copy.deepcopy(old_live)
    live['commit'], live['boundary'] = str(mapping[str(Path(old_live['commit']).resolve())]), str(mapping[str(Path(old_live['boundary']).resolve())])
    live['encoder'] = str(base / 'assets/reward-encoder')
    live_path = boundary / 'reward-probe-manifest.json'
    publish(live_path, live)
    live_record = record(live_path, relative_path=original['reward_probe_manifest']['relative_path'])
    copied.append(live_record)
    pair['reward_probe_manifest'] = live_record
    histories = pair.get('historical_artifacts', {})
    def historical(row):
        name = hashlib.sha256(str(row['path']).encode()).hexdigest() + '-' + Path(row['path']).name
        return member(row, base / 'preparation/historical' / name)
    histories['checkpoints'] = [historical(row) for row in histories.get('checkpoints', [])]
    histories['development'] = {key:historical(row) for key,row in histories.get('development', {}).items()}
    histories['reward_archives'] = [historical(row) for row in histories.get('reward_archives', [])]
    pair.update(source_base=str(base), checkpoint=str(cp), historical_artifacts=histories,
                question_probe_state_path=str(cp / 'training_state/question-probe-state.pt'),
                head_path=str(cp / 'training_state' / Path(original['head_path']).name),
                actual_cuda_restore_accepted=False,
                restore_copy_provenance=dict(original_receipt=str(path), original_receipt_sha256=sha(path),
                    original_run_id=original['run_id'], all_tensor_loads_must_use_target=True))
    pair_path = boundary / 'source-receipt.json'
    publish(pair_path, pair)
    copied.append(record(pair_path))
    model = base / 'assets/official-model'
    for name,digest in parent['initial_model_sha256'].items():
        copied.append(copy_verified(Path(parent['initial_model']) / name, model / name, digest))
    for name,digest in parent['input_sha256'].items():
        target = base / 'data' / name
        if target.is_symlink():
            target.unlink()  # New target's bootstrap link only; never unlink its source.
        copied.append(copy_verified(origin / 'data' / name, target, digest))
    for name,(size,digest) in ASSETS.items():
        key = {'head.safetensors':'head_safetensors', 'probe-config.json':'probe_config',
               'eligible-question-pool.json':'eligible_pool_path'}[name]
        copied.append(copy_verified(parent['question_probe'][key], base / 'assets/prefix20' / name, digest, size))
    run = copy.deepcopy(parent)
    run.pop('full_state_restore', None)
    run.update(source_commit=commit, source_sha256={str(p.relative_to(base / 'repo')):sha(p)
               for p in runtime_files(base / 'repo')}, initial_model=str(model), reference_model=str(model),
               completed_updates=0, execution_backend='slurm', execution_site='hpc3', training_world_size=4,
               checkpoint_dir=str(base / 'checkpoints/search-g1-common8'),
               train_log=str(base / 'logs/forbidden-training.log'), pretraining_baseline_policy='reuse_historical',
               initial_probe_version=pair['reward_probe_version'], restore_only_role=ROLE,
               native_ray_execution_id=uuid.uuid4().hex,
               full_state_restore=dict(kind='question_probe_pair', source_receipt=str(pair_path),
                   source_receipt_sha256=sha(pair_path), source_checkpoint=str(cp), source_base=str(base),
                   completed_outer=200, actual_optimizer_writes_per_rank=pair['attempt_progress']['cumulative_writes'],
                   reward_probe_version=pair['reward_probe_version'], reward_probe_round=pair['reward_probe_round']))
    for key,value in dict(model_dir=str(model), head_safetensors=str(base / 'assets/prefix20/head.safetensors'),
            probe_config=str(base / 'assets/prefix20/probe-config.json'),
            eligible_pool_path=str(base / 'assets/prefix20/eligible-question-pool.json'),
            output_dir=str(base / 'artifacts/restored-prefix20'), snapshot_path=None).items():
        run['question_probe'][key] = value
    for key in ('ray_tmp_dir', 'ray_tmp_dir_identity', 'ray_tmp_retention'):
        run.pop(key, None)
    publish(base / 'logs/run-manifest.json', run)
    copied.append(record(base / 'logs/run-manifest.json'))
    live_dir = base / 'artifacts/live'
    mkdir_durable(live_dir)
    publish(live_dir / 'manifest.json', live)
    copied.append(record(live_dir / 'manifest.json'))
    proof = dict(role=ROLE, target_base=str(base), step=200, world_size=4,
            target_pair_receipt=str(pair_path), original_receipt_sha256=sha(path), files=copied,
            copied_bytes=sum(row['bytes'] for row in copied), GPU_restore_accepted=False,
            reward_encoder_copied=True, source_files_untouched=True, checked_unix=time.time())
    proof_path = base / 'preparation/restore-copy-complete.json'
    publish(proof_path, proof)
    verify_target_copy(base, proof_path)
    read_source(pair_path)
    return proof_path

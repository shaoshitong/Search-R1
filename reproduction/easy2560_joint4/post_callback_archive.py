"""Independent CPU archive of an already sealed Easy post-callback boundary.

No model/Adam copying, Torch, RPC, training, or controller mutation is performed.
The source emits eight existing small files. The destination binds them to its
already accepted core archive and publishes a separate child atomically.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time
import uuid

from . import checkpoint_archive as core


STEPS = (5, 50, 100, 150, 200)
ROLE = 'easy2560_post_callback_boundary_source_snapshot'
STATUS = 'EASY2560_POST_CALLBACK_BOUNDARY_ARCHIVE_SHA_FSYNC_PASS'
TRANSFER_STATUS = 'EASY2560_POST_CALLBACK_BOUNDARY_TRANSFER_SHA_PASS'
BOUNDARY_NAMES = frozenset({'manifest.json', 'reward-probe-manifest.json', 'driver.pt'} |
                           {'rank-%d.pt' % rank for rank in range(4)})
NAMES = frozenset({'boundary/'+name for name in BOUNDARY_NAMES} | {'rolling-full-manifest.json'})
MAX_DOC = 8 * 1024**2
MAX_FILE = 128 * 1024**2
MAX_TOTAL = 256 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError('EASY2560_POST_CALLBACK_ARCHIVE_' + message)


def absolute(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'CANONICAL_PATH_REQUIRED')
    return path


def target_path(path):
    """HK target is lexical on HPC; do not probe another machine's filesystem."""
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'ABSOLUTE_CORE_TARGET_REQUIRED')
    return path


def record(path, maximum=MAX_FILE):
    path = absolute(path); before = path.stat()
    need(path.is_file() and 0 < before.st_size <= maximum, 'SMALL_REGULAR_FILE_REQUIRED')
    value = hashlib.sha256()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        need((opened.st_ino, opened.st_size, opened.st_mtime_ns) ==
             (before.st_ino, before.st_size, before.st_mtime_ns), 'FILE_REBOUND')
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
        after = os.fstat(stream.fileno())
    final = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
         (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns), 'FILE_CHANGED')
    return dict(path=str(path), bytes=before.st_size, sha256=value.hexdigest())


def document(path):
    proof = record(path, MAX_DOC)
    raw = Path(path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == proof['sha256'], 'DOCUMENT_CHANGED')
    return json.loads(raw), proof


def phase(step):
    need(type(step) is int and step in STEPS, 'FIXED_MILESTONE_REQUIRED')
    return dict(callbacks_complete=True, reward_refresh_pending=False,
        probe_round=min(step//50, 3)*50, post_callback_rng_captured=True,
        final51713_pending_at_boundary=step == 200,
        sampling_rng_stage=('post_fixed512_pre_final51713' if step == 200 else
                            'post_QA5' if step == 5 else 'post_fixed512_and_grounding_refit'))


def _source_paths(base, step):
    directory = base/'preparation/rolling-checkpoints'/('global_step_%d' % step)
    return {name:directory/('full-manifest.json' if name == 'rolling-full-manifest.json' else name)
            for name in NAMES}


def _shape(value, target=None):
    need(value.get('schema_version') == 1 and value.get('role') == ROLE, 'SNAPSHOT_ROLE_DIFFERS')
    base = absolute(value['base']); step = value['step']; expected_phase = phase(step)
    need(re.fullmatch('[0-9a-f]{32}', value.get('run_id', '')) is not None and
         re.fullmatch('[0-9a-f]{40}', value.get('source_commit', '')) is not None,
         'SOURCE_IDENTITY_REQUIRED')
    destination = target_path(value['core_archive'])
    need(destination.name == 'global_step_%d' % step and destination.parent.name == value['run_id'] and
         (target is None or destination == absolute(target)), 'CORE_TARGET_DIFFERS')
    need(value.get('phase') == expected_phase and value.get('actual_cuda_restore_accepted') is False and
         all(re.fullmatch('[0-9a-f]{64}', value.get(key, '')) is not None for key in
             ('core_checkpoint_complete_sha256', 'core_source_snapshot_sha256', 'core_source_acceptance_sha256')),
         'PHASE_OR_CORE_SHA_DIFFERS')
    files = value.get('files', {})
    need(set(files) == NAMES, 'EXACT_EIGHT_MEMBERS_REQUIRED')
    paths = _source_paths(base, step)
    for name, row in files.items():
        need(row.get('path') == str(paths[name]) and type(row.get('bytes')) is int and
             0 < row['bytes'] <= (MAX_DOC if name.endswith('.json') else MAX_FILE) and
             re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')) is not None, 'MEMBER_PATH_SIZE_OR_SHA_DIFFERS')
    need(value.get('file_count') == 8 and value.get('total_bytes') == sum(r['bytes'] for r in files.values())
         and value['total_bytes'] <= MAX_TOTAL, 'TOTAL_SIZE_DIFFERS')
    return base, step


def _boundary_documents(value, paths):
    base, step = _shape(value)
    rolling, _ = document(paths['rolling-full-manifest.json'])
    boundary, _ = document(paths['boundary/manifest.json'])
    reward, _ = document(paths['boundary/reward-probe-manifest.json'])
    need(rolling.get('manager') == 'easy2560_per_outer_full_state_v1' and
         rolling.get('status') == 'EASY2560_ROLLING_CHECKPOINT_READY' and
         all(rolling.get(k) == value[k] for k in ('base', 'run_id', 'source_commit', 'step')),
         'ROLLING_IDENTITY_DIFFERS')
    checkpoint = base/'checkpoints/search-g1-common8/actor'/('global_step_%d' % step)
    need(rolling.get('checkpoint') == str(checkpoint) and
         rolling['files']['complete.json']['sha256'] == value['core_checkpoint_complete_sha256'],
         'ROLLING_CORE_COMPLETE_DIFFERS')
    need(set(rolling.get('boundary_files', {})) == BOUNDARY_NAMES and rolling.get('boundary') == boundary and
         rolling.get('boundary_manifest') == rolling['boundary_files']['manifest.json'], 'ROLLING_BOUNDARY_DIFFERS')
    for name in BOUNDARY_NAMES:
        need(all(rolling['boundary_files'][name].get(k) == value['files']['boundary/'+name][k]
                 for k in ('path', 'bytes', 'sha256')), 'ROLLING_MEMBER_DIFFERS')
    need(boundary.get('role') == 'easy2560_completed_callback_boundary' and
         boundary.get('run_id') == value['run_id'] and boundary.get('source_commit') == value['source_commit'] and
         boundary.get('completed_outer') == step and boundary.get('callbacks_complete') is True and
         boundary.get('reward_refresh_pending') is False and
         boundary.get('probe_round') == value['phase']['probe_round'], 'CALLBACK_BOUNDARY_DIFFERS')
    workers = boundary.get('worker_files', [])
    need(len(workers) == 4 and sorted(r['rank'] for r in workers) == list(range(4)), 'FOUR_RANKS_REQUIRED')
    for name, proof in [(('rank-%d.pt' % r['rank']), r) for r in workers] + [
            ('driver.pt', boundary['driver_file']), ('reward-probe-manifest.json', boundary['reward_probe_manifest'])]:
        need(all(proof.get(k) == value['files']['boundary/'+name][k] for k in ('path', 'bytes', 'sha256')),
             'BOUNDARY_PROOF_DIFFERS')
    need(reward.get('run_id') == value['run_id'] and reward.get('version') == boundary.get('probe_version') and
         reward.get('round') == value['phase']['probe_round'], 'REWARD_VERSION_DIFFERS')


def core_binding(directory):
    """Read four small existing core documents; never rehash model/Adam bytes."""
    directory = absolute(directory)
    snapshot, snap_proof = document(directory/'source-snapshot.json')
    accepted, accept_proof = document(directory/'source-acceptance.json')
    received, receive_proof = document(directory/'archive-receipt.json')
    complete, complete_proof = document(directory/'checkpoint/complete.json')
    need(snapshot.get('role') == 'easy2560_full_checkpoint_source_snapshot' and
         snapshot.get('training_world_size') == 4 and snapshot.get('question_probe_saved') is False and
         accepted.get('status') == core.STATUS and received.get('status') ==
         'EASY2560_ARCHIVE_RECEIVED_FRESH_SHA_FSYNC_PASS', 'ACCEPTED_CORE_REQUIRED')
    need(directory.name == 'global_step_%d' % snapshot['step'] and directory.parent.name == snapshot['run_id'],
         'CORE_DIRECTORY_IDENTITY_DIFFERS')
    phase(snapshot['step'])
    for other in (accepted, received):
        need(all(other.get(k) == snapshot[k] for k in ('base', 'run_id', 'source_commit', 'step', 'total_bytes')) and
             other.get('snapshot_sha256') == snap_proof['sha256'] and
             other.get('file_count') == len(snapshot['files']) and
             other.get('fresh_destination_sha_verified') is True and other.get('destination_fsynced') is True and
             other.get('actual_cuda_restore_accepted') is False, 'CORE_RECEIPT_BINDING_DIFFERS')
    need(accepted.get('fresh_source_after_archive_sha_verified') is True and
         complete.get('run_id') == snapshot['run_id'] and complete.get('step') == snapshot['step'] and
         complete.get('finite_all_saved_tensors') is True and
         snapshot['files']['checkpoint/complete.json']['sha256'] == complete_proof['sha256'],
         'CORE_COMPLETE_SHA_DIFFERS')
    return dict(core_archive=str(directory), base=snapshot['base'], run_id=snapshot['run_id'],
        source_commit=snapshot['source_commit'], step=snapshot['step'],
        core_checkpoint_complete_sha256=complete_proof['sha256'],
        core_source_snapshot_sha256=snap_proof['sha256'], core_source_acceptance_sha256=accept_proof['sha256'],
        core_archive_receipt_sha256=receive_proof['sha256'])


def _bound_core(value, directory):
    binding = core_binding(directory)
    need(all(value.get(k) == binding[k] for k in ('core_archive', 'base', 'run_id', 'source_commit',
            'step', 'core_checkpoint_complete_sha256', 'core_source_snapshot_sha256',
            'core_source_acceptance_sha256')),
         'SOURCE_AND_ACCEPTED_CORE_DIFFER')
    return binding


def _source_core(base, run, step, checkpoint_sha):
    """Authenticate the existing source ACK locally; HK files are not opened."""
    directory = base/'preparation/checkpoint-archives'/('global_step_%d' % step)
    source, proof = document(directory/'source-snapshot.json')
    accepted, accepted_proof = document(directory/'complete.json')
    need(source.get('role') == 'easy2560_full_checkpoint_source_snapshot' and
         source.get('training_world_size') == 4 and source.get('question_probe_saved') is False and
         accepted.get('status') == core.STATUS and accepted.get('fresh_source_after_archive_sha_verified') is True and
         accepted.get('fresh_destination_sha_verified') is True and accepted.get('destination_fsynced') is True and
         accepted.get('actual_cuda_restore_accepted') is False, 'ACCEPTED_SOURCE_CORE_REQUIRED')
    need(source.get('base') == str(base) and source.get('step') == step and
         all(source.get(k) == run[k] for k in ('run_id', 'source_commit')) and
         all(accepted.get(k) == source[k] for k in ('base', 'run_id', 'source_commit', 'step', 'total_bytes')) and
         accepted.get('snapshot_sha256') == proof['sha256'] and accepted.get('file_count') == len(source['files']) and
         accepted.get('source_snapshot') == str(directory/'source-snapshot.json') and
         source['files']['checkpoint/complete.json']['sha256'] == checkpoint_sha, 'SOURCE_CORE_ACK_DIFFERS')
    return dict(core_source_snapshot_sha256=proof['sha256'], core_source_acceptance_sha256=accepted_proof['sha256'])


def snapshot(base, step, core_archive, output=None):
    base = absolute(base); phase(step)
    run, _ = document(base/'logs/run-manifest.json')
    need(run.get('easy2560_selection') is True and run.get('training_world_size') == 4 and
         run.get('question_probe') is None, 'EASY_NATIVE4_REQUIRED')
    rolling_path = base/'preparation/rolling-checkpoints'/('global_step_%d' % step)/'full-manifest.json'
    from search_g1.easy2560_rolling_checkpoint import verify_checkpoint_manifest
    verify_checkpoint_manifest(base, rolling_path, expected_run_id=run['run_id'], full_sha=False)
    checkpoint = base/'checkpoints/search-g1-common8/actor'/('global_step_%d' % step)
    complete, proof = document(checkpoint/'complete.json')
    need(complete.get('run_id') == run['run_id'] and complete.get('step') == step and
         complete.get('finite_all_saved_tensors') is True, 'SOURCE_CORE_NOT_COMPLETE')
    paths = _source_paths(base, step)
    files = {name:record(path, MAX_DOC if name.endswith('.json') else MAX_FILE) for name, path in paths.items()}
    value = dict(schema_version=1, role=ROLE, base=str(base), run_id=run['run_id'],
        source_commit=run['source_commit'], step=step, core_archive=str(target_path(core_archive)),
        core_checkpoint_complete_sha256=proof['sha256'], phase=phase(step), files=files,
        file_count=8, total_bytes=sum(row['bytes'] for row in files.values()),
        actual_cuda_restore_accepted=False, checked_unix=time.time())
    value.update(_source_core(base, run, step, proof['sha256']))
    _shape(value); _boundary_documents(value, paths)
    parent = base/'preparation/post-callback-archives'/('global_step_%d' % step)
    output = absolute(output or parent/'source-snapshot.json')
    need(output.parent == parent and output.name == 'source-snapshot.json', 'SOURCE_OUTPUT_SCOPE_DIFFERS')
    core.publish(output, value)
    return dict(snapshot=str(output), snapshot_sha256=core.sha(output), file_count=8,
                total_bytes=value['total_bytes'], phase=value['phase'])


def _source_snapshot(path):
    value, _ = document(path); _shape(value)
    paths = {name:Path(row['path']) for name, row in value['files'].items()}
    for name, source in paths.items():
        need(record(source) == value['files'][name], 'SOURCE_MEMBER_SHA_DIFFERS')
    _boundary_documents(value, paths)
    return value


def emit(snapshot_path, output):
    _source_snapshot(snapshot_path)
    core.emit(snapshot_path, output)
    _source_snapshot(snapshot_path)


def _fsync_tree(directory):
    for folder, _, _ in os.walk(directory, topdown=False):
        fd = os.open(folder, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def _verify_stage(directory, value):
    paths = {name:directory/name for name in NAMES}
    for name, path in paths.items():
        proof = record(path)
        need(all(proof[k] == value['files'][name][k] for k in ('bytes', 'sha256')), 'DESTINATION_SHA_DIFFERS')
    _boundary_documents(value, paths)


def receive(core_archive, stream):
    directory = absolute(core_archive); binding = core_binding(directory)
    destination = directory/'post-callback-boundary'
    need(not destination.exists() and not destination.is_symlink(), 'DESTINATION_EXISTS_INSPECT_RECEIPT')
    stage = directory/('.post-callback-boundary.partial-'+uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    # Failure deliberately leaves only this unique staging child for inspection.
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        seen = set(); value = None
        for member in archive:
            name = core.safe_name(member.name)
            need(member.isfile() and name not in seen, 'REGULAR_UNIQUE_MEMBERS_REQUIRED')
            seen.add(name); content = archive.extractfile(member)
            if value is None:
                need(name == 'source-snapshot.json' and 0 < member.size <= MAX_DOC, 'SNAPSHOT_FIRST_REQUIRED')
                raw = content.read(); value = json.loads(raw)
                need(core.encoded(value) == raw, 'CANONICAL_SNAPSHOT_REQUIRED')
                _shape(value, directory); need(_bound_core(value, directory) == binding, 'CORE_CHANGED')
                core.publish(stage/name, value)
                continue
            need(name in NAMES and member.size == value['files'][name]['bytes'], 'UNEXPECTED_MEMBER_OR_SIZE')
            target = stage/name; target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256(); count = 0
            with target.open('xb') as out:
                for block in iter(lambda: content.read(1024**2), b''):
                    out.write(block); h.update(block); count += len(block)
                out.flush(); os.fsync(out.fileno())
            need(count == member.size and h.hexdigest() == value['files'][name]['sha256'], 'RECEIVED_SHA_DIFFERS')
    need(value is not None and seen == NAMES | {'source-snapshot.json'}, 'INCOMPLETE_STREAM')
    _verify_stage(stage, value)
    receipt = dict(status=TRANSFER_STATUS, source_snapshot_sha256=core.sha(stage/'source-snapshot.json'),
        binding=binding, file_count=8, total_bytes=value['total_bytes'], phase=value['phase'],
        fresh_destination_sha_verified=True, actual_cuda_restore_accepted=False, checked_unix=time.time())
    core.publish(stage/'transfer-receipt.json', receipt); _fsync_tree(stage)
    return dict(stage=str(stage), status=TRANSFER_STATUS, file_count=8, total_bytes=value['total_bytes'])


def accept(core_archive, stage):
    directory = absolute(core_archive); stage = absolute(stage)
    need(stage.parent == directory and re.fullmatch(r'\.post-callback-boundary\.partial-[0-9a-f]{32}', stage.name)
         is not None, 'STAGING_SCOPE_DIFFERS')
    lock_path = directory.parent/('.post-callback-'+directory.name+'.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = directory/'post-callback-boundary'
        need(not destination.exists() and not destination.is_symlink(), 'DESTINATION_EXISTS_INSPECT_RECEIPT')
        value, _ = document(stage/'source-snapshot.json'); _shape(value, directory)
        transfer, _ = document(stage/'transfer-receipt.json')
        binding = _bound_core(value, directory)
        need(transfer.get('status') == TRANSFER_STATUS and transfer.get('binding') == binding and
             transfer.get('source_snapshot_sha256') == core.sha(stage/'source-snapshot.json') and
             transfer.get('file_count') == 8 and transfer.get('total_bytes') == value['total_bytes'] and
             transfer.get('phase') == value['phase'] and transfer.get('fresh_destination_sha_verified') is True and
             transfer.get('actual_cuda_restore_accepted') is False, 'TRANSFER_RECEIPT_DIFFERS')
        actual = {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file() or p.is_symlink()}
        need(actual == NAMES | {'source-snapshot.json', 'transfer-receipt.json'}, 'STAGING_MEMBERS_CHANGED')
        _verify_stage(stage, value)
        receipt = dict(status=STATUS, binding=binding, phase=value['phase'], destination=str(destination),
            source_snapshot_sha256=transfer['source_snapshot_sha256'], file_count=8, total_bytes=value['total_bytes'],
            files={name:{k:row[k] for k in ('bytes', 'sha256')} for name, row in value['files'].items()},
            fresh_destination_sha_verified=True, destination_fsynced=True, core_payload_unmodified=True,
            actual_cuda_restore_accepted=False, actual_model_optimizer_rng_dataequal_accepted=False,
            checked_unix=time.time(), scope='Preserved producer post-callback bytes; no actual reload performed')
        core.publish(stage/'boundary-archive-receipt.json', receipt); _fsync_tree(stage)
        os.rename(stage, destination)
        fd = os.open(directory, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    source = sub.add_parser('snapshot'); source.add_argument('--base', required=True)
    source.add_argument('--step', type=int, choices=STEPS, required=True)
    source.add_argument('--core-archive', required=True); source.add_argument('--output')
    sender = sub.add_parser('emit'); sender.add_argument('--snapshot', required=True)
    receiver = sub.add_parser('receive'); receiver.add_argument('--core-archive', required=True)
    accepted = sub.add_parser('accept'); accepted.add_argument('--core-archive', required=True)
    accepted.add_argument('--stage', required=True)
    args = parser.parse_args()
    if args.mode == 'emit': emit(args.snapshot, sys.stdout.buffer); return
    if args.mode == 'snapshot': result = snapshot(args.base, args.step, args.core_archive, args.output)
    elif args.mode == 'receive': result = receive(args.core_archive, sys.stdin.buffer)
    else: result = accept(args.core_archive, args.stage)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__': main()

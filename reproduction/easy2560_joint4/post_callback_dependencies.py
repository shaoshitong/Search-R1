"""Independent closure of a sealed post-callback grounding version.

The accepted core and eight-file boundary remain immutable. Small grounding
dependencies missing from that core are streamed into a separate child. Every
encoder file must map by name/bytes/SHA to an existing accepted core member;
model weights are never transferred by this module. No Torch/model/RPC import.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time
import uuid

from . import post_callback_archive as boundary

core = boundary.core
ROLE = 'easy2560_post_callback_grounding_dependencies_snapshot'
STATUS = 'EASY2560_POST_CALLBACK_DEPENDENCIES_SHA_FSYNC_PASS'
TRANSFER_STATUS = 'EASY2560_POST_CALLBACK_DEPENDENCIES_TRANSFER_SHA_PASS'
CHILD = 'post-callback-dependencies'
MAX_DOC = 8 * 1024**2
MAX_HEAD = 16 * 1024**2
MAX_DUMP = 2 * 1024**3
MAX_TRANSFER = 4 * 1024**3
MAX_DESCRIPTOR = 16 * 1024**2


def need(ok, reason):
    if not ok:
        raise ValueError('EASY2560_POST_CALLBACK_DEPENDENCIES_' + reason)


def lexical(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'ABSOLUTE_PATH_REQUIRED')
    return path


def canonical(path):
    return boundary.absolute(path)


def _limit(name):
    if name.startswith('grounding/dumps/') and name.endswith('.npz'):
        return MAX_DUMP
    return MAX_HEAD if name.endswith('.pkl') else MAX_DOC


def _record(path, maximum=None):
    path = canonical(path); before = path.stat()
    need(path.is_file() and 0 < before.st_size and
         (maximum is None or before.st_size <= maximum), 'FILE_SIZE_OR_TYPE_DIFFERS')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        need((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) ==
             (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns), 'FILE_REBOUND')
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
    final = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    need(identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
         (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns), 'FILE_CHANGED')
    return dict(path=str(path), bytes=before.st_size, sha256=digest.hexdigest(),
                inode=before.st_ino, mtime_ns=before.st_mtime_ns)


def _unchanged(row):
    path = canonical(row['path']); seen = path.stat()
    need(path.is_file() and (seen.st_ino, seen.st_size, seen.st_mtime_ns) ==
         (row['inode'], row['bytes'], row['mtime_ns']), 'SOURCE_IDENTITY_CHANGED')


def _core_encoder_record(path, expected, source):
    """Reuse the accepted fresh-source ACK, without rereading model tensor bodies."""
    path = canonical(path); seen = path.stat()
    need(path.is_file() and seen.st_size > 0, 'ENCODER_FILE_REQUIRED')
    matches = [name for name, row in source['files'].items()
               if row.get('path') == str(path) and row.get('sha256') == expected and row.get('bytes') == seen.st_size]
    need(bool(matches), 'ENCODER_NOT_COVERED_BY_ACCEPTED_CORE')
    row = dict(path=str(path), bytes=seen.st_size, sha256=expected, inode=seen.st_ino,
               mtime_ns=seen.st_mtime_ns, source_sha_basis='accepted_core_fresh_source_ack')
    return row, sorted(matches)[0]


def _json(path):
    proof = _record(path, MAX_DESCRIPTOR)
    raw = Path(path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == proof['sha256'], 'JSON_CHANGED')
    return json.loads(raw), proof


def _same(a, b):
    return all(a.get(key) == b.get(key) for key in ('bytes', 'sha256'))


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _reward(probe, base, step):
    expected_round = boundary.phase(step)['probe_round']
    need(probe.get('round') == expected_round and isinstance(probe.get('version'), str) and
         bool(probe['version']), 'EXACT_REWARD_ROUND_REQUIRED')
    commit = lexical(probe['commit']); folder = commit.parent.parent
    need(folder.parent == base/'artifacts' and folder.name.startswith('round_%d-' % expected_round) and
         commit == folder/'heads/commit_probe.pkl' and
         lexical(probe['boundary']) == folder/'heads/boundary_probe.pkl', 'REWARD_PATH_SCOPE_DIFFERS')
    heads = probe.get('head_sha256', {})
    need(set(heads) == {'commit_probe.pkl', 'boundary_probe.pkl'} and
         all(_digest(v) for v in heads.values()) and _digest(probe.get('calibration_report_sha256')),
         'HEAD_REPORT_SHA_REQUIRED')
    dumps = probe.get('dump_sha256', {})
    need(isinstance(dumps, dict) and bool(dumps), 'COMPLETE_DUMP_SHA_REQUIRED')
    for name, digest in dumps.items():
        need(isinstance(name, str) and core.safe_name(name) == name and Path(name).name == name and
             name not in ('.', '..') and _digest(digest), 'DUMP_NAME_OR_SHA_DIFFERS')
    encoder = probe.get('encoder_sha256', {})
    need(isinstance(encoder, dict) and bool(encoder), 'ENCODER_SHA_REQUIRED')
    for name, digest in encoder.items():
        need(core.safe_name(name) == name and _digest(digest), 'ENCODER_NAME_OR_SHA_DIFFERS')
    lexical(probe['encoder'])
    return folder


def _paths(probe, base, step, origin=None):
    folder = _reward(probe, base, step)
    result = {'grounding/version-manifest.json': (folder/'version-manifest.json', None),
              'grounding/heads/commit_probe.pkl': (folder/'heads/commit_probe.pkl', probe['head_sha256']['commit_probe.pkl']),
              'grounding/heads/boundary_probe.pkl': (folder/'heads/boundary_probe.pkl', probe['head_sha256']['boundary_probe.pkl']),
              'grounding/heads/calibration-report.json': (folder/'heads/calibration-report.json', probe['calibration_report_sha256'])}
    dump_root = lexical(origin['dump_directory']) if origin else folder/'dumps'
    result.update({'grounding/dumps/'+name: (dump_root/name, digest)
                   for name, digest in probe['dump_sha256'].items()})
    if origin:
        result.update({name:(lexical(row['path']), row['sha256']) for name, row in origin['proofs'].items()})
    return result


ORIGIN_RECEIPT = 'provenance/restore-source-receipt.json'
ORIGIN_REWARD = 'provenance/origin-reward-manifest.json'
ORIGIN_VERSION = 'provenance/origin-version-manifest.json'


def _same_reward(origin, restored):
    # install_restored_reward_probe changes only these two filesystem paths.
    return {k:v for k,v in origin.items() if k not in ('commit','boundary')} == {
        k:v for k,v in restored.items() if k not in ('commit','boundary')}


def _origin_binding(origin, run, probe, base, step):
    need(isinstance(origin, dict) and origin.get('kind') == 'sha_bound_easy_restore_single_hop', 'ORIGIN_KIND_DIFFERS')
    restore = run.get('full_state_restore', {})
    need(restore.get('kind') == 'easy2560_native4' and
         restore.get('source_receipt') == origin.get('receipt_path') and
         restore.get('source_receipt_sha256') == origin.get('receipt_sha256') and
         restore.get('source_base') == origin.get('source_base') and
         lexical(origin['source_base']) != base, 'ORIGIN_NOT_BOUND_BY_CURRENT_RUN')
    proofs = origin.get('proofs', {})
    need(set(proofs) == {ORIGIN_RECEIPT, ORIGIN_REWARD, ORIGIN_VERSION} and
         proofs[ORIGIN_RECEIPT] == dict(path=origin['receipt_path'], sha256=origin['receipt_sha256']) and
         all(_digest(row.get('sha256')) and lexical(row['path']) for row in proofs.values()), 'ORIGIN_PROOFS_DIFFER')
    folder = lexical(origin['dump_directory']).parent
    need(lexical(origin['dump_directory']) == folder/'dumps' and
         folder.parent == lexical(origin['source_base'])/'artifacts' and
         folder.name.startswith('round_%d-' % boundary.phase(step)['probe_round']) and
         proofs[ORIGIN_VERSION]['path'] == str(folder/'version-manifest.json'), 'ORIGIN_DUMP_SCOPE_DIFFERS')


def _find_origin(base, step, probe, run):
    """One authenticated restore hop only; never scan other runs or silently drop dumps."""
    restore = run.get('full_state_restore', {})
    need(restore.get('kind') == 'easy2560_native4', 'MISSING_DUMPS_WITHOUT_BOUND_RESTORE_ORIGIN')
    receipt_path = canonical(restore['source_receipt']); receipt, proof = _json(receipt_path)
    need(proof['sha256'] == restore['source_receipt_sha256'] and
         receipt.get('kind') == 'easy2560_native4' and
         receipt.get('status') == 'EASY2560_NATIVE4_FULL_SHA_CPU_STATES_PASS' and
         receipt.get('run_id') == probe['run_id'] and receipt.get('source_base') == restore.get('source_base'),
         'ORIGIN_RECEIPT_SHA_OR_IDENTITY_DIFFERS')
    record = receipt['reward_probe_manifest']; reward_path = canonical(record['path'])
    old, reward_proof = _json(reward_path)
    need(reward_proof['sha256'] == record.get('full_read_sha256', record.get('sha256')) and
         _same_reward(old, probe), 'ORIGIN_REWARD_SHA_OR_VERSION_DIFFERS')
    folder = _reward(old, lexical(receipt['source_base']), step)
    original, version_proof = _json(folder/'version-manifest.json')
    need(original == old and all((folder/'dumps'/name).is_file() for name in old['dump_sha256']),
         'SINGLE_HOP_ORIGIN_DUMPS_MISSING_OR_VERSION_DIFFERS')
    origin = dict(kind='sha_bound_easy_restore_single_hop', receipt_path=str(receipt_path),
        receipt_sha256=proof['sha256'], source_base=receipt['source_base'], dump_directory=str(folder/'dumps'),
        proofs={ORIGIN_RECEIPT:dict(path=str(receipt_path), sha256=proof['sha256']),
                ORIGIN_REWARD:dict(path=str(reward_path), sha256=reward_proof['sha256']),
                ORIGIN_VERSION:dict(path=str(folder/'version-manifest.json'), sha256=version_proof['sha256'])})
    _origin_binding(origin, run, probe, base, step)
    return origin


def _source_boundary(base, step):
    path = base/'preparation/post-callback-archives'/('global_step_%d' % step)/'source-snapshot.json'
    value = boundary._source_snapshot(path)
    need(value['base'] == str(base) and value['step'] == step, 'SOURCE_BOUNDARY_DIFFERS')
    probe_path = Path(value['files']['boundary/reward-probe-manifest.json']['path'])
    probe, proof = _json(probe_path)
    need(proof['sha256'] == value['files']['boundary/reward-probe-manifest.json']['sha256'] and
         probe['run_id'] == value['run_id'], 'SOURCE_REWARD_DIFFERS')
    run, _ = _json(base/'logs/run-manifest.json')
    ack = boundary._source_core(base, run, step, value['core_checkpoint_complete_sha256'])
    need(all(value[k] == ack[k] for k in ack), 'SOURCE_CORE_ACK_CHANGED')
    source, _ = _json(base/'preparation/checkpoint-archives'/('global_step_%d' % step)/'source-snapshot.json')
    return value, probe, core.sha(path), source


def _accepted_boundary(directory, verify=True):
    directory = canonical(directory); binding = boundary.core_binding(directory)
    child = canonical(directory/'post-callback-boundary')
    value, proof = _json(child/'source-snapshot.json')
    boundary._shape(value, directory); boundary._bound_core(value, directory)
    receipt, receipt_proof = _json(child/'boundary-archive-receipt.json')
    need(receipt.get('status') == boundary.STATUS and receipt.get('binding') == binding and
         receipt.get('source_snapshot_sha256') == proof['sha256'] and
         receipt.get('file_count') == 8 and receipt.get('phase') == value['phase'] and
         receipt.get('fresh_destination_sha_verified') is True and receipt.get('destination_fsynced') is True and
         receipt.get('actual_cuda_restore_accepted') is False and
         set(receipt.get('files', {})) == boundary.NAMES and
         all(_same(receipt['files'][name], row) for name, row in value['files'].items()),
         'ACCEPTED_EIGHT_FILE_BOUNDARY_REQUIRED')
    if verify:
        boundary._verify_stage(child, value)
    probe, probe_proof = _json(child/'boundary/reward-probe-manifest.json')
    need(probe_proof['sha256'] == value['files']['boundary/reward-probe-manifest.json']['sha256'],
         'ARCHIVED_REWARD_MANIFEST_DIFFERS')
    manifest, _ = _json(child/'boundary/manifest.json')
    need(probe['run_id'] == value['run_id'] and probe['version'] == manifest['probe_version'] and
         probe['round'] == manifest['probe_round'], 'ARCHIVED_REWARD_IDENTITY_DIFFERS')
    return dict(core=binding, boundary_source_snapshot_sha256=proof['sha256'],
                boundary_archive_receipt_sha256=receipt_proof['sha256'],
                boundary_reward_manifest_sha256=probe_proof['sha256']), value, probe


def _cover(row, source, preferred=()):
    matches = [name for name, item in source['files'].items()
               if item.get('path') == row['path'] and _same(row, item)]
    if not matches:
        return None
    exact = [name for name in matches if source['files'][name].get('path') == row['path']]
    return next((name for name in preferred if name in matches), None) or sorted(exact or matches)[0]


def _shape(value, probe, source, directory, run):
    need(value.get('schema_version') == 1 and value.get('role') == ROLE, 'DESCRIPTOR_ROLE_DIFFERS')
    base = lexical(value['base']); step = value['step']; boundary.phase(step)
    need(value.get('core_archive') == str(directory) and value['run_id'] == probe['run_id'] and
         value['reward_version'] == probe['version'] and value['reward_round'] == probe['round'] and
         value.get('actual_cuda_restore_accepted') is False, 'DESCRIPTOR_IDENTITY_DIFFERS')
    origin = value.get('origin')
    if origin: _origin_binding(origin, run, probe, base, step)
    required = _paths(probe, base, step, origin)
    need(set(value.get('dependencies', {})) == set(required), 'EXACT_DEPENDENCIES_REQUIRED')
    transfer = 0
    for name, (path, expected) in required.items():
        row = value['dependencies'][name]
        need(row.get('path') == str(path) and type(row.get('bytes')) is int and
             0 < row['bytes'] <= _limit(name) and _digest(row.get('sha256')) and
             (expected is None or row['sha256'] == expected), 'DEPENDENCY_PATH_SIZE_SHA_DIFFERS')
        if row.get('storage') == 'core':
            member = core.safe_name(row.get('core_member', ''))
            need(member in source['files'] and source['files'][member].get('path') == row['path'] and
                 _same(row, source['files'][member]), 'CORE_COVERAGE_DIFFERS')
        else:
            need(row.get('storage') == 'transfer' and 'core_member' not in row, 'DEPENDENCY_STORAGE_DIFFERS')
            transfer += row['bytes']
    need(set(value.get('encoder_files', {})) == set(probe['encoder_sha256']), 'ALL_ENCODER_FILES_REQUIRED')
    for name, expected in probe['encoder_sha256'].items():
        row = value['encoder_files'][name]; member = core.safe_name(row.get('core_member', ''))
        need(row.get('path') == str(lexical(probe['encoder'])/name) and row.get('sha256') == expected and
             row.get('storage') == 'core' and member in source['files'] and
             source['files'][member].get('path') == row['path'] and
             _same(row, source['files'][member]), 'ENCODER_CORE_COVERAGE_MISSING')
    need(value.get('transfer_bytes') == transfer and transfer <= MAX_TRANSFER and
         sum(row['bytes'] for row in value['dependencies'].values()) <= MAX_TRANSFER and
         value.get('file_count') == len(required) and
         value.get('transfer_file_count') == sum(row['storage'] == 'transfer' for row in value['dependencies'].values()),
         'TRANSFER_TOTAL_DIFFERS')
    return required


def snapshot(base, step, core_archive, output=None):
    base = canonical(base); boundary.phase(step); destination = boundary.target_path(core_archive)
    parent = base/'preparation/post-callback-dependencies'/('global_step_%d' % step)
    output = canonical(output or parent/'source-snapshot.json')
    need(output.parent == parent and output.name == 'source-snapshot.json', 'SOURCE_OUTPUT_SCOPE_DIFFERS')
    parent.mkdir(parents=True, exist_ok=True)
    with (parent/'source.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not any((parent/name).exists() for name in ('source-claim.json', 'source-snapshot.json', 'source-failure.json')),
             'SOURCE_ALREADY_ATTEMPTED_NO_REPLAY')
        old, probe, old_sha, source = _source_boundary(base, step)
        need(old['core_archive'] == str(destination), 'SOURCE_CORE_TARGET_DIFFERS')
        core.publish(parent/'source-claim.json', dict(base=str(base), step=step, run_id=old['run_id'],
            boundary_source_snapshot_sha256=old_sha, pid=os.getpid(), checked_unix=time.time()))
        try:
            run, run_proof = _json(base/'logs/run-manifest.json')
            need(_same(run_proof, source['files']['metadata/run-manifest.json']) and
                 source['files']['metadata/run-manifest.json']['path'] == str(base/'logs/run-manifest.json'),
                 'SOURCE_RUN_NOT_COVERED_BY_CORE')
            folder = _reward(probe, base, step)
            origin = None
            if not all((folder/'dumps'/name).is_file() for name in probe['dump_sha256']):
                origin = _find_origin(base, step, probe, run)
            dependencies = {}
            for name, (path, expected) in _paths(probe, base, step, origin).items():
                row = _record(path, _limit(name))
                need(expected is None or row['sha256'] == expected, 'SOURCE_DEPENDENCY_SHA_DIFFERS')
                member = _cover(row, source)
                dependencies[name] = dict(row, storage='transfer') if member is None else dict(row, storage='core', core_member=member)
            original, _ = _json(Path(dependencies['grounding/version-manifest.json']['path']))
            need(original == probe, 'ORIGINAL_VERSION_MANIFEST_DIFFERS')
            encoders = {}
            for name, expected in sorted(probe['encoder_sha256'].items()):
                row, member = _core_encoder_record(lexical(probe['encoder'])/name, expected, source)
                encoders[name] = dict(row, storage='core', core_member=member)
            core_keys = ('core_archive', 'base', 'run_id', 'source_commit', 'step', 'core_checkpoint_complete_sha256',
                         'core_source_snapshot_sha256', 'core_source_acceptance_sha256')
            # Only HK knows the exact receive-receipt bytes; bind its core on receive.
            binding = {key:old[key] for key in core_keys}
            binding.pop('base'); binding.pop('run_id'); binding.pop('source_commit'); binding.pop('step')
            value = dict(schema_version=1, role=ROLE, **{key:old[key] for key in ('base', 'run_id', 'source_commit', 'step', 'core_archive')},
                core_source_binding=binding, boundary_source_snapshot_sha256=old_sha,
                boundary_reward_manifest_sha256=old['files']['boundary/reward-probe-manifest.json']['sha256'],
                reward_version=probe['version'], reward_round=probe['round'], origin=origin,
                dependencies=dependencies, encoder_files=encoders,
                file_count=len(dependencies), transfer_file_count=sum(r['storage'] == 'transfer' for r in dependencies.values()),
                transfer_bytes=sum(r['bytes'] for r in dependencies.values() if r['storage'] == 'transfer'),
                actual_cuda_restore_accepted=False, checked_unix=time.time())
            _shape(value, probe, source, destination, run)
            for row in list(dependencies.values()) + list(encoders.values()): _unchanged(row)
            core.publish(output, value)
            return dict(snapshot=str(output), snapshot_sha256=core.sha(output), file_count=value['file_count'],
                        transfer_file_count=value['transfer_file_count'], transfer_bytes=value['transfer_bytes'],
                        encoder_file_count=len(encoders), reward_version=probe['version'], reward_round=probe['round'])
        except BaseException as error:
            core.publish(parent/'source-failure.json', dict(error_type=type(error).__name__, error=str(error), checked_unix=time.time()))
            raise


def _source_snapshot(path):
    value, _ = _json(path); base = canonical(value['base'])
    old, probe, old_sha, source = _source_boundary(base, value['step'])
    need(old_sha == value['boundary_source_snapshot_sha256'] and old['core_archive'] == value['core_archive'] and
         probe['version'] == value['reward_version'], 'SOURCE_BOUNDARY_CHANGED')
    run, run_proof = _json(base/'logs/run-manifest.json')
    need(_same(run_proof, source['files']['metadata/run-manifest.json']), 'SOURCE_RUN_MANIFEST_CHANGED')
    need(all(value['core_source_binding'].get(key) == old[key] for key in
             ('core_archive','core_checkpoint_complete_sha256','core_source_snapshot_sha256','core_source_acceptance_sha256')),
         'SOURCE_CORE_BINDING_CHANGED')
    _shape(value, probe, source, boundary.target_path(value['core_archive']), run)
    for row in list(value['dependencies'].values()) + list(value['encoder_files'].values()): _unchanged(row)
    return value


def emit(snapshot_path, output):
    path = canonical(snapshot_path); value = _source_snapshot(path)
    with (path.parent/'emit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not any((path.parent/name).exists() for name in ('emit-claim.json', 'emit-result.json', 'emit-failure.json')),
             'EMIT_ALREADY_ATTEMPTED_NO_REPLAY')
        core.publish(path.parent/'emit-claim.json', dict(snapshot_sha256=core.sha(path), pid=os.getpid(), checked_unix=time.time()))
        try:
            with tarfile.open(fileobj=output, mode='w|') as archive:
                raw = core.encoded(value); header = tarfile.TarInfo('source-snapshot.json'); header.size = len(raw)
                archive.addfile(header, io.BytesIO(raw))
                for name, row in sorted(value['dependencies'].items()):
                    if row['storage'] != 'transfer': continue
                    _unchanged(row)
                    with Path(row['path']).open('rb') as stream:
                        reader = core.HashReader(stream); header = tarfile.TarInfo(name); header.size = row['bytes']
                        archive.addfile(header, reader)
                    need(reader.count == row['bytes'] and reader.digest.hexdigest() == row['sha256'], 'EMIT_SHA_DIFFERS')
            output.flush(); _source_snapshot(path)
            core.publish(path.parent/'emit-result.json', dict(snapshot_sha256=core.sha(path), complete=True, checked_unix=time.time()))
        except BaseException as error:
            core.publish(path.parent/'emit-failure.json', dict(error_type=type(error).__name__, error=str(error), checked_unix=time.time()))
            raise


def _bound(directory, value):
    binding, old, probe = _accepted_boundary(directory)
    need(all(value.get(k) == old[k] for k in ('base', 'run_id', 'source_commit', 'step', 'core_archive')) and
         value.get('boundary_source_snapshot_sha256') == binding['boundary_source_snapshot_sha256'] and
         value.get('boundary_reward_manifest_sha256') == binding['boundary_reward_manifest_sha256'], 'BOUNDARY_BINDING_DIFFERS')
    need(all(value['core_source_binding'].get(key) == binding['core'][key]
             for key in ('core_archive', 'core_checkpoint_complete_sha256', 'core_source_snapshot_sha256', 'core_source_acceptance_sha256')),
         'CORE_SOURCE_BINDING_DIFFERS')
    source, _ = _json(directory/'source-snapshot.json')
    run, run_proof = _json(directory/'metadata/run-manifest.json')
    need(_same(run_proof, source['files']['metadata/run-manifest.json']) and
         source['files']['metadata/run-manifest.json']['path'] == str(lexical(value['base'])/'logs/run-manifest.json'),
         'ARCHIVED_RUN_MANIFEST_SHA_DIFFERS')
    _shape(value, probe, source, directory, run)
    return binding, probe


def _verify_files(directory, stage, value, probe, *, rehash=True, deep=False):
    mappings = {}; cache = {}
    def inspect_file(path, row, covered):
        path = canonical(path)
        need(path.is_file() and path.stat().st_size == row['bytes'], 'DESTINATION_SIZE_DIFFERS')
        if rehash and (not covered or deep):
            if str(path) not in cache: cache[str(path)] = _record(path)
            need(_same(cache[str(path)], row), 'FRESH_DESTINATION_SHA_DIFFERS')
        seen = path.stat()
        return str(path), dict(inode=seen.st_ino, bytes=seen.st_size, mtime_ns=seen.st_mtime_ns)
    for name, row in value['dependencies'].items():
        target = directory/row['core_member'] if row['storage'] == 'core' else stage/name
        path, identity = inspect_file(target, row, row['storage'] == 'core')
        mappings[name] = dict(source_path=row['path'], archive_path=path, archive_identity=identity,
                              bytes=row['bytes'], sha256=row['sha256'], storage=row['storage'],
                              sha_basis='accepted_core_fresh_ack' if row['storage'] == 'core' else 'fresh_transferred_readback')
    for name, row in value['encoder_files'].items():
        target = directory/row['core_member']
        path, identity = inspect_file(target, row, True)
        mappings['encoder/'+name] = dict(source_path=row['path'], archive_path=path, archive_identity=identity,
            bytes=row['bytes'], sha256=row['sha256'], storage='core', sha_basis='accepted_core_fresh_ack')
    original, _ = _json(Path(mappings['grounding/version-manifest.json']['archive_path']))
    need(original == probe, 'ARCHIVED_ORIGINAL_VERSION_DIFFERS')
    if value.get('origin'):
        origin = value['origin']
        receipt, _ = _json(Path(mappings[ORIGIN_RECEIPT]['archive_path']))
        old, old_proof = _json(Path(mappings[ORIGIN_REWARD]['archive_path']))
        original, _ = _json(Path(mappings[ORIGIN_VERSION]['archive_path']))
        record = receipt.get('reward_probe_manifest', {})
        need(receipt.get('kind') == 'easy2560_native4' and
             receipt.get('status') == 'EASY2560_NATIVE4_FULL_SHA_CPU_STATES_PASS' and
             receipt.get('source_base') == origin['source_base'] and receipt.get('run_id') == probe['run_id'] and
             record.get('path') == origin['proofs'][ORIGIN_REWARD]['path'] and
             record.get('full_read_sha256', record.get('sha256')) == old_proof['sha256'] and
             _same_reward(old, probe) and original == old and
             str(_reward(old, lexical(origin['source_base']), value['step'])/'dumps') == origin['dump_directory'],
             'ARCHIVED_ORIGIN_PROOF_CHAIN_DIFFERS')
    return mappings


def _tree_fsync(path):
    boundary._fsync_tree(path)


def receive(core_archive, stream):
    directory = canonical(core_archive); binding, _, _ = _accepted_boundary(directory)
    control = directory.parent/('.post-callback-dependencies-'+directory.name)
    control.mkdir(mode=0o700, exist_ok=True); canonical(control)
    with (control/'receive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not any((control/name).exists() for name in ('receive-claim.json', 'receive-failure.json', 'receive-result.json')) and
             not (directory/CHILD).exists(), 'RECEIVE_ALREADY_ATTEMPTED_NO_REPLAY')
        stage = directory/('.post-callback-dependencies.partial-'+uuid.uuid4().hex)
        core.publish(control/'receive-claim.json', dict(binding=binding, stage=str(stage), pid=os.getpid(), checked_unix=time.time()))
        stage.mkdir(mode=0o700)
        try:
            value = None; seen = set()
            with tarfile.open(fileobj=stream, mode='r|') as archive:
                for member in archive:
                    name = core.safe_name(member.name)
                    need(member.isfile() and name not in seen, 'REGULAR_UNIQUE_MEMBERS_REQUIRED'); seen.add(name)
                    content = archive.extractfile(member)
                    if value is None:
                        need(name == 'source-snapshot.json' and 0 < member.size <= MAX_DESCRIPTOR, 'SNAPSHOT_FIRST_REQUIRED')
                        raw = content.read(); value = json.loads(raw)
                        need(core.encoded(value) == raw, 'CANONICAL_DESCRIPTOR_REQUIRED')
                        actual, probe = _bound(directory, value); need(actual == binding, 'BOUNDARY_CHANGED')
                        core.publish(stage/name, value); continue
                    need(name in value['dependencies'] and value['dependencies'][name]['storage'] == 'transfer' and
                         member.size == value['dependencies'][name]['bytes'], 'UNEXPECTED_MEMBER_OR_SIZE')
                    target = stage/name; target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256(); count = 0
                    with target.open('xb') as out:
                        for block in iter(lambda: content.read(8 * 1024**2), b''):
                            out.write(block); digest.update(block); count += len(block)
                        out.flush(); os.fsync(out.fileno())
                    need(count == member.size and digest.hexdigest() == value['dependencies'][name]['sha256'], 'RECEIVE_SHA_DIFFERS')
            need(value is not None and seen == {'source-snapshot.json'} | {
                name for name, row in value['dependencies'].items() if row['storage'] == 'transfer'}, 'INCOMPLETE_STREAM')
            mappings = _verify_files(directory, stage, value, probe)
            receipt = dict(status=TRANSFER_STATUS, binding=binding, source_snapshot_sha256=core.sha(stage/'source-snapshot.json'),
                file_count=value['file_count'], transfer_file_count=value['transfer_file_count'], transfer_bytes=value['transfer_bytes'],
                encoder_file_count=len(value['encoder_files']), reward_version=probe['version'], reward_round=probe['round'],
                fresh_transferred_destination_sha_verified=True, fresh_transferred_dependency_sha_checked=True,
                covered_core_sha_ack_verified=True,
                core_covered_payload_rehashed=False, actual_cuda_restore_accepted=False, mappings=mappings, checked_unix=time.time())
            core.publish(stage/'transfer-receipt.json', receipt); _tree_fsync(stage)
            result = dict(stage=str(stage), status=TRANSFER_STATUS, file_count=value['file_count'],
                          transfer_file_count=value['transfer_file_count'], transfer_bytes=value['transfer_bytes'])
            core.publish(control/'receive-result.json', result); return result
        except BaseException as error:
            core.publish(control/'receive-failure.json', dict(error_type=type(error).__name__, error=str(error), stage=str(stage), checked_unix=time.time()))
            raise


def accept(core_archive, stage):
    directory = canonical(core_archive); stage = canonical(stage)
    need(stage.parent == directory and re.fullmatch(r'\.post-callback-dependencies\.partial-[0-9a-f]{32}', stage.name), 'STAGE_SCOPE_DIFFERS')
    control = directory.parent/('.post-callback-dependencies-'+directory.name)
    with (control/'accept.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not any((control/name).exists() for name in ('accept-claim.json', 'accept-result.json', 'accept-failure.json')) and
             not (directory/CHILD).exists(), 'ACCEPT_ALREADY_ATTEMPTED_NO_REPLAY')
        core.publish(control/'accept-claim.json', dict(stage=str(stage), pid=os.getpid(), checked_unix=time.time()))
        try:
            value, _ = _json(stage/'source-snapshot.json'); binding, probe = _bound(directory, value)
            transfer, _ = _json(stage/'transfer-receipt.json')
            need(transfer.get('status') == TRANSFER_STATUS and transfer.get('binding') == binding and
                 transfer.get('source_snapshot_sha256') == core.sha(stage/'source-snapshot.json') and
                 transfer.get('fresh_transferred_destination_sha_verified') is True and
                 transfer.get('covered_core_sha_ack_verified') is True and transfer.get('actual_cuda_restore_accepted') is False,
                 'TRANSFER_RECEIPT_DIFFERS')
            expected = {'source-snapshot.json', 'transfer-receipt.json'} | {
                name for name, row in value['dependencies'].items() if row['storage'] == 'transfer'}
            actual = {str(path.relative_to(stage)) for path in stage.rglob('*') if path.is_file() or path.is_symlink()}
            need(actual == expected, 'STAGING_MEMBERS_CHANGED')
            mappings = _verify_files(directory, stage, value, probe)
            need(mappings == transfer.get('mappings'), 'TRANSFER_FILE_IDENTITIES_CHANGED')
            destination = directory/CHILD
            for row in mappings.values():
                path = Path(row['archive_path'])
                if path.is_relative_to(stage): row['archive_path'] = str(destination/path.relative_to(stage))
            receipt = dict(status=STATUS, binding=binding, source_snapshot_sha256=core.sha(stage/'source-snapshot.json'),
                destination=str(destination), reward_version=probe['version'], reward_round=probe['round'],
                file_count=value['file_count'], transfer_file_count=value['transfer_file_count'], transfer_bytes=value['transfer_bytes'],
                encoder_file_count=len(value['encoder_files']), mappings=mappings, all_manifest_dependencies_covered=True,
                encoder_weights_copied=False, fresh_transferred_destination_sha_verified=True,
                fresh_transferred_dependency_sha_checked=True,
                covered_core_sha_ack_verified=True, core_covered_payload_rehashed=False, destination_fsynced=True,
                core_payload_unmodified=True, eight_file_boundary_unmodified=True, actual_cuda_restore_accepted=False,
                actual_model_optimizer_rng_dataequal_accepted=False, checked_unix=time.time())
            core.publish(stage/'dependency-archive-receipt.json', receipt); _tree_fsync(stage)
            os.rename(stage, destination)
            fd = os.open(directory, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
            core.publish(control/'accept-result.json', dict(status=STATUS, destination=str(destination), checked_unix=time.time()))
            return receipt
        except BaseException as error:
            core.publish(control/'accept-failure.json', dict(error_type=type(error).__name__, error=str(error), checked_unix=time.time()))
            raise


def inspect(core_archive, verify_files=True, deep=False):
    directory = canonical(core_archive); destination = canonical(directory/CHILD)
    value, _ = _json(destination/'source-snapshot.json'); binding, probe = _bound(directory, value)
    receipt, _ = _json(destination/'dependency-archive-receipt.json')
    need(receipt.get('status') == STATUS and receipt.get('binding') == binding and
         receipt.get('source_snapshot_sha256') == core.sha(destination/'source-snapshot.json') and
         receipt.get('reward_version') == probe['version'] and receipt.get('reward_round') == probe['round'] and
         receipt.get('all_manifest_dependencies_covered') is True and receipt.get('actual_cuda_restore_accepted') is False,
         'ACCEPTED_DEPENDENCY_RECEIPT_REQUIRED')
    mappings = _verify_files(directory, destination, value, probe, rehash=verify_files, deep=deep)
    need(mappings == receipt.get('mappings'), 'ACCEPTED_MAPPING_DIFFERS')
    return dict(status=STATUS, destination=str(destination), binding=binding, reward_version=probe['version'],
                reward_round=probe['round'], file_count=value['file_count'], transfer_file_count=value['transfer_file_count'],
                transfer_bytes=value['transfer_bytes'], encoder_file_count=len(value['encoder_files']), mappings=mappings,
                all_manifest_dependencies_covered=True, fresh_transferred_dependency_sha_checked=bool(verify_files),
                covered_core_sha_ack_verified=True, core_covered_payload_rehashed=bool(verify_files and deep),
                actual_cuda_restore_accepted=False, actual_model_optimizer_rng_dataequal_accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='mode', required=True)
    item = sub.add_parser('snapshot'); item.add_argument('--base', required=True); item.add_argument('--step', type=int, choices=boundary.STEPS, required=True)
    item.add_argument('--core-archive', required=True); item.add_argument('--output')
    item = sub.add_parser('emit'); item.add_argument('--snapshot', required=True)
    item = sub.add_parser('receive'); item.add_argument('--core-archive', required=True)
    item = sub.add_parser('accept'); item.add_argument('--core-archive', required=True); item.add_argument('--stage', required=True)
    item = sub.add_parser('inspect'); item.add_argument('--core-archive', required=True); item.add_argument('--no-verify-files', action='store_true')
    item.add_argument('--deep', action='store_true', help='Explicitly reread covered core encoder tensor bodies too')
    args = parser.parse_args()
    if args.mode == 'emit': emit(args.snapshot, sys.stdout.buffer); return
    if args.mode == 'snapshot': result = snapshot(args.base, args.step, args.core_archive, args.output)
    elif args.mode == 'receive': result = receive(args.core_archive, sys.stdin.buffer)
    elif args.mode == 'accept': result = accept(args.core_archive, args.stage)
    else: result = inspect(args.core_archive, verify_files=not args.no_verify_files, deep=args.deep)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__': main()

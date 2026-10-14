"""CPU-only copy/verify, with a separate explicit low-space native relocation.

Producer manifests stay byte-exact. The archive's complete receipt supplies the
original-path -> accepted archive-path mapping; absence alone is never success.
No model import, GPU call, training lock, HTTP request or signal is used.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time

from release.recursive_bran_a.ablation150_storage_v1 import inspect as meta
from release.recursive_bran_a.ablation150_closed_storage_v2 import execute as closed

ROOT = meta.ROOT
MOUNT = Path('/mnt/shared/workspace/training_data_node_10')
ARCHIVE_ROOT = MOUNT / 'shaoshitong/recursive-ablation150-archive-20261014'
MOUNT_SOURCE = 'h100-10:/mnt/data'
PROTOCOL = 'ablation150_closed_same_cluster_archive_v1'
READY_PROTOCOL = 'ablation150_existing_same_cluster_storage_ready_v1'
CHUNK = 8 * 1024**2


def require(value, reason):
    meta.require(value, reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(CHUNK), b''):
            digest.update(data)
    return digest.hexdigest()


def metadata(path, root):
    return closed.fingerprint(path, root)


def stable_sha(row, root):
    closed.same_file(row, metadata(row['path'], root))
    result = sha(row['path'])
    closed.same_file(row, metadata(row['path'], root))
    return result


def inside(path, root):
    path, root = Path(path), Path(root)
    require(path.is_absolute() and path.is_relative_to(root) and '..' not in path.parts,
            'path outside exact archive/source namespace')
    require(all(not p.is_symlink() for p in (path, *path.parents)), 'symlink path/ancestor')
    return path


def directory(path, root):
    path = inside(path, root)
    path.mkdir(parents=True, exist_ok=True)
    meta.safe_path(path, root, 'directory')
    return path


def storage_ready(path, expected_sha, root=ROOT, archive_root=ARCHIVE_ROOT):
    path = Path(path)
    require(meta.SHA.fullmatch(expected_sha or ''), 'storage receipt SHA required')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'storage receipt changed')
    value = json.loads(raw)
    require(value.get('protocol') == READY_PROTOCOL and value.get('ready') is True and
            value.get('study_root') == str(root) and value.get('archive_root') == str(archive_root),
            'storage not approved for exact own study/archive namespace')
    observed = meta.mounted_capacity(archive_root)
    recorded = value.get('mount', {})
    require(recorded.get('mountpoint') == str(MOUNT) and recorded.get('source') == MOUNT_SOURCE and
            recorded.get('fstype') in ('nfs', 'nfs4') and
            all(observed.get(k) == recorded[k] for k in ('mountpoint', 'source', 'fstype')),
            'actual existing same-cluster mount identity differs')
    probe = value.get('probe', {})
    require(probe.get('bytes') == 64 and probe.get('write_rc') == probe.get('server_read_rc') == 0 and
            meta.SHA.fullmatch(probe.get('sha256', '')) and
            probe.get('local_sha256') == probe.get('server_sha256') == probe['sha256'],
            'missing actual writable/two-node same-object SHA evidence')
    probe_path = inside(probe.get('path', ''), archive_root)
    actual = metadata(probe_path, archive_root)
    require(actual['bytes'] == 64 and stable_sha(actual, archive_root) == probe['sha256'],
            'shared probe object changed/missing')
    return dict(path=str(path), sha256=expected_sha, mount=observed,
                retention_until=value.get('retention_until'),
                retention_date_known=value.get('retention_until') is not None,
                accepted_probe=probe)


def closure(root, group, source_commit):
    require(group in meta.GROUPS and meta.COMMIT.fullmatch(source_commit),
            'only named seven new groups/frozen scientific source permitted')
    return closed.closed_snapshot(root, group, source_commit)


def small_closure(plan, root):
    """Recheck exact closed owner/exit/wait/release without assuming pruned files exist."""
    receipt, _ = meta.binding(root, plan['group'], plan['source_commit'])
    require(all(receipt.get(k) == plan[k] for k in ('run_id', 'config_sha256', 'group', 'source_commit')),
            'closed group receipt identity changed')
    study, _ = meta.json_file(root / 'study-progress.json', root)
    meta.completed_group(root / plan['group'], receipt, study, root, plan['source_commit'])
    require(study.get('complete', {}).get(plan['group']) == plan['accepted_study_row'],
            'accepted study closure changed')
    closed.check_refs(plan['closure_refs'], root)
    closed.no_writers(root / plan['group'])


def inventory(snapshot, root, archive_root, implementation_commit):
    require(meta.COMMIT.fullmatch(implementation_commit), 'published archive implementation commit required')
    receipt = snapshot['receipt']
    require(re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', receipt['run_id']), 'unsafe archive run ID')
    target = archive_root / snapshot['group'] / receipt['run_id']
    values = {}
    for step, state in snapshot['posts'].items():
        expected = root / snapshot['group'] / f'checkpoints/training_state/global_step_{step}/post-evaluation'
        require(Path(state['path']) == expected / 'manifest.json', 'unexpected canonical post manifest')
        for row in state['members']:
            path = Path(row['path'])
            require(path.parent == expected and path.name in closed.NAMES,
                    'native member outside exact canonical nine-file allowlist')
            values[row['path']] = dict(metadata(path, root), sha256=row['recorded_sha256'],
                                      kind='CANONICAL_POST_NATIVE', step=step)
        for row in state['hf']:
            values[row['path']] = dict(metadata(row['path'], root), sha256=row['recorded_sha256'],
                                      kind='HF_EXPORT', step=step)
    for row in snapshot['closure_refs']:
        values[row['path']] = dict(metadata(row['path'], root), sha256=row['sha256'], kind='SMALL_PROOF')
    members = []
    for path in sorted(values):
        row = values[path]
        require(meta.SHA.fullmatch(row['sha256']), 'invalid recorded canonical/member SHA')
        destination = target / 'members' / Path(path).relative_to(root)
        members.append(dict(row, archive_path=str(destination)))
    require(sum(row['kind'] == 'CANONICAL_POST_NATIVE' for row in members) == 36,
            'must archive all four complete eight-rank/driver states')
    return dict(protocol=PROTOCOL, study_root=str(root), archive_root=str(archive_root),
        archive_directory=str(target), group=receipt['group'], run_id=receipt['run_id'],
        config_sha256=receipt['config_sha256'], source_commit=receipt['source_commit'],
        implementation_commit=implementation_commit, members=members,
        closure_refs=snapshot['closure_refs'], accepted_study_row=snapshot['accepted_study_row'],
        checkpoint_steps=list(meta.KEYS), total_member_bytes=sum(row['bytes'] for row in members),
        source_provenance=dict(scientific_receipt=receipt,
            archive_helper_sha256=sha(__file__)),
        retained_original_manifests=True, tensor_semantics_verified=False)


def sync_parent(path):
    descriptor = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_member(row, root, archive_root, progress):
    """Resume a same-file partial after checking its entire prefix against source."""
    source = Path(row['path'])
    dest = inside(row['archive_path'], archive_root)
    directory(dest.parent, archive_root)
    require(stable_sha(row, root) == row['sha256'], 'source SHA mismatch before transfer')
    if dest.exists():
        actual = metadata(dest, archive_root)
        require(actual['bytes'] == row['bytes'] and stable_sha(actual, archive_root) == row['sha256'],
                'existing final destination differs; no overwrite or reclaim')
        return dict(source_sha256_before=row['sha256'], source_sha256_after=stable_sha(row, root),
                    destination_sha256=row['sha256'], reused_accepted_destination=True,
                    destination_metadata=actual)
    partial = inside(dest.with_name(dest.name + '.partial'), archive_root)
    offset = 0
    if partial.exists():
        previous = metadata(partial, archive_root)
        offset = previous['bytes']
        require(offset <= row['bytes'], 'partial exceeds immutable source size')
        with source.open('rb') as a, partial.open('rb') as b:
            remaining = offset
            while remaining:
                count = min(CHUNK, remaining)
                require(a.read(count) == b.read(count), 'partial prefix mismatch; no unsafe resume')
                remaining -= count
    mode = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
    descriptor = os.open(partial, mode, 0o600)
    last = time.monotonic()
    copied = offset
    with source.open('rb') as source_stream, os.fdopen(descriptor, 'r+b') as target_stream:
        source_stream.seek(offset)
        target_stream.seek(offset)
        while True:
            data = source_stream.read(CHUNK)
            if not data:
                break
            target_stream.write(data)
            copied += len(data)
            if time.monotonic() - last >= 10:
                target_stream.flush()
                progress(dict(path=row['path'], copied_bytes=copied, total_bytes=row['bytes'],
                              phase='COPYING', time=time.time()))
                last = time.monotonic()
        target_stream.flush()
        os.fsync(target_stream.fileno())
    require(copied == row['bytes'], 'source transfer size changed')
    source_after = stable_sha(row, root)
    require(source_after == row['sha256'], 'source SHA mismatch after transfer')
    target_metadata = metadata(partial, archive_root)
    require(target_metadata['bytes'] == row['bytes'] and
            stable_sha(target_metadata, archive_root) == row['sha256'], 'destination full SHA mismatch')
    require(not os.path.lexists(dest), 'destination appeared during copy; refuse replace')
    os.link(partial, dest, follow_symlinks=False)
    partial.unlink()
    sync_parent(dest)
    return dict(source_sha256_before=row['sha256'], source_sha256_after=source_after,
        destination_sha256=row['sha256'], reused_accepted_destination=False,
        destination_metadata=metadata(dest, archive_root))


def progress_record(directory_path, value):
    closed.durable_json(directory_path / 'transfer-progress.json', value)
    print('ABLATION_ARCHIVE_PROGRESS=' + json.dumps(value, sort_keys=True), flush=True)


def plan_equal(old, new):
    require(old == new, 'archive immutable plan/source metadata changed')


def copy_closed(root, group, source_commit, archive_root, readiness, implementation_commit):
    snapshot = closure(root, group, source_commit)
    plan = inventory(snapshot, root, archive_root, implementation_commit)
    target = directory(plan['archive_directory'], archive_root)
    plan_path = target / 'inventory.json'
    if plan_path.exists():
        old, plan_sha = meta.json_file(plan_path, archive_root)
        plan_equal(old, plan)
    else:
        closed.durable_json(plan_path, plan, exclusive=True)
        plan_sha = sha(plan_path)
    complete_path = target / 'complete-sha-verified.json'
    if complete_path.exists():
        return verify_complete(complete_path, sha(complete_path), root, archive_root, readiness)
    missing_bytes = sum(row['bytes'] for row in plan['members'] if not Path(row['archive_path']).exists())
    require(closed.free_bytes(archive_root) > missing_bytes + meta.HEADROOM,
            'archive mount lacks actual free bytes for remaining copy plus 16GiB')
    small_closure(plan, root)
    results = []
    for index, row in enumerate(plan['members']):
        proof = copy_member(row, root, archive_root,
            lambda value: progress_record(target, dict(value, group=group, index=index)))
        require(proof['source_sha256_after'] == row['sha256'], 'reused destination source changed')
        record = dict(index=index, source=row['path'], destination=row['archive_path'],
                      bytes=row['bytes'], sha256=row['sha256'], verification=proof)
        closed.durable_json(target / 'files' / f'{index:04d}-accepted.json', record)
        results.append(record)
        progress_record(target, dict(phase='MEMBER_ACCEPTED', group=group, index=index,
            copied_members=index+1, total_members=len(plan['members']), time=time.time()))
    small_closure(plan, root)
    for row in plan['members']:
        closed.same_file(row, metadata(row['path'], root))
        require(stable_sha(metadata(row['archive_path'], archive_root), archive_root) == row['sha256'],
                'archive member changed before complete receipt')
    complete = dict(protocol=PROTOCOL, status='COMPLETE_FULL_SHA_ACCEPTED_COPY_ONLY', time=time.time(),
        inventory=str(plan_path), inventory_sha256=plan_sha, group=group, run_id=plan['run_id'],
        source_commit=source_commit, implementation_commit=implementation_commit,
        config_sha256=plan['config_sha256'], total_members=len(plan['members']),
        total_member_bytes=plan['total_member_bytes'], members=results, storage_ready=readiness,
        source_before_after_verified=True, destination_full_sha_verified=True,
        original_payloads_reclaimed=False, tensor_semantics_verified=False)
    closed.durable_json(complete_path, complete, exclusive=True)
    link = root / group / 'receipts' / ('canonical-archive-copy-' + plan_sha[:16] + '.json')
    mapping = dict(protocol=PROTOCOL, status='ACCEPTED_COPY_ORIGINALS_RETAINED',
        complete_receipt=str(complete_path), complete_receipt_sha256=sha(complete_path),
        inventory_sha256=plan_sha, group=group, run_id=plan['run_id'],
        original_manifests_unmodified=True, original_payloads_reclaimed=False,
        members=[dict(original=row['path'], archive=row['archive_path'], sha256=row['sha256'],
                      kind=row['kind'], bytes=row['bytes']) for row in plan['members']])
    if link.exists():
        previous, _ = meta.json_file(link, root)
        require(previous == mapping, 'canonical archive link differs')
    else:
        closed.durable_json(link, mapping, exclusive=True)
    return dict(status=complete['status'], complete_receipt=str(complete_path),
                complete_receipt_sha256=sha(complete_path), total_members=len(plan['members']),
                total_member_bytes=plan['total_member_bytes'], original_payloads_reclaimed=False)


def verify_complete(path, expected_sha, root, archive_root, readiness, full_sha=True):
    path = inside(path, archive_root)
    value, actual_sha = meta.json_file(path, archive_root)
    require(actual_sha == expected_sha and value.get('protocol') == PROTOCOL and
            value.get('status') == 'COMPLETE_FULL_SHA_ACCEPTED_COPY_ONLY' and
            value.get('source_before_after_verified') is True and
            value.get('destination_full_sha_verified') is True, 'archive complete receipt not accepted')
    plan, plan_sha = meta.json_file(value['inventory'], archive_root)
    require(plan_sha == value['inventory_sha256'] and plan.get('protocol') == PROTOCOL and
            plan.get('study_root') == str(root) and plan.get('archive_root') == str(archive_root) and
            Path(plan['archive_directory']) == path.parent and plan['group'] in meta.GROUPS,
            'archive inventory not exact same namespace/receipt')
    small_closure(plan, root)
    require(value['total_members'] == len(plan['members']) == len(value['members']) and
            value['total_member_bytes'] == sum(row['bytes'] for row in plan['members']),
            'archive complete coverage differs')
    for index, row in enumerate(plan['members']):
        expected = dict(index=index, source=row['path'], destination=row['archive_path'],
                        bytes=row['bytes'], sha256=row['sha256'])
        require(all(value['members'][index].get(k) == v for k, v in expected.items()),
                'archive source/destination/member map mismatch')
        destination = inside(row['archive_path'], archive_root)
        require(destination == path.parent / 'members' / Path(row['path']).relative_to(root),
                'archive member outside immutable mapped path')
        actual = metadata(destination, archive_root)
        require(actual['bytes'] == row['bytes'], 'archive destination size mismatch')
        if full_sha:
            require(stable_sha(actual, archive_root) == row['sha256'],
                    'archive destination fresh SHA mismatch; originals retained')
        else:
            closed.same_file(value['members'][index]['verification']['destination_metadata'], actual)
    return dict(status='ARCHIVE_FRESH_FULL_SHA_VERIFIED', complete_receipt=str(path),
        complete_receipt_sha256=actual_sha, total_members=len(plan['members']),
        total_member_bytes=plan['total_member_bytes'], source_sha_mode='HISTORICAL_BEFORE_AFTER_ACCEPTANCE',
        current_destination_full_sha_verified=full_sha, storage_ready=readiness,
        destination_sha_mode=('FULL_STREAM_FRESH_THIS_CALL' if full_sha else
            'PRIOR_ACCEPTED_FULL_SHA_PLUS_FRESH_UNCHANGED_FILE_METADATA'))


def prior_pruning_finished(plan, root):
    """Do not remove v2's canonical dependency before its own candidates are fully gone."""
    base = root / plan['group']
    candidates = []
    for step in meta.KEYS:
        candidates.extend(base.glob(f'checkpoints/training_state/global_step_{step}/rank-*.pt'))
        path = base / f'checkpoints/training_state/global_step_{step}/driver.pt'
        if path.exists():
            candidates.append(path)
    candidates.extend((base / 'checkpoints/native-rolling').glob('global_step_*/*.pt'))
    require(not candidates, 'superseded pre/rolling native payloads must be fully pruned before canonical reclaim')
    directory_path = base / 'receipts/closed-storage-v2'
    proof, _ = meta.json_file(directory_path / 'actual.json', root)
    pruning_plan, plan_sha = meta.json_file(directory_path / 'planned.json', root)
    require(proof.get('protocol') == closed.PROTOCOL and
            proof.get('status') == 'COMPLETE_SUPERSEDED_PAYLOADS_PRUNED' and
            proof.get('group') == plan['group'] and
            proof.get('plan_sha256') == plan_sha and
            proof.get('current_absent_files') == proof.get('total_planned_files'),
            'prior completed v2 pruning receipt required; unexplained absence is insufficient')
    require(pruning_plan.get('protocol') == closed.PROTOCOL and pruning_plan.get('root') == str(root) and
            all(pruning_plan.get(k) == plan[k] for k in ('group', 'run_id', 'source_commit', 'config_sha256')) and
            pruning_plan.get('closure_refs') == plan['closure_refs'] and
            pruning_plan.get('accepted_study_row') == plan['accepted_study_row'],
            'v2 complete pruning belongs to another exact closed run')
    all_files = [(state, row) for state in pruning_plan['candidate_states'] for row in state['files']]
    require(len(all_files) == proof['total_planned_files'], 'v2 pruning coverage differs')
    for index, (state, row) in enumerate(all_files):
        closed.check_allowlist(row['path'], state, base)
        binding = closed.file_binding(plan_sha, index, state, row)
        intent, _ = meta.json_file(directory_path / 'files' / f'{index:04d}-intent.json', root)
        actual, _ = meta.json_file(directory_path / 'files' / f'{index:04d}-actual.json', root)
        require(intent == binding and actual.get('binding_sha256') == closed.digest(binding) and
                actual.get('action') in ('UNLINKED_THIS_EXECUTION', 'OBSERVED_ABSENT_AFTER_DURABLE_INTENT') and
                not os.path.lexists(row['path']), 'v2 missing file not exact durable actual pruning')


def observed_burst(root, source_commit, archive_root):
    """Use actual local sizes, or accepted archived sizes when post payloads moved.

    An archived dimension is a previous real checkpoint measurement with fresh
    unchanged destination metadata, never a fabricated future allocation promise.
    """
    try:
        return closed.measured_burst(root, source_commit)
    except ValueError as error:
        if 'no actual complete native/HF sizes' not in str(error):
            raise
    natives, hfs = [], []
    for group in meta.GROUPS:
        for complete_path in (archive_root / group).glob('*/complete-sha-verified.json'):
            value, checksum = meta.json_file(complete_path, archive_root)
            if value.get('source_commit') != source_commit:
                continue
            verify_complete(complete_path, checksum, root, archive_root, {}, full_sha=False)
            plan, _ = meta.json_file(value['inventory'], archive_root)
            for step in meta.KEYS:
                natives.append(sum(row['bytes'] for row in plan['members']
                    if row['kind'] == 'CANONICAL_POST_NATIVE' and row['step'] == step))
                hfs.append(sum(row['bytes'] for row in plan['members']
                    if row['kind'] == 'HF_EXPORT' and row['step'] == step))
    require(natives and hfs and min(natives) > 0 and min(hfs) > 0,
            'neither local complete states nor accepted archive dimensions available')
    return dict(native_bytes=max(natives), hf_bytes=max(hfs), headroom_bytes=meta.HEADROOM,
        next_key_burst_bytes=2*max(natives)+max(hfs)+meta.HEADROOM,
        size_evidence='ACTUAL_ARCHIVED_CHECKPOINT_SIZES_WITH_FRESH_UNCHANGED_DESTINATION_METADATA',
        future_size_guarantee=False)


def reclaim(path, expected_sha, root, archive_root, readiness, next_key_burst):
    """Explicit opt-in only. Keep HF/manifests/logs; unlink at most 36 post .pt files."""
    require(type(next_key_burst) is int and next_key_burst > meta.HEADROOM,
            'explicit previously observed next-key burst bytes required')
    measured = observed_burst(root, meta.json_file(path, archive_root)[0]['source_commit'], archive_root)
    require(next_key_burst == measured['next_key_burst_bytes'],
            'reclaim threshold differs from freshly observed native/HF next-key burst')
    value, _ = meta.json_file(path, archive_root)
    plan, _ = meta.json_file(value['inventory'], archive_root)
    prior_pruning_finished(plan, root)
    if closed.free_bytes(root) >= next_key_burst:
        return dict(status='LOCAL_SPACE_SUFFICIENT_NO_CANONICAL_RECLAIM',
                    deleted_files_this_call=0, group=plan['group'])
    verified = verify_complete(path, expected_sha, root, archive_root, readiness)
    value, _ = meta.json_file(path, archive_root)
    plan, _ = meta.json_file(value['inventory'], archive_root)
    prior_pruning_finished(plan, root)
    output = directory(root / plan['group'] / 'receipts/canonical-archive-reclaim-v1', root)
    native = [row for row in plan['members'] if row['kind'] == 'CANONICAL_POST_NATIVE']
    require(len(native) == 36, 'reclaim allowlist not complete canonical36')
    deleted, old_absent = 0, 0
    for index, row in enumerate(native):
        original = Path(row['path'])
        expected = root / plan['group'] / f"checkpoints/training_state/global_step_{row['step']}/post-evaluation"
        require(original.parent == expected and original.name in closed.NAMES,
                'reclaim target outside canonical nine-file allowlist')
        binding = dict(protocol=PROTOCOL, inventory_sha256=value['inventory_sha256'],
            complete_receipt=str(path), complete_receipt_sha256=expected_sha,
            original=row, archive_path=row['archive_path'])
        intent = output / f'{index:04d}-intent.json'
        actual = output / f'{index:04d}-actual.json'
        if intent.exists():
            saved, _ = meta.json_file(intent, root)
            require(saved == binding, 'reclaim intent not exact accepted archive mapping')
        if not os.path.lexists(original):
            require(intent.exists(), 'missing canonical file has no own durable reclaim intent')
            if not actual.exists():
                closed.durable_json(actual, dict(time=time.time(), binding_sha256=closed.digest(binding),
                    action='OBSERVED_ABSENT_AFTER_DURABLE_INTENT'), exclusive=True)
            else:
                saved, _ = meta.json_file(actual, root)
                require(saved.get('binding_sha256') == closed.digest(binding), 'actual reclaim receipt differs')
            old_absent += 1
            continue
        require(not actual.exists(), 'reclaimed original unexpectedly reappeared; use independent restore receipt')
        closed.same_file(row, metadata(original, root))
        if closed.free_bytes(root) >= next_key_burst:
            break
        small_closure(plan, root)
        # Reverify the replacement immediately before deletion. Original hash/metadata
        # must still match the copy's accepted before/after source bytes.
        archive_member = metadata(row['archive_path'], archive_root)
        require(stable_sha(archive_member, archive_root) == row['sha256'] and
                stable_sha(row, root) == row['sha256'], 'replacement/original SHA changed before reclaim')
        if not intent.exists():
            closed.durable_json(intent, binding, exclusive=True)
        closed.unlink_bound(row, root)
        closed.durable_json(actual, dict(time=time.time(), binding_sha256=closed.digest(binding),
            action='UNLINKED_THIS_EXECUTION'), exclusive=True)
        deleted += 1
    result = dict(protocol=PROTOCOL, time=time.time(), complete_receipt=str(path),
        complete_receipt_sha256=expected_sha, group=plan['group'], run_id=plan['run_id'],
        status='ALL_CANONICAL_NATIVE_PAYLOADS_RELOCATED' if all(not os.path.lexists(row['path'])
            for row in native) else 'PARTIAL_OR_NOT_NEEDED_LOW_SPACE_RELOCATION',
        deleted_files_this_call=deleted, previous_or_recovery_absent=old_absent,
        local_free_bytes=closed.free_bytes(root), next_key_burst_bytes=next_key_burst,
        preserved='HF exports, original manifests/configs/logs/failed evidence remain local',
        original_manifests_unmodified=True, archived_map_is_required_for_checkpoint_validation=True,
        destination_full_sha_this_call=verified['current_destination_full_sha_verified'],
        original_payload_status='RELOCATED_OR_PARTIAL; NOT_FRESH_LOCAL_COMPLETE',
        restore_instructions='Verify complete archive SHA and member digests; copy mapped archive .pt files '
            'through fsynced .partial to their exact original paths while this group remains closed. '
            'Keep producer manifests unchanged; validate full restored file SHA before any native resume.')
    closed.durable_json(output / 'actual.json', result)
    return result


@contextmanager
def operation_lock(directory_path, archive_root, name='.archive-operation.lock'):
    directory(directory_path, archive_root)
    descriptor = os.open(inside(Path(directory_path) / name, archive_root),
                         os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def maintenance_once(root, source_commit, archive_root, readiness, implementation_commit,
                     prune_superseded=False, reclaim_accepted=False):
    """Background-only composition; errors are receipts and never training actions."""
    result = dict(protocol=PROTOCOL, time=time.time(), phase='MAINTENANCE', groups=[],
                  pruning_enabled=prune_superseded, canonical_reclaim_enabled=reclaim_accepted)
    # Existing v2 retains all protected post states. It refuses groups whose own
    # canonical files have already been explicitly relocated, rather than guessing.
    result['superseded_pruning'] = closed.once(root, source_commit, apply=prune_superseded)
    study, _ = meta.json_file(root / 'study-progress.json', root)
    require(study.get('source_commit') == source_commit, 'study source mismatch')
    for group in meta.GROUPS:
        if group not in study.get('complete', {}) or study.get('current') == group:
            continue
        try:
            receipt, _ = meta.binding(root, group, source_commit)
            target = archive_root / group / receipt['run_id']
            complete = target / 'complete-sha-verified.json'
            with operation_lock(archive_root / group, archive_root):
                if complete.exists():
                    # Ordinary polling does not rehash hundreds of GB every300s.
                    # Full fresh archive SHA is mandatory before any actual reclaim.
                    status = verify_complete(complete, sha(complete), root, archive_root, readiness, full_sha=False)
                else:
                    status = copy_closed(root, group, source_commit, archive_root,
                                         readiness, implementation_commit)
                record = dict(group=group, archive=status)
                if reclaim_accepted:
                    burst = observed_burst(root, source_commit, archive_root)['next_key_burst_bytes']
                    if closed.free_bytes(root) < burst:
                        record['canonical_reclaim'] = reclaim(complete, sha(complete), root,
                            archive_root, readiness, burst)
                result['groups'].append(record)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            result['groups'].append(dict(group=group, status='ATTENTION_OR_COPY_INCOMPLETE',
                                        error=str(error), training_action_executed=False))
    result.update(finished=time.time(), local_free_bytes=closed.free_bytes(root),
                  training_action_executed=False, gpu_action_executed=False)
    return result


def maintenance_loop(args):
    require(args.interval >= 30, 'maintenance interval must be at least30 seconds')
    with operation_lock(ARCHIVE_ROOT, ARCHIVE_ROOT, '.archive-maintenance.lock'):
        while True:
            try:
                ready = storage_ready(args.storage_ready, args.storage_ready_sha256)
                result = maintenance_once(ROOT, args.source_commit, ARCHIVE_ROOT, ready,
                    args.implementation_commit, args.prune_superseded_after_low_space,
                    args.reclaim_after_accepted)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                result = dict(protocol=PROTOCOL, time=time.time(), status='ATTENTION_NO_STORAGE_ACTION',
                              error=str(error), training_action_executed=False)
            closed.durable_json(ARCHIVE_ROOT / 'maintenance-current.json', result)
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
            if args.once:
                break
            time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('copy', 'verify', 'reclaim', 'maintain'))
    parser.add_argument('--group', choices=meta.GROUPS)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--implementation-commit', required=True)
    parser.add_argument('--storage-ready', required=True)
    parser.add_argument('--storage-ready-sha256', required=True)
    parser.add_argument('--complete-receipt')
    parser.add_argument('--complete-receipt-sha256')
    parser.add_argument('--next-key-burst-bytes', type=int)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--prune-superseded-after-low-space', action='store_true')
    parser.add_argument('--reclaim-after-accepted', action='store_true')
    args = parser.parse_args()
    require(meta.COMMIT.fullmatch(args.source_commit) and meta.COMMIT.fullmatch(args.implementation_commit),
            'published implementation and frozen science commits required')
    if args.action == 'maintain':
        maintenance_loop(args)
        return
    ready = storage_ready(args.storage_ready, args.storage_ready_sha256)
    if args.action == 'copy':
        require(args.group is not None, 'copy requires a named group')
        lock_base = directory(ARCHIVE_ROOT / args.group, ARCHIVE_ROOT)
    else:
        require(args.complete_receipt and meta.SHA.fullmatch(args.complete_receipt_sha256 or ''),
                'verify/reclaim require exact complete receipt and SHA')
        saved, _ = meta.json_file(args.complete_receipt, ARCHIVE_ROOT)
        require(saved['source_commit'] == args.source_commit and
                saved['implementation_commit'] == args.implementation_commit,
                'CLI science/implementation commit differs from accepted archive')
        lock_base = ARCHIVE_ROOT / saved['group']
    # This private archival lock never touches or acquires training locks.
    with operation_lock(lock_base, ARCHIVE_ROOT):
        if args.action == 'copy':
            value = copy_closed(ROOT, args.group, args.source_commit, ARCHIVE_ROOT,
                                ready, args.implementation_commit)
        elif args.action == 'verify':
            value = verify_complete(args.complete_receipt, args.complete_receipt_sha256,
                                    ROOT, ARCHIVE_ROOT, ready)
        else:
            value = reclaim(args.complete_receipt, args.complete_receipt_sha256,
                            ROOT, ARCHIVE_ROOT, ready, args.next_key_burst_bytes)
        print(json.dumps(value, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

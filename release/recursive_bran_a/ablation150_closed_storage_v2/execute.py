"""CPU-only, low-space-only pruning for the seven completed ablation150 groups.

The immutable source manifests are retained. Separate durable receipts explicitly
mark their pruned payloads as historical; missing files alone never prove pruning.
No tensor library, GPU/HTTP call, process signal or training lock is used.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

from release.recursive_bran_a.ablation150_storage_v1 import inspect as meta

PROTOCOL = 'ablation150_closed_storage_v2'
NAMES = tuple(f'rank-{rank:02d}.pt' for rank in range(8)) + ('driver.pt',)
FINGERPRINT = ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns', 'uid', 'mode', 'nlink')


def require(condition, reason):
    meta.require(condition, reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fingerprint(path, root):
    path, info = meta.safe_path(path, root)
    require(info.st_nlink == 1, 'payload has multiple hard links')
    return dict(path=str(path), device=info.st_dev, inode=info.st_ino, bytes=info.st_size,
        allocated_bytes=info.st_blocks * 512, mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns, uid=info.st_uid, mode=info.st_mode, nlink=info.st_nlink)


def same_file(expected, current):
    require(expected['path'] == current['path'] and
            all(expected[k] == current[k] for k in FINGERPRINT),
            'payload metadata changed: ' + expected['path'])


def durable_json(path, value, exclusive=False):
    """Same-filesystem replace plus file and parent fsync; no training directory mutation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(all(not p.is_symlink() for p in (path, *path.parents)), 'receipt ancestor symlink')
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    pending = path.with_name(path.name + f'.pending-{os.getpid()}-{time.time_ns()}')
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(pending, path, follow_symlinks=False)
            pending.unlink()
        else:
            os.replace(pending, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if pending.exists():
            pending.unlink()


def read_json(path, root):
    return meta.json_file(path, root)


def no_writers(base):
    require(not meta.open_writers(base), 'closed group has an open writable descriptor')


def closed_records(base, receipt, root, posts):
    """Prove all 150 native records and each actual fixed512 milestone, across retries."""
    records = {}
    refs = []
    for path in sorted((base / 'output').glob('attempt-*/completed-outers.jsonl')):
        path, info = meta.safe_path(path, root)
        require(info.st_size <= 16 * 1024**2, 'completed-round metadata unexpectedly large')
        raw = path.read_bytes()
        require(info.st_size == path.stat().st_size and info.st_mtime_ns == path.stat().st_mtime_ns,
                'completed-round records changed while read')
        refs.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
        for line in raw.splitlines():
            value = json.loads(line)
            require(all(value.get(k) == receipt[k] for k in
                    ('group', 'run_id', 'source_commit', 'config_sha256')), 'round identity mismatch')
            outer = value.get('completed_outer')
            require(type(outer) is int and 1 <= outer <= 150, 'invalid completed outer')
            projection = {k: value.get(k) for k in ('completed_outer', 'checkpoint', 'metrics',
                'optimizer_calls_per_rank')}
            require(outer not in records or records[outer] == projection,
                    'conflicting actual completed outer records')
            records[outer] = projection
    require(sorted(records) == list(range(1, 151)), 'missing complete native150 round chain')
    for step in meta.KEYS:
        row, post = records[step], posts[step]
        require(row['checkpoint'] == dict(path=post['path'], sha256=post['sha256'], native_only=False),
                'closed milestone not bound to canonical post state')
        metrics = row.get('metrics', {})
        require(metrics.get('val/qa_count/all') == 512 and
                metrics.get('val/qa_count/nq') == metrics.get('val/qa_count/hotpotqa') == 256,
                'missing actual complete balanced dev512')
        hits = []
        for source, count in (('all', 512), ('nq', 256), ('hotpotqa', 256)):
            value = metrics.get('val/em/' + source)
            require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1 and
                    math.isclose(value * count, round(value * count), abs_tol=1e-7),
                    'invalid complete512 EM')
            hits.append(round(value * count))
        require(hits[0] == hits[1] + hits[2], 'source EM counts disagree')
    return refs


def closed_snapshot(root, group, source_commit):
    """Protected states stay live; pre/rolling manifests may have v2-pruned payloads."""
    base = root / group
    receipt, receipt_sha = meta.binding(root, group, source_commit)
    study, _ = read_json(root / 'study-progress.json', root)
    require(study.get('source_commit') == source_commit, 'study source mismatch')
    complete = meta.completed_group(base, receipt, study, root, source_commit)
    no_writers(base)
    posts = {}
    for step in meta.KEYS:
        path = base / f'checkpoints/training_state/global_step_{step}/post-evaluation/manifest.json'
        posts[step] = meta.state_metadata(path, receipt, root, step, 'post')
    final = posts[150]
    require(complete.get('checkpoint') == dict(path=final['path'], sha256=final['sha256'], native_only=False),
            'complete150 lacks exact final canonical state')
    records = closed_records(base, receipt, root, posts)
    entry, entry_sha = read_json(base / 'receipts/current-entry.json', root)
    output = Path(entry['output'])
    refs = [dict(path=str(base / 'receipts/configuration.json'), sha256=receipt_sha),
            dict(path=receipt['config'], sha256=receipt['config_sha256']),
            dict(path=str(base / 'receipts/current-entry.json'), sha256=entry_sha)]
    for path in (output / 'training-complete.json', output / 'entry-exit.json',
                 base / 'receipts/actual-parent-wait.json'):
        _, sha = read_json(path, root)
        refs.append(dict(path=str(path), sha256=sha))
    refs.extend(records)
    refs.extend(dict(path=p['path'], sha256=p['sha256']) for p in posts.values())
    return dict(group=group, base=str(base), receipt=receipt, posts=posts, closure_refs=refs,
                accepted_study_row=study['complete'][group])


def payload_refs(path, receipt, root, step, phase, post):
    """Validate immutable candidate manifest, without treating absence as COMPLETE."""
    value, sha = read_json(path, root)
    identity = value.get('identity', {})
    require(value.get('status') == 'COMPLETE_SAME_WORLD_STATE_SAVED' and
            identity.get('code_commit') == receipt['source_commit'] and
            identity.get('config_sha256') == receipt['config_sha256'] and
            identity.get('data_identity') == receipt['data_identity'] and
            identity.get('actual_update_limit') == 150 and identity.get('completed_step') == step,
            'candidate identity/status mismatch')
    require(value.get('boundary_phase') is None, 'candidate is a protected post state')
    require(identity.get('model_identity') and identity.get('job_id'), 'candidate model/job identity missing')
    if phase == 'pre':
        require(identity == post['identity'] and value.get('hf_identity') == post['hf_identity'] and
                value.get('hf_checkpoint') == str(Path(receipt['base']) / f'checkpoints/actor/global_step_{step}'),
                'pre state lacks exact same-key post/HF replacement')
    else:
        require(value.get('snapshot_kind') == 'native_only_rolling_two' and
                value.get('hf_checkpoint') is None and value.get('hf_identity') is None,
                'unexpected rolling payload kind')
    workers = value.get('worker_states', [])
    require(len(workers) == 8 and sorted(w.get('identity', {}).get('rank', -1) for w in workers) == list(range(8)),
            'candidate lacks eight exact ranks')
    directory = Path(path).parent
    files = []
    for worker in sorted(workers, key=lambda w: w['identity']['rank']):
        ident = worker['identity']
        name = f"rank-{ident['rank']:02d}.pt"
        require(ident.get('world_size') == 8 and all(ident.get(k) == v for k, v in identity.items()) and
                worker.get('full_worker_state_finite') is True and worker.get('scheduler_last_epoch') == step,
                'candidate worker identity/finite/counter mismatch')
        require(worker.get('path') == str(directory / name) and meta.SHA.fullmatch(worker.get('sha256', '')) and
                meta.SHA.fullmatch(worker.get('logical_state_sha256', '')), 'candidate worker path/digest mismatch')
        files.append(dict(path=worker['path'], recorded_sha256=worker['sha256'], recorded_bytes=worker.get('bytes')))
    driver = value.get('driver', {})
    require(driver.get('path') == str(directory / 'driver.pt') and driver.get('next_step') == step+1 and
            meta.SHA.fullmatch(driver.get('sha256', '')) and meta.SHA.fullmatch(driver.get('logical_state_sha256', '')),
            'candidate driver path/cursor/digest mismatch')
    files.append(dict(path=driver['path'], recorded_sha256=driver['sha256'], recorded_bytes=driver.get('bytes')))
    return dict(path=str(path), sha256=sha, step=step, phase=phase, files=files,
                canonical_manifest=post['path'], canonical_manifest_sha256=post['sha256'])


def candidate_states(snapshot, root):
    base, receipt = Path(snapshot['base']), snapshot['receipt']
    result = []
    for step in meta.KEYS:
        path = base / f'checkpoints/training_state/global_step_{step}/manifest.json'
        if path.exists():
            result.append(payload_refs(path, receipt, root, step, 'pre', snapshot['posts'][step]))
    rolling = base / 'checkpoints/native-rolling'
    if rolling.exists():
        meta.safe_path(rolling, root, 'directory')
        for directory in sorted(rolling.iterdir()):
            match = re.fullmatch(r'global_step_(\d+)', directory.name)
            require(match is not None and directory.is_dir(), 'unexpected rolling directory')
            step = int(match[1])
            require(1 <= step < 150 and step not in meta.KEYS, 'unexpected rolling step')
            result.append(payload_refs(directory / 'manifest.json', receipt, root, step, 'rolling', snapshot['posts'][150]))
    return result


def measured_burst(root, source_commit):
    """Actual currently present, complete manifest sizes; no forecast cleanup gate."""
    native, hf, errors = [], [], []
    for group in meta.GROUPS:
        base = root / group
        if not base.exists():
            continue
        try:
            receipt, _ = meta.binding(root, group, source_commit)
            paths = [(step, base / f'checkpoints/training_state/global_step_{step}/post-evaluation/manifest.json', 'post')
                     for step in meta.KEYS]
            paths += [(int(p.parent.name.removeprefix('global_step_')), p, 'rolling') for p in
                      (base / 'checkpoints/native-rolling').glob('global_step_*/manifest.json')]
            for step, path, phase in paths:
                if not path.exists():
                    continue
                try:
                    state = meta.state_metadata(path, receipt, root, step, phase)
                    native.append(sum(f['bytes'] for f in state['members']))
                    if state['hf']:
                        hf.append(sum(f['bytes'] for f in state['hf']))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    errors.append(str(error))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
    require(native and hf, 'no actual complete native/HF sizes for next-key space gate')
    return dict(native_bytes=max(native), hf_bytes=max(hf), headroom_bytes=meta.HEADROOM,
                next_key_burst_bytes=2*max(native)+max(hf)+meta.HEADROOM, observation_errors=errors)


def free_bytes(root):
    fs = os.statvfs(root)
    return fs.f_bavail * fs.f_frsize


def check_refs(refs, root):
    for row in refs:
        path, _ = meta.safe_path(row['path'], root)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'],
                'immutable closure/config/manifest changed: ' + str(path))


def guarded_current(snapshot, root, source_commit):
    now = closed_snapshot(root, snapshot['group'], source_commit)
    require(now['closure_refs'] == snapshot['closure_refs'] and
            now['accepted_study_row'] == snapshot['accepted_study_row'], 'completed-group closure changed')
    check_refs(snapshot['closure_refs'], root)
    return now


def canonical_members(snapshot, root):
    files = {}
    for post in snapshot['posts'].values():
        for row in post['members'] + post['hf']:
            current = fingerprint(row['path'], root)
            files[current['path']] = dict(current, recorded_sha256=row['recorded_sha256'])
    return [files[name] for name in sorted(files)]


def check_allowlist(path, state, base):
    path = Path(path)
    require(path.name in NAMES and path.parent == Path(state['path']).parent,
            'outside exact rank/driver file allowlist')
    expected = (base / f"checkpoints/training_state/global_step_{state['step']}" if state['phase'] == 'pre' else
                base / f"checkpoints/native-rolling/global_step_{state['step']}")
    require(path.parent == expected and '/post-evaluation/' not in str(path), 'outside exact superseded directory')


def plan_for(snapshot, root):
    states = candidate_states(snapshot, root)
    for state in states:
        for row in state['files']:
            check_allowlist(row['path'], state, Path(snapshot['base']))
            actual = fingerprint(row['path'], root)
            require(actual['bytes'] > 0 and (row['recorded_bytes'] is None or
                    actual['bytes'] == row['recorded_bytes']), 'candidate payload size differs')
            row.update(actual)
    receipt = snapshot['receipt']
    return dict(protocol=PROTOCOL, created=time.time(), source_commit=receipt['source_commit'],
        group=receipt['group'], run_id=receipt['run_id'], config_sha256=receipt['config_sha256'],
        root=str(root), closure_refs=snapshot['closure_refs'], accepted_study_row=snapshot['accepted_study_row'],
        canonical_members=canonical_members(snapshot, root), candidate_states=states,
        limitation='Historical manifest bytes are retained; pruned payloads are not a fresh COMPLETE checkpoint')


def check_plan(plan, snapshot, root):
    receipt = snapshot['receipt']
    require(plan.get('protocol') == PROTOCOL and plan.get('root') == str(root) and
            all(plan.get(k) == receipt[k] for k in ('group', 'run_id', 'source_commit', 'config_sha256')),
            'pruning plan not exact bound source/run/config/root')
    require(plan['closure_refs'] == snapshot['closure_refs'] and
            plan['accepted_study_row'] == snapshot['accepted_study_row'], 'pruning plan closure changed')
    require(plan['canonical_members'] == canonical_members(snapshot, root), 'canonical protected metadata changed')
    states = candidate_states(snapshot, root)
    require(len(states) == len(plan['candidate_states']), 'candidate manifest inventory changed')
    for current, saved in zip(states, plan['candidate_states']):
        require(all(current[k] == saved[k] for k in ('path', 'sha256', 'step', 'phase',
                'canonical_manifest', 'canonical_manifest_sha256')), 'candidate manifest changed')
        require(len(saved['files']) == 9, 'candidate file allowlist incomplete')
        for reference, expected in zip(current['files'], saved['files']):
            check_allowlist(expected['path'], saved, Path(snapshot['base']))
            require(all(reference[k] == expected[k] for k in ('path', 'recorded_sha256', 'recorded_bytes')),
                    'candidate file reference changed')


def stream_sha(row, root):
    path = Path(row['path'])
    same_file(row, fingerprint(path, root))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    h = hashlib.sha256()
    try:
        with os.fdopen(fd, 'rb') as stream:
            while True:
                block = stream.read(8*1024**2)
                if not block:
                    break
                h.update(block)
    finally:
        same_file(row, fingerprint(path, root))
    require(h.hexdigest() == row['recorded_sha256'], 'canonical payload SHA mismatch: ' + str(path))
    return dict(path=str(path), sha256=h.hexdigest(), bytes=row['bytes'])


def verify_canonical(plan, plan_sha, snapshot, root, directory):
    path = directory / 'canonical-verified.json'
    if path.exists():
        proof, _ = read_json(path, root)
        require(proof.get('protocol') == PROTOCOL and proof.get('plan_sha256') == plan_sha and
                proof.get('members') == [dict(path=f['path'], sha256=f['recorded_sha256'], bytes=f['bytes'])
                                        for f in plan['canonical_members']], 'canonical verification receipt changed')
        return 'REUSED_PREVIOUS_FULL_SHA_WITH_FRESH_UNCHANGED_METADATA'
    guarded_current(snapshot, root, plan['source_commit'])
    members = [stream_sha(row, root) for row in plan['canonical_members']]
    guarded_current(snapshot, root, plan['source_commit'])
    require(plan['canonical_members'] == canonical_members(snapshot, root), 'canonical metadata changed after full SHA')
    durable_json(path, dict(protocol=PROTOCOL, time=time.time(), plan_sha256=plan_sha, members=members,
        full_stream_sha_verified=True, tensor_semantics_verified=False), exclusive=True)
    return 'FULL_STREAM_SHA_VERIFIED_THIS_CALL'


def unlink_bound(row, root):
    """Unlink one exact inode through non-symlink directory FDs; never rmtree."""
    path = Path(row['path'])
    same_file(row, fingerprint(path, root))
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parent.relative_to(root).parts:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        info = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        require(stat.S_ISREG(info.st_mode) and info.st_ino == row['inode'] and info.st_dev == row['device'] and
                info.st_size == row['bytes'] and info.st_mtime_ns == row['mtime_ns'] and
                info.st_ctime_ns == row['ctime_ns'] and info.st_nlink == 1, 'unlink target changed')
        os.unlink(path.name, dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_binding(plan_sha, index, state, row):
    return dict(protocol=PROTOCOL, plan_sha256=plan_sha, index=index, path=row['path'],
        candidate_manifest=state['path'], candidate_manifest_sha256=state['sha256'],
        canonical_manifest=state['canonical_manifest'],
        canonical_manifest_sha256=state['canonical_manifest_sha256'], expected=row)


def execute_group(snapshot, root, burst):
    base = Path(snapshot['base'])
    directory = base / 'receipts/closed-storage-v2'
    if directory.exists():
        meta.safe_path(directory, root, 'directory')
    plan_path = directory / 'planned.json'
    if plan_path.exists():
        plan, plan_sha = read_json(plan_path, root)
        check_plan(plan, snapshot, root)
    else:
        plan = plan_for(snapshot, root)
        if not plan['candidate_states']:
            return dict(group=plan['group'], status='NO_SUPERSEDED_PAYLOADS', deleted_files_this_call=0,
                        current_absent_files=0, total_planned_files=0, fresh_free_bytes=free_bytes(root),
                        canonical_sha_evidence='NOT_NEEDED_NO_PRUNING_CANDIDATES')
        durable_json(plan_path, plan, exclusive=True)
        _, plan_sha = read_json(plan_path, root)
    # Validate the entire retained/pruned inventory before hashing or deleting
    # any further payload. A manifest-only file missing without our prior durable
    # intent is an unexplained disappearance, not a completed cleanup.
    all_files = [(state, row) for state in plan['candidate_states'] for row in state['files']]
    for index, (state, row) in enumerate(all_files):
        intent = directory / 'files' / f'{index:04d}-intent.json'
        actual = directory / 'files' / f'{index:04d}-actual.json'
        binding = file_binding(plan_sha, index, state, row)
        require(not Path(row['path']).is_symlink(), 'candidate payload symlink')
        if intent.exists():
            old, _ = read_json(intent, root)
            require(old == binding, 'file intent not exact immutable plan')
        if actual.exists():
            old, _ = read_json(actual, root)
            require(intent.exists() and old.get('binding_sha256') == digest(binding) and
                    old.get('action') in ('UNLINKED_THIS_EXECUTION', 'OBSERVED_ABSENT_AFTER_DURABLE_INTENT'),
                    'actual receipt not exact own durable intent')
        if Path(row['path']).exists():
            same_file(row, fingerprint(row['path'], root))
            require(not actual.exists(), 'previously pruned payload unexpectedly reappeared')
        else:
            require(intent.exists(), 'missing payload has no own durable deletion intent')
    sha_mode = verify_canonical(plan, plan_sha, snapshot, root, directory)
    deleted = 0
    observed_absent = 0
    for index, (state, row) in enumerate(all_files):
        intent = directory / 'files' / f'{index:04d}-intent.json'
        actual = directory / 'files' / f'{index:04d}-actual.json'
        binding = file_binding(plan_sha, index, state, row)
        if intent.exists():
            old, _ = read_json(intent, root)
            require(old == binding, 'file intent not exact immutable plan')
        if actual.exists():
            old, _ = read_json(actual, root)
            require(old.get('binding_sha256') == digest(binding) and
                    old.get('action') in ('UNLINKED_THIS_EXECUTION', 'OBSERVED_ABSENT_AFTER_DURABLE_INTENT') and
                    not Path(row['path']).exists(), 'actual deletion receipt not consistent with absence')
            observed_absent += 1
            continue
        if not Path(row['path']).exists():
            require(intent.exists(), 'missing payload has no own durable deletion intent')
            guarded_current(snapshot, root, plan['source_commit'])
            durable_json(actual, dict(time=time.time(), binding_sha256=digest(binding),
                         action='OBSERVED_ABSENT_AFTER_DURABLE_INTENT', actual_unlink_observed_this_call=False), exclusive=True)
            observed_absent += 1
            continue
        same_file(row, fingerprint(row['path'], root))
        if free_bytes(root) >= burst:
            break
        guarded_current(snapshot, root, plan['source_commit'])
        require(plan['canonical_members'] == canonical_members(snapshot, root), 'canonical metadata changed before unlink')
        if not intent.exists():
            durable_json(intent, binding, exclusive=True)
        unlink_bound(row, root)
        durable_json(actual, dict(time=time.time(), binding_sha256=digest(binding), action='UNLINKED_THIS_EXECUTION',
                     actual_unlink_observed_this_call=True), exclusive=True)
        deleted += 1
    absent = sum(not os.path.lexists(row['path']) for _, row in all_files)
    completed = absent == len(all_files) and all((directory / 'files' / f'{i:04d}-actual.json').exists()
                                               for i in range(len(all_files)))
    value = dict(protocol=PROTOCOL, time=time.time(), plan_sha256=plan_sha, group=plan['group'],
        deleted_files_this_call=deleted, previously_or_recovery_observed_absent=observed_absent,
        total_planned_files=len(all_files), current_absent_files=absent,
        status='COMPLETE_SUPERSEDED_PAYLOADS_PRUNED' if completed else 'PARTIAL_LOW_SPACE_PRUNING',
        canonical_sha_evidence=sha_mode, fresh_free_bytes=free_bytes(root), tensor_semantics_verified=False,
        tombstones=[dict(manifest=s['path'], manifest_sha256=s['sha256'], historical_payload_status='PRUNED' if
                        all(not os.path.lexists(f['path']) for f in s['files']) else 'PARTIAL_OR_RETAINED',
                        canonical_manifest=s['canonical_manifest'], canonical_manifest_sha256=s['canonical_manifest_sha256'],
                        exact_payloads=[f['path'] for f in s['files']]) for s in plan['candidate_states']])
    durable_json(directory / 'actual.json', value)
    if completed and not (directory / 'complete.json').exists():
        guarded_current(snapshot, root, plan['source_commit'])
        durable_json(directory / 'complete.json', value, exclusive=True)
    return {k: value[k] for k in ('group', 'status', 'deleted_files_this_call', 'current_absent_files',
                                'total_planned_files', 'fresh_free_bytes', 'canonical_sha_evidence')}


def once(root, source_commit, apply=False):
    require(meta.COMMIT.fullmatch(source_commit), 'invalid frozen scientific commit')
    meta.safe_path(root, root, 'directory')
    result = dict(protocol=PROTOCOL, time=time.time(), source_commit=source_commit, root=str(root),
                  deletion_enabled=apply, groups=[])
    try:
        burst = measured_burst(root, source_commit)
        result.update(space=burst, fresh_free_bytes=free_bytes(root))
        if result['fresh_free_bytes'] >= burst['next_key_burst_bytes']:
            result['status'] = 'SPACE_SUFFICIENT_NO_PRUNING'
            return result
        result['status'] = 'LOW_SPACE'
        for group in meta.GROUPS:
            if not (root / group).exists():
                continue
            try:
                snapshot = closed_snapshot(root, group, source_commit)
                if not apply:
                    states = candidate_states(snapshot, root)
                    result['groups'].append(dict(group=group, status='CLOSED_GROUP_DRY_RUN',
                                                candidate_manifests=len(states), deletion_executed=False))
                else:
                    result['groups'].append(execute_group(snapshot, root, burst['next_key_burst_bytes']))
                if free_bytes(root) >= burst['next_key_burst_bytes']:
                    break
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                result['groups'].append(dict(group=group, status='ATTENTION_NO_FURTHER_PRUNING', error=str(error)))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        result.update(status='ATTENTION_NO_PRUNING', error=str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--watch', action='store_true')
    parser.add_argument('--root', choices=[str(meta.ROOT)], default=str(meta.ROOT))
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--apply-after-low-space', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    require(args.interval >= 30, 'watch interval must be at least30 seconds')
    while True:
        print(json.dumps(once(Path(args.root), args.source_commit, args.apply_after_low_space),
                         sort_keys=True, allow_nan=False), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()

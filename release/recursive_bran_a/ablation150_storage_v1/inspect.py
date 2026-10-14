"""Read small JSON/configs and stat payloads; never load or delete tensors.

This is intentionally a candidate tool, not a deletion command. Fresh filesystem
metadata can support a separately reviewed cleanup, but does not verify payload
SHA256 or the contents of model/Adam/RNG tensors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

ROOT = Path('/dataset/shaoshitong/recursive-ablation150-20261014')
GROUPS = ('C1-root-only', 'C2-random-final', 'budget-5', 'budget-20',
          'position-early', 'position-late', 'generation-2')
KEYS = (5, 50, 100, 150)
PROTOCOL = 'recursive_mechanism_ablation150_v1'
SHA = re.compile(r'^[0-9a-f]{64}$')
COMMIT = re.compile(r'^[0-9a-f]{40}$')
HEADROOM = 16 * 1024**3


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def mounted_capacity(path):
    """Measure only an already mounted path; never mount or transfer anything."""
    path = Path(path).resolve(strict=True)
    # statvfs first resolves an existing automount before its backing identity
    # is read. The autofs trigger is not the filesystem storing payloads.
    fs = os.statvfs(path)
    matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        fields, filesystem = left.split(), right.split()
        mount = Path(fields[4].replace('\\040', ' ').replace('\\134', '\\'))
        if path.is_relative_to(mount):
            matches.append((len(mount.parts), mount, filesystem, int(fields[0])))
    require(matches, 'existing mount identity unavailable')
    _, mount, filesystem, _ = max(matches,
        key=lambda row: (row[0], row[2][0] != 'autofs', row[3]))
    return dict(path=str(path), mountpoint=str(mount), fstype=filesystem[0],
                source=filesystem[1], free_bytes=fs.f_bavail * fs.f_frsize,
                total_bytes=fs.f_blocks * fs.f_frsize, transfer_executed=False)


def safe_path(path, root, kind='file'):
    path, root = Path(path), Path(root)
    require(path.is_absolute() and path.is_relative_to(root), 'outside exact owned root')
    require('..' not in path.parts, 'parent traversal')
    for ancestor in (path, *path.parents):
        require(not ancestor.is_symlink(), 'symlink: ' + str(ancestor))
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) if kind == 'file' else stat.S_ISDIR(info.st_mode),
            'wrong file kind: ' + str(path))
    return path, info


def json_file(path, root):
    path, info = safe_path(path, root)
    require(info.st_size <= 2 * 1024**2, 'metadata unexpectedly large')
    raw = path.read_bytes()
    after = path.stat()
    require((info.st_ino, info.st_size, info.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'metadata changed while read')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def file_metadata(path, root):
    path, info = safe_path(path, root)
    require(info.st_size > 0, 'empty payload: ' + str(path))
    return dict(path=str(path), bytes=info.st_size, allocated_bytes=info.st_blocks * 512,
                inode=info.st_ino, device=info.st_dev, mtime_ns=info.st_mtime_ns,
                uid=info.st_uid)


def owner_present(entry):
    """Any present current PID blocks candidates, including a reused PID."""
    require(type(entry.get('pid')) is int and entry['pid'] > 0, 'invalid current owner PID')
    require(type(entry.get('uid')) is int and type(entry.get('tick')) is int,
            'missing owner identity')
    require(Path('/proc/self/stat').is_file(), 'live scan requires Linux /proc')
    return Path('/proc', str(entry['pid'])).exists()


def open_writers(root):
    """Find writable file descriptors under this root; unreadable proc fails closed."""
    result = []
    require(Path('/proc/self/stat').is_file(), 'live scan requires Linux /proc')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fds = list((proc / 'fd').iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        for fd in fds:
            try:
                target = os.readlink(fd).removesuffix(' (deleted)')
                path = Path(target)
                if not path.is_absolute() or not path.is_relative_to(root):
                    continue
                flags = next(line.split()[1] for line in
                             (proc / 'fdinfo' / fd.name).read_text().splitlines()
                             if line.startswith('flags:'))
                if int(flags, 8) & os.O_ACCMODE:
                    result.append(dict(pid=int(proc.name), path=target))
            except (FileNotFoundError, ProcessLookupError):
                continue
    return result


def binding(root, group, source_commit):
    base = root / group
    safe_path(base, root, 'directory')
    receipt, receipt_sha = json_file(base / 'receipts/configuration.json', root)
    require(receipt.get('protocol') == PROTOCOL and receipt.get('group') == group,
            'wrong protocol/group')
    require(receipt.get('source_commit') == source_commit, 'wrong frozen scientific source')
    require(receipt.get('base') == str(base) and receipt.get('output') == str(base / 'output'),
            'wrong exact base/output')
    require(receipt.get('evaluation_steps') == list(KEYS) and
            receipt.get('actual_update_limit') == 150, 'wrong schedule')
    require(SHA.fullmatch(receipt.get('config_sha256', '')) is not None,
            'invalid config SHA')
    config, info = safe_path(receipt['config'], root)
    require(config == base / 'receipts/resolved.yaml' and info.st_size < 2 * 1024**2,
            'wrong bound config path')
    require(hashlib.sha256(config.read_bytes()).hexdigest() == receipt['config_sha256'],
            'bound config changed')
    require(bool(receipt.get('run_id')) and SHA.fullmatch(receipt.get('data_identity', '')),
            'missing run/data identity')
    return receipt, receipt_sha


def state_metadata(path, receipt, root, step, phase):
    value, manifest_sha = json_file(path, root)
    ident = value.get('identity', {})
    require(value.get('status') == 'COMPLETE_SAME_WORLD_STATE_SAVED', 'incomplete native state')
    require(ident.get('code_commit') == receipt['source_commit'] and
            ident.get('config_sha256') == receipt['config_sha256'] and
            ident.get('data_identity') == receipt['data_identity'] and
            ident.get('actual_update_limit') == 150 and ident.get('completed_step') == step,
            'native source/config/data/step mismatch')
    require(ident.get('model_identity') and ident.get('job_id'), 'missing model/job identity')
    require(value.get('boundary_phase') == ('post-evaluation' if phase == 'post' else None),
            'wrong checkpoint boundary phase')
    if phase == 'rolling':
        require(value.get('snapshot_kind') == 'native_only_rolling_two' and
                value.get('hf_checkpoint') is None and value.get('hf_identity') is None,
                'wrong rolling snapshot kind')
    members, ranks = [], []
    workers = value.get('worker_states', [])
    require(len(workers) == 8, 'missing eight workers')
    directory = Path(path).parent
    for worker in workers:
        worker_id = worker.get('identity', {})
        rank = worker_id.get('rank')
        require(type(rank) is int and rank in range(8), 'invalid worker rank')
        ranks.append(rank)
        require(worker_id.get('world_size') == 8 and
                all(worker_id.get(k) == v for k, v in ident.items()), 'worker identity mismatch')
        require(worker.get('full_worker_state_finite') is True and
                worker.get('scheduler_last_epoch') == step, 'incomplete worker counters/finite evidence')
        require(worker.get('path') == str(directory / f'rank-{rank:02d}.pt'),
                'unexpected worker payload path')
        require(SHA.fullmatch(worker.get('sha256', '')) and
                SHA.fullmatch(worker.get('logical_state_sha256', '')), 'missing worker digests')
        meta = file_metadata(worker['path'], root)
        require(type(worker.get('bytes')) is int and worker['bytes'] == meta['bytes'],
                'worker size changed/missing')
        members.append(dict(meta, recorded_sha256=worker['sha256']))
    require(set(ranks) == set(range(8)) and len(set(ranks)) == 8, 'duplicate/missing rank')
    driver = value.get('driver', {})
    require(driver.get('path') == str(directory / 'driver.pt') and driver.get('next_step') == step + 1,
            'wrong driver path/cursor')
    require(SHA.fullmatch(driver.get('sha256', '')) and
            SHA.fullmatch(driver.get('logical_state_sha256', '')), 'missing driver digests')
    members.append(dict(file_metadata(driver['path'], root), recorded_sha256=driver['sha256']))
    hf = []
    if phase != 'rolling':
        hf_path = Path(receipt['base']) / 'checkpoints/actor' / f'global_step_{step}'
        require(value.get('hf_checkpoint') == str(hf_path), 'wrong HF checkpoint path')
        safe_path(hf_path, root, 'directory')
        hashes = value.get('hf_identity', {}).get('files', {})
        require(isinstance(hashes, dict) and 'config.json' in hashes and
                'tokenizer_config.json' in hashes and any(
                    name.endswith('.safetensors') or
                    (name.startswith('pytorch_model') and name.endswith('.bin')) for name in hashes),
                'incomplete HF model/tokenizer metadata')
        require(hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() ==
                value.get('hf_identity', {}).get('sha256'), 'HF manifest identity mismatch')
        for name, digest in hashes.items():
            require(SHA.fullmatch(digest) and not Path(name).is_absolute() and '..' not in Path(name).parts,
                    'invalid HF member identity/path')
            hf.append(dict(file_metadata(hf_path / name, root), recorded_sha256=digest))
        # Metadata verifies every recorded shard exists, and detects unrecorded files/symlinks.
        actual_files = set()
        for member in hf_path.rglob('*'):
            safe_path(member, root, 'directory' if member.is_dir() else 'file')
            if member.is_file():
                actual_files.add(str(member.relative_to(hf_path)))
        require(actual_files == set(hashes), 'HF inventory differs from saved identity')
    return dict(path=str(path), sha256=manifest_sha, step=step, phase=phase,
                identity=ident, hf_identity=value.get('hf_identity'), members=members, hf=hf,
                payload_sha256_verified=False, tensor_contents_verified=False)


def completed_group(base, receipt, study, root, source_commit):
    entry, _ = json_file(base / 'receipts/current-entry.json', root)
    require(all(entry.get(k) == receipt[k] for k in ('group', 'run_id', 'config_sha256', 'source_commit')),
            'current entry not exact bound run')
    output = Path(entry['output'])
    require(output.parent == base / 'output' and output.name.startswith('attempt-'), 'wrong attempt output')
    safe_path(output, root, 'directory')
    require(not owner_present(entry), 'current entry owner PID still present')
    complete, complete_sha = json_file(output / 'training-complete.json', root)
    terminal, terminal_sha = json_file(output / 'entry-exit.json', root)
    wait, _ = json_file(base / 'receipts/actual-parent-wait.json', root)
    require(complete.get('status') == 'COMPLETE_NATIVE_150_TRAINING_AND_FOUR_SCHEDULED_DEV' and
            complete.get('completed_outer') == 150 and complete.get('evaluation_steps') == list(KEYS),
            'missing complete150/four scheduled dev')
    require(all(complete.get(k) == receipt[k] for k in ('group', 'run_id', 'config_sha256', 'source_commit')),
            'completion belongs to another run')
    require(all(terminal.get(k) == entry[k] for k in ('pid', 'uid', 'tick', 'group', 'run_id')) and
            terminal.get('rc') == 0 and terminal.get('error') is None, 'missing actual exact entry exit0')
    require(wait.get('pid') == entry['pid'] and wait.get('group') == receipt['group'] and
            wait.get('source_commit') == source_commit and wait.get('rc') == 0,
            'missing exact actual parent wait0')
    accepted = study.get('complete', {}).get(receipt['group'], {})
    require(study.get('current') != receipt['group'], 'study still names group active')
    require(accepted.get('training_complete') == str(output / 'training-complete.json') and
            accepted.get('complete_sha256') == complete_sha, 'study completion not exact accepted file')
    release = accepted.get('owned_gpu_release', {})
    require(release.get('complete') is True and release.get('query_rc') == 0 and
            release.get('applications') == [], 'owned release not accepted')
    if 'actual_parent_wait' in accepted:
        require(accepted['actual_parent_wait'] == str(base / 'receipts/actual-parent-wait.json') and
                accepted.get('actual_parent_wait_rc') == 0, 'study wait acceptance mismatch')
    else:
        require(accepted.get('entry_exit') == str(output / 'entry-exit.json') and
                accepted.get('entry_exit_sha256') == terminal_sha, 'study exit acceptance mismatch')
    return complete


def collect(root, source_commit):
    require(COMMIT.fullmatch(source_commit), 'require published frozen scientific commit')
    safe_path(root, root, 'directory')
    study, study_sha = json_file(root / 'study-progress.json', root)
    require(study.get('source_commit') == source_commit, 'study source mismatch')
    writers = open_writers(root)
    rows = []
    for group in GROUPS:
        base = root / group
        if not base.exists():
            rows.append(dict(group=group, status='NOT_PREPARED', states=[], candidates=[]))
            continue
        row = dict(group=group, status='BOUND_METADATA', states=[], candidates=[], refusals=[])
        try:
            receipt, receipt_sha = binding(root, group, source_commit)
            row.update(config_sha256=receipt['config_sha256'], run_id=receipt['run_id'],
                       receipt_sha256=receipt_sha)
            entry_path = base / 'receipts/current-entry.json'
            if entry_path.exists():
                entry, _ = json_file(entry_path, root)
                require(all(entry.get(k) == receipt[k] for k in
                            ('group', 'run_id', 'config_sha256', 'source_commit')),
                        'current entry not exact bound run')
                output = Path(entry['output'])
                require(output.parent == base / 'output' and output.name.startswith('attempt-'),
                        'wrong current attempt output')
                safe_path(output, root, 'directory')
                progress_path = output / 'progress.json'
                if progress_path.exists():
                    progress, progress_sha = json_file(progress_path, root)
                    require(all(progress.get(k) == receipt[k] for k in
                                ('group', 'run_id', 'config_sha256', 'source_commit')),
                            'progress not exact bound run')
                    row['progress_metadata'] = dict(completed_outer=progress['completed_outer'],
                        checkpoint=progress['checkpoint'], progress_sha256=progress_sha)
            state_by_key = {}
            for step in KEYS:
                directory = base / 'checkpoints/training_state' / f'global_step_{step}'
                pair = {}
                for phase, path in (('pre', directory / 'manifest.json'),
                                    ('post', directory / 'post-evaluation/manifest.json')):
                    if path.exists():
                        try:
                            pair[phase] = state_metadata(path, receipt, root, step, phase)
                            row['states'].append(pair[phase])
                        except (ValueError, OSError, KeyError, TypeError) as exc:
                            row['refusals'].append(dict(path=str(path), reason=str(exc)))
                state_by_key[step] = pair
            rolling = base / 'checkpoints/native-rolling'
            if rolling.exists():
                safe_path(rolling, root, 'directory')
                for directory in sorted(rolling.glob('global_step_*')):
                    match = re.fullmatch(r'global_step_(\d+)', directory.name)
                    require(match is not None, 'invalid rolling directory name')
                    step = int(match[1])
                    require(1 <= step < 150 and step not in KEYS, 'unexpected rolling key/step')
                    try:
                        row['states'].append(state_metadata(directory / 'manifest.json', receipt,
                                                            root, step, 'rolling'))
                    except (ValueError, OSError, KeyError, TypeError) as exc:
                        row['refusals'].append(dict(path=str(directory), reason=str(exc)))
            rolling_steps = sorted(s['step'] for s in row['states'] if s['phase'] == 'rolling')
            newest_rolling = max(rolling_steps, default=None)
            row['rolling_retention_observation'] = dict(complete_steps_present=rolling_steps,
                earlier_complete_rolling_steps_present=[step for step in rolling_steps
                    if step < newest_rolling - 1] if newest_rolling is not None else [],
                latest_two_policy_observed=bool(rolling_steps) and len(rolling_steps) <= 2 and
                    all(step >= newest_rolling - 1 for step in rolling_steps),
                cleanup_execution_by_this_tool=False,
                historical_deleted_bytes=None,
                scope='Current complete directory/member metadata only; absence is not a deletion receipt')
            try:
                complete = completed_group(base, receipt, study, root, source_commit)
                require(not any(Path(w['path']).is_relative_to(base) for w in writers),
                        'group has an open writable file descriptor')
                final = state_by_key[150].get('post')
                require(all('post' in state_by_key[step] for step in KEYS),
                        'completed group missing a protected post-evaluation keypoint')
                require(final is not None and complete.get('checkpoint') ==
                        dict(path=final['path'], sha256=final['sha256'], native_only=False),
                        'completed150 not bound to canonical post-evaluation manifest')
                require(not row['refusals'], 'missing/unsafe native state: refuse all group candidates')
                for step, pair in state_by_key.items():
                    if 'pre' in pair:
                        require('post' in pair and pair['pre']['identity'] == pair['post']['identity'] and
                                pair['pre']['hf_identity'] == pair['post']['hf_identity'],
                                'pre-evaluation state lacks exact same-key canonical post/HF')
                row['status'] = 'COMPLETE_EXIT_WAIT_RELEASE_ACCEPTED'
                for step, pair in state_by_key.items():
                    if 'pre' in pair:
                        row['candidates'].append(dict(kind='SUPERSEDED_PRE_EVALUATION_PAYLOADS',
                            step=step, manifest=pair['pre']['path'], canonical_manifest=pair['post']['path'],
                            files=pair['pre']['members']))
                for snapshot in row['states']:
                    if snapshot['phase'] == 'rolling':
                        row['candidates'].append(dict(kind='COMPLETED_GROUP_OBSOLETE_ROLLING_PAYLOADS',
                            step=snapshot['step'], manifest=snapshot['path'], canonical_manifest=final['path'],
                            files=snapshot['members']))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                row['cleanup_refusal'] = str(exc)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            row.update(status='REFUSED_BINDING_OR_INVENTORY', candidates=[], cleanup_refusal=str(exc))
        rows.append(row)
    fs = os.statvfs(root)
    mounts = []
    for path in (root, Path('/mnt/shared/workspace/training_data_node_5')):
        try:
            mounts.append(mounted_capacity(path))
        except (OSError, ValueError) as error:
            mounts.append(dict(path=str(path), error=str(error), transfer_executed=False))
    return dict(protocol='ablation150_storage_metadata_v1', observed_time=time.time(),
        root=str(root), source_commit=source_commit, study_sha256=study_sha,
        statvfs_free_bytes=fs.f_bavail * fs.f_frsize, existing_mounted_capacity=mounts,
        open_writers=writers, groups=rows,
        actual_receipt=dict(deletion_executed=False, deleted_files=[], deleted_bytes=0),
        limits='Stat/metadata only; no payload hashing, tensor reload, archival, deletion or training action')


def plan(snapshot):
    states = [s for group in snapshot['groups'] for s in group.get('states', [])]
    native_sizes = [sum(f['bytes'] for f in s['members']) for s in states]
    hf_sizes = [sum(f['bytes'] for f in s['hf']) for s in states if s['hf']]
    require(native_sizes and hf_sizes, 'need at least one observed full native and HF checkpoint size')
    native, hf = max(native_sizes), max(hf_sizes)
    protected = {f['path']: f for s in states if s['phase'] == 'post' for f in s['members'] + s['hf']}
    protected_now = sum(f['bytes'] for f in protected.values())
    inventoried = {f['path']: f for s in states for f in s['members'] + s['hf']}
    inventoried_now = sum(f['bytes'] for f in inventoried.values())
    final = len(GROUPS) * len(KEYS) * (native + hf)
    peak = final + 6 * native + HEADROOM
    # Existing two rolling saves remain occupied; a future keypoint adds HF,
    # pre-eval native, then post-eval native. Guard only this real next-write
    # burst, not the full seven-run forecast. The forecast can request archival.
    burst = 2 * native + hf + HEADROOM
    free = snapshot['statvfs_free_bytes']
    low = free < burst
    candidates = [c for group in snapshot['groups'] for c in group.get('candidates', [])]
    files = {f['path']: f for c in candidates for f in c['files']}
    reclaim = sum(f['allocated_bytes'] for f in files.values())
    return dict(protocol='ablation150_storage_plan_v1', planned_time=time.time(),
        metadata_observed_time=snapshot['observed_time'], root=snapshot['root'],
        source_commit=snapshot['source_commit'], deletion_executed=False,
        measured_native_bytes_max=native, measured_hf_bytes_max=hf,
        measured_protected_bytes_now=protected_now,
        measured_all_inventoried_payload_bytes_now=inventoried_now,
        projected_seven_group_protected_bytes=final,
        projected_additional_protected_bytes=max(0, final - protected_now),
        projected_additional_final_bytes_after_completed_group_pruning=max(0, final - inventoried_now),
        projected_peak_after_completed_group_cleanup_bytes=peak,
        projected_additional_peak_bytes_after_completed_group_pruning=max(0, peak - inventoried_now),
        projected_additional_peak_deficit_on_current_mount_bytes=max(0, peak - inventoried_now - free),
        projected_seven_group_payload_bytes_without_any_pruning=final + 6 * native * len(GROUPS),
        projection_scope='Max observed sizes; final retains four HF + four post-eval full states/group. Peak retains current group four pre-eval + two rolling, cleaned completed groups. Uncounted logs/partials/cache require added capacity.',
        space_guard=dict(fresh_free_bytes=free, next_key_write_burst_with_headroom_bytes=burst,
                         cleanup_needed=low, additional_bytes_needed=max(0, burst - free)),
        metadata_eligible_candidate_count=len(candidates), candidate_allocated_bytes=reclaim,
        candidates_for_review=candidates if low else [],
        withheld_candidates_reason=None if low else 'actual next-write space is sufficient; no cleanup needed',
        actual_receipt=snapshot['actual_receipt'],
        required_before_any_separate_deletion=['Fresh same-host collect+plan; Linux /proc writer scan.',
            'Recheck exact manifest/config/member metadata and no active load/save reference.',
            'Hash every protected canonical native/HF payload against recorded digest.',
            'Persist a separate actual receipt with exact deleted file list; retained pre manifest must say payload pruned, never fresh COMPLETE.'],
        recommendation='If protected final/peak exceeds capacity, archive only completed canonical own checkpoints to existing same-cluster storage with full per-file SHA acceptance; do not widen cleanup scope.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--metadata-json', help='Replay already captured small metadata; no remote access')
    parser.add_argument('--root', default=str(ROOT), choices=[str(ROOT)])
    args = parser.parse_args()
    if args.metadata_json:
        snapshot = json.loads(Path(args.metadata_json).read_text())
        require(snapshot['root'] == str(ROOT) and snapshot['source_commit'] == args.source_commit,
                'replay metadata not exact root/source')
        result = dict(metadata_replayed=True, live_cleanup_authorized=False, plan=plan(snapshot))
    else:
        snapshot = collect(Path(args.root), args.source_commit)
        result = dict(metadata_replayed=False, live_cleanup_authorized=False,
                      metadata=snapshot, plan=plan(snapshot))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

"""Retire only redundant raw candidate tensors from old, successful Easy pools.

This separate CPU/stdlib operator never imports Torch or modifies training code.
It preserves selected-token evidence, all-group selection JSON, before/after RNG,
active/failed pools, and at least two latest completed outers. The retired raw
tensors can no longer reconstruct unselected candidates' full token outputs.
Checkpoint/optimizer/RNG recovery does not consume these diagnostic tensors in
the reviewed Easy recipe. A caller-selected source must carry matching published
source, runtime-copy and run-manifest bindings; HEAD is never inferred here.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid


REVIEWED_TRAINING_COMMIT = '93eda66a978dae7a1086fdb2e6f07ef4a7fc3d48'
RAW_NAMES = ('0.pt', '512.pt', '1024.pt', '1536.pt', '2048.pt')
STATUS = 'EASY2560_OLD_SUCCESSFUL_CANDIDATE_RAW_RETIRED'


def require(condition, message):
    if not condition: raise ValueError('EASY2560_RAW_RETIRE_' + message)


def owned_base(value):
    path = Path(value)
    require(path.is_absolute() and path.resolve() == path and re.fullmatch(
        r'/data/user/jxu957/experiments/easy2560-corollout-joint4-[A-Za-z0-9._-]+', str(path)),
        'DEDICATED_NAMESPACE_REQUIRED')
    return path


def identity(st):
    return dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size,
                mtime_ns=st.st_mtime_ns, uid=st.st_uid, links=st.st_nlink)


def checked_path(base, relative, *, directory=False):
    path = base / relative
    require(path.resolve() == path and path.is_relative_to(base), 'SYMLINK_OR_ESCAPE')
    st = path.stat()
    require(st.st_uid == os.getuid(), 'FOREIGN_OWNER')
    require(stat.S_ISDIR(st.st_mode) if directory else stat.S_ISREG(st.st_mode), 'WRONG_FILE_KIND')
    return path


def read_file(base, relative, *, parse_json=False):
    path = checked_path(base, relative)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd); h = hashlib.sha256(); data = [] if parse_json else None
        while True:
            block = os.read(fd, 8 * 1024**2)
            if not block: break
            h.update(block)
            if data is not None: data.append(block)
        after = os.fstat(fd)
        require(identity(before) == identity(after) == identity(path.stat()), 'FILE_CHANGED_DURING_READ')
    finally: os.close(fd)
    record = dict(relative_path=str(relative), sha256=h.hexdigest(), **identity(after))
    return record, json.loads(b''.join(data)) if parse_json else None


def completed_updates(base):
    # The ledger is append-only during healthy training. A trailing incomplete
    # write is not a completed outer and cannot authorize retirement.
    path = checked_path(base, 'logs/actual-updates.jsonl')
    raw = path.read_bytes(); prefix = b'\n'.join(raw.split(b'\n')[:-1])
    if prefix: prefix += b'\n'
    rows = [json.loads(line) for line in prefix.splitlines() if line]
    require([row['step'] for row in rows] == list(range(1, len(rows)+1)), 'NONSEQUENTIAL_COMPLETED_LEDGER')
    require(len({row['attempt_id'] for row in rows}) == len(rows), 'DUPLICATE_COMPLETED_ATTEMPT')
    return rows, dict(bytes=len(prefix), sha256=hashlib.sha256(prefix).hexdigest())


def bind(base, run_id, training_commit, source_job):
    require(isinstance(training_commit, str) and re.fullmatch('[0-9a-f]{40}', training_commit),
            'EXPLICIT_FULL_TRAINING_COMMIT_REQUIRED')
    require(isinstance(run_id, str) and re.fullmatch('[0-9a-f]{32}', run_id) and
        isinstance(source_job, str) and re.fullmatch('[1-9][0-9]*', source_job),
        'EXPLICIT_RUN_OR_SOURCE_DIFFERS')
    manifest_record, run = read_file(base, 'logs/run-manifest.json', parse_json=True)
    require(run.get('run_id') == run_id and run.get('source_commit') == training_commit,
            'EXPLICIT_RUN_OR_SOURCE_DIFFERS')
    require(run.get('easy2560_selection') is True and run.get('profile') == 'common8' and
        run.get('training_world_size') == 4 and run.get('execution_site') == 'hpc3' and
        run.get('execution_backend', 'slurm') == 'slurm' and run.get('data_selection_mode') == 'mixed_first' and
        type(run.get('actor_logical_micro_batch_per_rank')) is int and run['actor_logical_micro_batch_per_rank'] == 4 and
        type(run.get('actor_micro_batch_per_rank')) is int and run['actor_micro_batch_per_rank'] in (32, 16, 8, 4) and
        run.get('all_correct_virtual_zero') is False and not run.get('question_probe') and
        run.get('selection_candidate_questions') == 2560 and run.get('selection_selected_questions') == 512 and
        run.get('selection_fixed_candidate_budget') is True and not run.get('full_state_restore'),
        'EXPLICIT_FRESH_NATIVE4_EASY_SCOPE_REQUIRED')
    allocation_record, allocation = read_file(base, 'logs/native-allocation-admission.json', parse_json=True)
    require(allocation.get('accepted') is True and allocation.get('run_id') == run_id and
        allocation.get('slurm_job_id') == source_job, 'SOURCE_JOB_DIFFERS')
    if 'slurm_job_id' in run: require(run['slurm_job_id'] == source_job, 'RUN_JOB_DIFFERS')
    bindings = [manifest_record, allocation_record]
    if training_commit != REVIEWED_TRAINING_COMMIT:
        # The launcher already SHA-verified all published source bytes before
        # copying the runtime repo. Authenticate those small immutable receipts,
        # not 1301 source files again on each maintenance poll.
        publication_record, publication = read_file(base,
            'source/' + training_commit + '/source-publication.json', parse_json=True)
        runtime_record, runtime = read_file(base, 'preparation/run-source.json', parse_json=True)
        require(publication.get('source_commit') == training_commit and
            publication.get('fresh_source_all_files_sha_verified') is True and
            isinstance(publication.get('files'), dict) and publication['files'],
            'EXPLICIT_PUBLISHED_SOURCE_REQUIRED')
        require(runtime.get('source_commit') == training_commit and
            runtime.get('files') == publication['files'] and runtime.get('runtime_repo') == str(base/'repo'),
            'PUBLISHED_RUNTIME_SOURCE_DIFFERS')
        source_map = run.get('source_sha256')
        require(isinstance(source_map, dict) and source_map, 'BOUND_RUNTIME_SOURCE_MAP_REQUIRED')
        for name, sha in source_map.items():
            record = publication['files'].get(name)
            require(isinstance(name, str) and name and not Path(name).is_absolute() and '..' not in Path(name).parts and
                isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha) and
                isinstance(record, dict) and record.get('sha256') == sha and
                type(record.get('bytes')) is int and record['bytes'] >= 0,
                'PUBLISHED_RUNTIME_SOURCE_SHA_DIFFERS')
        bindings.extend((publication_record, runtime_record))
    return run, bindings


def selection_pool(base, run, update):
    """Cheap exact pool binding; it never authorizes deletion by itself."""
    attempt = update['attempt_id']; step = update['step']
    require(re.fullmatch('[0-9a-f]{32}', attempt), 'INVALID_ATTEMPT_ID')
    audit_record, audit = read_file(base, 'artifacts/mixed-selection/' + attempt + '.json', parse_json=True)
    snapshot = run['run_id'] + ':' + str(step) + ':policy-' + attempt
    require(audit['step'] == step and audit['source_commit'] == run['source_commit'] and
        audit['policy_snapshot_id'] == snapshot and audit['fixed_candidate_budget'] is True and
        audit['attempted_candidates'] == 2560 and audit['responses_per_group'] == 5 and
        audit['selected_groups'] == 512, 'SELECTION_BINDING_DIFFERS')
    retry = audit['pool_retry']
    require(type(retry) is int and retry >= 0, 'INVALID_POOL_RETRY')
    relative = 'artifacts/mixed-candidate-pools/' + attempt + '-pool-' + str(retry)
    pool = checked_path(base, relative, directory=True)
    require(audit['pool_path'] == str(pool) and not (pool/'selection-failure.json').exists(),
            'FAILED_OR_FOREIGN_POOL')
    return audit_record, audit, relative


def pool_evidence(base, run, update, source_job):
    attempt = update['attempt_id']; step = update['step']
    audit_record, audit, relative = selection_pool(base, run, update)
    snapshot = run['run_id'] + ':' + str(step) + ':policy-' + attempt
    retry = audit['pool_retry']
    groups = audit['groups']
    require([g['ordinal'] for g in groups] == list(range(2560)), 'INCOMPLETE_SELECTION_GROUPS')
    mixed = []; correct = []; all_ids = []; selected = []
    for group in groups:
        values = group['strict_correctness']; ids = group['trajectory_ids']; ordinal = group['ordinal']
        require(len(values) == len(ids) == 5 and all(type(v) is int and v in (0, 1) for v in values),
                'INVALID_STRICT_GROUP')
        category = 'mixed' if 0 < sum(values) < 5 else 'all_correct' if sum(values) == 5 else 'all_wrong'
        require(group['category'] == category and type(group['selected']) is bool, 'GROUP_CLASSIFICATION_DIFFERS')
        if category == 'mixed': mixed.append(ordinal)
        elif category == 'all_correct': correct.append(ordinal)
        if group['selected']: selected.append(ordinal)
        all_ids.extend(ids)
    require(len(set(all_ids)) == 12800, 'CANDIDATE_IDS_REPEAT')
    expected = sorted(mixed[:512] if len(mixed) >= 512 else mixed + correct[:512-len(mixed)])
    require(len(expected) == 512 and selected == expected, 'MIXED_FIRST_SELECTION_DIFFERS')
    selected_ids = [value for group in groups if group['selected'] for value in group['trajectory_ids']]
    gate_record, gate = read_file(base, 'logs/behavior-probability-attempt-' + attempt + '.json', parse_json=True)
    require(gate_record['sha256'] == update['behavior_probability_gate_sha256'] and
        gate['accepted'] is True and gate['run_id'] == run['run_id'] and
        gate['source_sha256'] == run['source_sha256'] and gate['slurm_job_id'] == source_job and
        gate['logical_update'] == step and gate['attempt_id'] == attempt and
        gate['attempt_index'] == update['attempt_index'] and gate['policy_snapshot_id'] == snapshot and
        gate['trajectory_count'] == 2560 and gate['trajectory_ids'] == selected_ids and
        set(gate['row_tensor_sha256']) == set(selected_ids), 'SELECTED_BEHAVIOR_GATE_DIFFERS')
    trace_record, _ = read_file(base, 'artifacts/behavior-traces/attempt-' + attempt + '.json')
    require(trace_record['sha256'] == update['behavior_trace_sha256'] ==
        gate['trace_sha256'] == gate['private_trace_sha256'], 'SELECTED_PRIVATE_TRACE_SHA_DIFFERS')
    progress_record, progress = read_file(base, relative + '/progress.json', parse_json=True)
    require(progress['step'] == step and progress['complete_ordinals'] == list(range(2560)) and
            progress['policy_snapshot_id'] == snapshot and len(progress['reserved']) == 2560,
            'POOL_GENERATION_INCOMPLETE')
    kept = []
    for start in (0, 512, 1024, 1536, 2048):
        for suffix in ('-before-generation.pt', '-after-generation.pt'):
            record, _ = read_file(base, relative + '/' + str(start) + suffix); kept.append(record)
        for suffix in ('-worker-rng', '-after-generation-worker-rng'):
            for rank in range(4):
                record, _ = read_file(base, relative + '/' + str(start) + suffix + '/rank-' + str(rank) + '.pt')
                kept.append(record)
    return dict(step=step, attempt_id=attempt, pool_retry=retry, pool_relative_path=relative,
        preserved_evidence=[audit_record, gate_record, trace_record, progress_record, *kept])


def is_full_success(update):
    return (update.get('update_status') == 'complete' and update.get('full_budget_complete') is True and
        update.get('actual_optimizer_writes') == 20 and update.get('opportunities_processed') == 20 and
        update.get('optimizer_writes_after') - update.get('optimizer_writes_before') == 20)


def build_plan(base, run_id, training_commit, source_job, keep_completed=2):
    base = owned_base(base)
    require(type(keep_completed) is int and keep_completed >= 2, 'KEEP_AT_LEAST_TWO_COMPLETED')
    run, bindings = bind(base, run_id, training_commit, source_job)
    rows, ledger = completed_updates(base)
    require(all(row['run_id'] == run_id for row in rows), 'FOREIGN_COMPLETED_LEDGER')
    pools = []; skipped = []
    for update in rows[:-keep_completed]:
        if not is_full_success(update):
            skipped.append(dict(step=update['step'], reason='partial_or_nonfull_update_preserved')); continue
        try:
            # Completed old pools are immutable. Once all five target paths are
            # absent, there is no deletion to authorize and no free-byte claim
            # to make. Avoid repeatedly hashing their large selected traces.
            _, _, relative = selection_pool(base, run, update)
            states = []
            for name in RAW_NAMES:
                try: states.append((name, (base/relative/name).lstat()))
                except FileNotFoundError: states.append((name, None))
            if all(value is None for _, value in states):
                skipped.append(dict(step=update['step'],
                    reason='all_five_raw_absent_no_new_retirement_or_free_byte_claim')); continue
            require(all(value is not None for _, value in states), 'PARTIAL_RAW_SET_PRESERVED')
            require(all(stat.S_ISREG(value.st_mode) and value.st_uid == os.getuid() and value.st_nlink == 1
                        for _, value in states), 'ABNORMAL_RAW_SET_PRESERVED')
            pool = pool_evidence(base, run, update, source_job)
            files = [read_file(base, pool['pool_relative_path'] + '/' + name)[0] for name in RAW_NAMES]
            require(all(record['links'] == 1 for record in files), 'RAW_HARDLINK_NOT_UNIQUE')
            pool.update(files=files); pools.append(pool)
        except (OSError, ValueError, KeyError, TypeError) as error:
            skipped.append(dict(step=update['step'], reason=type(error).__name__ + ': ' + str(error)))
    return dict(schema_version=1, role='easy2560_candidate_raw_retirement_plan', base=str(base),
        run_id=run_id, training_commit=training_commit, source_job=source_job,
        keep_completed=keep_completed, latest_completed_outer=len(rows),
        ledger_prefix=ledger, binding_files=bindings, pools=pools, skipped=skipped,
        removable_logical_bytes=sum(record['bytes'] for pool in pools for record in pool['files']),
        retained_scope='selected private token/probability trace; all-group selection JSON; all RNG/before/after; all checkpoints',
        lost_scope='full token/tensor replay of unselected candidates in retired old successful pools',
        operator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checked_unix=time.time())


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


def publish(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as out:
        out.write(raw); out.flush(); os.fsync(out.fileno())
    fsync_directory(path.parent)


def save_plan(plan):
    base = owned_base(plan['base'])
    root = base/'preparation/candidate-raw-retirement'
    require(root.resolve() == root, 'OPERATOR_ROOT_SYMLINK')
    root.mkdir(parents=True, exist_ok=True)
    directory = root/uuid.uuid4().hex; directory.mkdir()
    fsync_directory(root)
    path = directory/'plan.json'; publish(path, plan)
    return path


def apply_plan(plan_path, run_id, training_commit, source_job):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text()); base = owned_base(plan['base'])
    relative = plan_path.relative_to(base)
    require(re.fullmatch(r'preparation/candidate-raw-retirement/[0-9a-f]{32}/plan.json', str(relative)),
            'PLAN_PATH_OUTSIDE_OPERATOR_SCOPE')
    plan_record, disk_plan = read_file(base, relative, parse_json=True)
    require(plan == disk_plan and (run_id, training_commit, source_job) ==
        (plan['run_id'], plan['training_commit'], plan['source_job']), 'PLAN_IDENTITY_DIFFERS')
    root = plan_path.parent.parent
    lock_fd = os.open(root/'maintenance.lock', os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = plan_path.parent/'result.json'
        if result_path.exists():
            _, result = read_file(base, result_path.relative_to(base), parse_json=True)
            require(result['plan_sha256'] == plan_record['sha256'], 'EXISTING_RESULT_PLAN_DIFFERS')
            return result
        run, bindings = bind(base, run_id, training_commit, source_job)
        require(bindings == plan['binding_files'], 'RUN_BINDING_FILES_CHANGED')
        rows, _ = completed_updates(base); ledger = checked_path(base, 'logs/actual-updates.jsonl')
        with ledger.open('rb') as source: prefix = source.read(plan['ledger_prefix']['bytes'])
        require(hashlib.sha256(prefix).hexdigest() == plan['ledger_prefix']['sha256'], 'COMPLETED_LEDGER_PREFIX_CHANGED')
        keep = plan['keep_completed']; require(type(keep) is int and keep >= 2, 'KEEP_AT_LEAST_TWO_COMPLETED')
        allowed = {row['step']: row for row in rows[:-keep]}
        # Before any deletion, reauthenticate all retained evidence and exactly
        # five target paths per pool. No unlisted filename can reach unlink.
        for pool in plan['pools']:
            update = allowed.get(pool['step'])
            require(update is not None and is_full_success(update), 'POOL_NOT_OLD_FULL_SUCCESS')
            fresh = pool_evidence(base, run, update, source_job)
            require(fresh == {key: value for key, value in pool.items() if key != 'files'}, 'PRESERVED_EVIDENCE_CHANGED')
            require([Path(row['relative_path']).name for row in pool['files']] == list(RAW_NAMES) and
                all(str(Path(row['relative_path']).parent) == pool['pool_relative_path'] for row in pool['files']),
                'TARGETS_OUTSIDE_FIVE_RAW_FILES')
            for record in pool['files']:
                path = base/record['relative_path']
                if path.exists():
                    checked_path(base, record['relative_path'])
                    require(identity(path.stat()) == {key: record[key] for key in identity(path.stat())} and
                            record['links'] == 1, 'RAW_METADATA_CHANGED')
        journal = plan_path.parent/'actions.jsonl'
        prior = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        prepared = {row['relative_path']: row for row in prior if row['stage'] == 'PREPARED'}
        actions = []
        def log(row):
            fd = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            try:
                content = (json.dumps(row, sort_keys=True, allow_nan=False)+'\n').encode()
                require(os.write(fd, content) == len(content), 'SHORT_JOURNAL_WRITE'); os.fsync(fd)
            finally: os.close(fd)
            fsync_directory(journal.parent)
        for pool in plan['pools']:
            directory = checked_path(base, pool['pool_relative_path'], directory=True)
            dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                for record in pool['files']:
                    name = Path(record['relative_path']).name
                    try: fresh, _ = read_file(base, record['relative_path'])
                    except FileNotFoundError:
                        require(prepared.get(record['relative_path'], {}).get('file') == record,
                                'UNEXPLAINED_MISSING_RAW_FILE')
                        row = dict(stage='ABSENT_AFTER_PREPARED', relative_path=record['relative_path'],
                            file=record, checked_unix=time.time()); log(row); actions.append(row); continue
                    require(fresh == record and fresh['links'] == 1, 'RAW_CHANGED_OR_HARDLINKED')
                    row = dict(stage='PREPARED', relative_path=record['relative_path'], file=record,
                               checked_unix=time.time()); log(row)
                    # Recheck the exact inode through the already-open pool
                    # directory immediately before unlinking that basename.
                    current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                    require(stat.S_ISREG(current.st_mode) and identity(current) ==
                        {key: record[key] for key in identity(current)},
                        'RAW_REPLACED_BEFORE_UNLINK')
                    os.unlink(name, dir_fd=dir_fd); os.fsync(dir_fd)
                    require(not (directory/name).exists(), 'RAW_STILL_PRESENT_AFTER_UNLINK')
                    row = dict(stage='UNLINKED', relative_path=record['relative_path'], file=record,
                               checked_unix=time.time()); log(row); actions.append(row)
            finally: os.close(dir_fd)
        result = dict(status=STATUS, run_id=run_id, training_commit=training_commit, source_job=source_job,
            plan_sha256=plan_record['sha256'], pool_count=len(plan['pools']), actions=actions,
            retired_logical_bytes=sum(row['file']['bytes'] for row in actions if row['stage'] == 'UNLINKED'),
            unconfirmed_prior_unlink_bytes=sum(row['file']['bytes'] for row in actions if row['stage'] != 'UNLINKED'),
            filesystem_quota_free_bytes_measured=False, preserved_scope=plan['retained_scope'],
            lost_scope=plan['lost_scope'], checked_unix=time.time())
        publish(result_path, result); return result
    finally: os.close(lock_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'apply'))
    parser.add_argument('--base'); parser.add_argument('--plan')
    parser.add_argument('--run-id', required=True); parser.add_argument('--training-commit', required=True)
    parser.add_argument('--source-job', required=True); parser.add_argument('--keep-completed', type=int, default=2)
    args = parser.parse_args()
    if args.mode == 'plan':
        plan = build_plan(args.base, args.run_id, args.training_commit, args.source_job, args.keep_completed)
        path = save_plan(plan)
        print(json.dumps(dict(plan_path=str(path), pools=len(plan['pools']),
            removable_logical_bytes=plan['removable_logical_bytes'], skipped=plan['skipped'])), flush=True)
    else:
        print(json.dumps(apply_plan(args.plan, args.run_id, args.training_commit, args.source_job)), flush=True)


if __name__ == '__main__': main()

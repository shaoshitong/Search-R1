"""One owned CPU process maintains old successful candidate raw tensors.

This runner observes one explicit Slurm job and delegates every retirement to
the separately published, SHA-bound helper. Observation failures never imply a
terminal job or authorize deletion. It imports no model/Torch/GPU/credentials,
never submits/restarts/cancels training, and never edits the active source.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time


BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-20261017'
RUN_ID = 'e6d2bcb5dbf44850a1917070f9e4b880'
TRAINING_COMMIT = '93eda66a978dae7a1086fdb2e6f07ef4a7fc3d48'
SOURCE_JOB = '730080'
TERMINAL = frozenset(('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                     'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'))
EXPECTED_SKIP = frozenset(('partial_or_nonfull_update_preserved',
    'all_five_raw_absent_no_new_retirement_or_free_byte_claim'))


def require(condition, message):
    if not condition: raise ValueError('EASY2560_RAW_RUNNER_' + message)


def scope(base, run_id, training_commit, source_job):
    binding(run_id, training_commit, source_job)
    path = Path(base)
    require(path.is_absolute() and path.resolve() == path and re.fullmatch(
        r'/data/user/jxu957/experiments/easy2560-corollout-joint4-[A-Za-z0-9._-]+', str(path)) and
        os.getuid() == 14090, 'CANONICAL_JXU957_SCOPE_REQUIRED')
    return path


def binding(run_id=None, training_commit=None, source_job=None):
    """Keep legacy defaults while threading explicit caller identity everywhere."""
    run_id = RUN_ID if run_id is None else run_id
    training_commit = TRAINING_COMMIT if training_commit is None else training_commit
    source_job = SOURCE_JOB if source_job is None else source_job
    require(isinstance(run_id, str) and re.fullmatch('[0-9a-f]{32}', run_id) and
        isinstance(training_commit, str) and re.fullmatch('[0-9a-f]{40}', training_commit) and
        isinstance(source_job, str) and re.fullmatch('[1-9][0-9]*', source_job),
        'EXPLICIT_RUN_SOURCE_JOB_REQUIRED')
    return dict(run_id=run_id, training_commit=training_commit, source_job=source_job)


def load_helper(base, helper_commit, helper_sha256, *, run_id=None, training_commit=None, source_job=None):
    bound = binding(run_id, training_commit, source_job)
    require(re.fullmatch('[0-9a-f]{40}', helper_commit) and
            re.fullmatch('[0-9a-f]{64}', helper_sha256), 'EXPLICIT_HELPER_COMMIT_AND_SHA_REQUIRED')
    root = base/'preparation/candidate-raw-maintenance-code'/helper_commit
    require(root.resolve() == root and root.stat().st_uid == os.getuid(), 'HELPER_DIRECTORY_DIFFERS')
    manifest_path = root/'source-manifest.json'; script = root/'retire_candidate_raw.py'
    require(manifest_path.resolve() == manifest_path and script.resolve() == script, 'HELPER_SYMLINK')
    manifest = json.loads(manifest_path.read_text()); raw = script.read_bytes()
    require(manifest['operator_commit'] == helper_commit and
        all(manifest.get(key) == value for key, value in bound.items()) and manifest['base'] == str(base),
        'HELPER_MANIFEST_RUN_BINDING_DIFFERS')
    require(manifest['files']['retire_candidate_raw.py'] ==
        dict(bytes=len(raw), sha256=helper_sha256) and hashlib.sha256(raw).hexdigest() == helper_sha256,
        'HELPER_SOURCE_SHA_DIFFERS')
    spec = importlib.util.spec_from_file_location('bound_easy2560_candidate_raw_helper', script)
    module = importlib.util.module_from_spec(spec)
    # Execute the bytes just authenticated, rather than reopening the file or
    # writing a bytecode cache into the immutable helper deployment directory.
    exec(compile(raw, str(script), 'exec'), module.__dict__)
    return module


def observe_job(job, run_command=subprocess.run, *, expected_job=None):
    """squeue proves current presence; only exact main-job sacct proves terminal."""
    require(job == (SOURCE_JOB if expected_job is None else expected_job) and
            isinstance(job, str) and re.fullmatch('[1-9][0-9]*', job), 'CANNOT_OBSERVE_ANOTHER_JOB')
    commands = [(['squeue', '--noheader', '--jobs', job, '--format=%i|%T'], 'squeue'),
                (['sacct', '--noheader', '--parsable2', '--jobs', job,
                  '--format=JobIDRaw,State,ExitCode,End'], 'sacct')]
    for command, authority in commands:
        try: result = run_command(command, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as failure:
            return dict(observation='UNKNOWN', source_job=job, authority=authority,
                        error=type(failure).__name__, terminal_confirmed=False)
        if result.returncode:
            # Slurm commonly returns1 once this exact job has left squeue.
            # Accounting still has to confirm the main job's terminal state;
            # queue failure alone is never treated as completion or deletion.
            if authority == 'squeue' and result.returncode == 1: continue
            return dict(observation='UNKNOWN', source_job=job, authority=authority,
                        returncode=result.returncode, error=result.stderr[-1000:], terminal_confirmed=False)
        rows = [line.strip().split('|') for line in result.stdout.splitlines() if line.strip()]
        if authority == 'squeue':
            if not rows: continue
            if len(rows) != 1 or len(rows[0]) != 2 or rows[0][0] != job:
                return dict(observation='UNKNOWN', source_job=job, authority=authority,
                            error='unexpected_queue_identity', terminal_confirmed=False)
            states = rows[0][1].split()
            if not states:
                return dict(observation='UNKNOWN', source_job=job, authority=authority,
                            error='missing_queue_state', terminal_confirmed=False)
            state = states[0].rstrip('+')
            # A queued terminal-looking state still needs accounting confirmation.
            if state in TERMINAL: continue
            return dict(observation='LIVE', source_job=job, state=state,
                        authority=authority, terminal_confirmed=False)
        exact = [row for row in rows if row and row[0] == job]
        if len(exact) == 1 and len(exact[0]) == 4:
            _, raw_state, exit_code, ended = exact[0]
            states = raw_state.split(); state = states[0].rstrip('+') if states else ''
            if state in TERMINAL and ended not in ('', 'Unknown', 'N/A', 'None'):
                return dict(observation='TERMINAL', source_job=job, state=state, exit_code=exit_code,
                    end_time=ended, authority=authority, terminal_confirmed=True)
        return dict(observation='UNKNOWN', source_job=job, authority=authority,
                    error='missing_or_nonterminal_exact_accounting_row', terminal_confirmed=False)


def initial_state():
    return dict(polls=0, last_planned_frontier=0, last_observed_completed_outer=0,
        confirmed_unlinked_allocated_bytes=0, retired_logical_bytes=0,
        filesystem_quota_free_bytes_measured=False, operations=[])


def cycle(base, helper, state, observation, *, dry_observe=False,
          run_id=None, training_commit=None, source_job=None):
    bound = binding(run_id, training_commit, source_job)
    require(observation['source_job'] == bound['source_job'], 'OBSERVATION_JOB_DIFFERS')
    state['polls'] += 1
    state['source_job_observation'] = observation
    if observation['observation'] == 'UNKNOWN':
        state['phase'] = 'WAITING_OBSERVATION'; return False
    if observation['observation'] == 'TERMINAL':
        require(observation['terminal_confirmed'] is True and observation['authority'] == 'sacct',
                'UNCONFIRMED_TERMINAL_OBSERVATION')
        if observation['state'] != 'COMPLETED' or observation['exit_code'] != '0:0':
            state['phase'] = 'SOURCE_JOB_TERMINAL_FAILURE'; return True
    helper.bind(base, **bound)
    ledger = base/'logs/actual-updates.jsonl'
    try: ledger.lstat()
    except FileNotFoundError: rows = []
    else: rows, _ = helper.completed_updates(base)
    require(all(row['run_id'] == bound['run_id'] for row in rows), 'FOREIGN_COMPLETED_LEDGER')
    frontier = rows[-1]['step'] if rows else 0; state['last_observed_completed_outer'] = frontier
    state['full_success_outer_count'] = sum(helper.is_full_success(row) for row in rows)
    eligible = [row['step'] for row in rows[:-2] if helper.is_full_success(row)]
    state['eligible_old_success_outer_count'] = len(eligible)
    terminal = observation['observation'] == 'TERMINAL'
    if dry_observe:
        state['phase'] = 'DRY_OBSERVE_ONLY'; return terminal
    if state['full_success_outer_count'] >= 3 and eligible and frontier > state['last_planned_frontier']:
        state['last_planned_frontier'] = frontier
        plan = helper.build_plan(base, **bound, keep_completed=2)
        path = helper.save_plan(plan)
        operation = dict(frontier=frontier, plan_path=str(path), pool_count=len(plan['pools']),
                         phase='PLAN_SAVED', checked_unix=time.time())
        state['operations'].append(operation)
        abnormal = [row for row in plan['skipped'] if row['reason'] not in EXPECTED_SKIP]
        require(not abnormal, 'PLAN_HAS_ABNORMAL_PRESERVED_POOLS:' + json.dumps(abnormal, sort_keys=True))
        if plan['pools']:
            allocation = {}
            for pool in plan['pools']:
                for record in pool['files']:
                    actual = (base/record['relative_path']).lstat()
                    require(stat.S_ISREG(actual.st_mode) and actual.st_nlink == 1 and
                        all(helper.identity(actual)[key] == record[key] for key in helper.identity(actual)),
                        'PRE_APPLY_RAW_IDENTITY_DIFFERS')
                    allocation[record['relative_path']] = int(actual.st_blocks)*512
            result = helper.apply_plan(path, **bound)
            require(result['status'] == helper.STATUS, 'HELPER_DID_NOT_CONFIRM_RETIREMENT')
            actions = result['actions']; unlinked = [row for row in actions if row['stage'] == 'UNLINKED']
            require(len({row['relative_path'] for row in actions}) == len(actions) and
                {row['relative_path'] for row in actions} == set(allocation), 'HELPER_ACTION_PATHS_DIFFER')
            require(all(not (base/row['relative_path']).exists() for row in unlinked), 'UNLINKED_PATH_REAPPEARED')
            # This is measured pre-unlink block allocation of exact inodes that
            # the helper confirms unlinking. It is not planned logical size and
            # not a fresh quota/free-space measurement during active training.
            confirmed = sum(allocation[row['relative_path']] for row in unlinked)
            state['confirmed_unlinked_allocated_bytes'] += confirmed
            state['retired_logical_bytes'] += result['retired_logical_bytes']
            operation.update(phase='APPLIED', helper_result_path=str(path.parent/'result.json'),
                helper_result_sha256=hashlib.sha256((path.parent/'result.json').read_bytes()).hexdigest(),
                confirmed_unlinked_allocated_bytes=confirmed, retired_logical_bytes=result['retired_logical_bytes'],
                unconfirmed_prior_unlink_bytes=result['unconfirmed_prior_unlink_bytes'])
        else: operation['phase'] = 'NO_NEW_RAW_TARGETS'
    state['phase'] = 'SOURCE_JOB_COMPLETED' if terminal else 'WAITING_NEXT_COMPLETED_FRONTIER'
    return terminal


def process_identity():
    raw = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(raw[19]), uid=os.getuid(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        pid_namespace=os.readlink('/proc/self/ns/pid'))


def durable_json(path, value, *, replace=False):
    require(path.resolve() == path, 'RUNNER_RECEIPT_SYMLINK')
    temporary = path.with_name(path.name+'.partial-'+str(os.getpid())) if replace else path
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    if replace: os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


def persistent(base, helper, *, helper_commit, helper_sha256, poll_seconds=180,
               observer=None, sleeper=time.sleep, run_id=None, training_commit=None, source_job=None):
    bound = binding(run_id, training_commit, source_job)
    helper.bind(base, **bound)
    if observer is None: observer = lambda job: observe_job(job, expected_job=bound['source_job'])
    root = base/'preparation/candidate-raw-maintenance-runner'
    require(root.resolve() == root, 'RUNNER_DIRECTORY_SYMLINK')
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    lock = os.open(root/'runner.lock', os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claim_path = root/'launch-claim.json'
        require(not claim_path.exists(), 'EXISTING_LAUNCH_CLAIM_INSPECT_BIRTH_AND_RESULT_NO_AUTOMATIC_RESTART')
        birth = process_identity()
        claim = dict(role='easy2560_candidate_raw_maintenance_runner', base=str(base), **bound, process_identity=birth,
            helper_commit=helper_commit, helper_sha256=helper_sha256, poll_seconds=poll_seconds,
            operator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checked_unix=time.time())
        durable_json(claim_path, claim)
        state = dict(initial_state(), **claim)
        try:
            while True:
                done = cycle(base, helper, state, observer(bound['source_job']), **bound)
                state['checked_unix'] = time.time()
                durable_json(root/'progress.json', state, replace=True)
                if done:
                    durable_json(root/'result.json', state); return state
                sleeper(poll_seconds)
        except Exception as failure:
            state.update(phase='MAINTENANCE_FAILED_NO_AUTOMATIC_RETRY',
                error_type=type(failure).__name__, error=str(failure), checked_unix=time.time())
            durable_json(root/'failure.json', state); raise
    finally: os.close(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-observe', action='store_true'); mode.add_argument('--persistent', action='store_true')
    parser.add_argument('--base', default=BASE); parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--training-commit', default=TRAINING_COMMIT); parser.add_argument('--source-job', default=SOURCE_JOB)
    parser.add_argument('--helper-commit', required=True); parser.add_argument('--helper-sha256', required=True)
    parser.add_argument('--poll-seconds', type=int, default=180)
    args = parser.parse_args(); require(30 <= args.poll_seconds <= 3600, 'BOUNDED_POLL_INTERVAL_REQUIRED')
    base = scope(args.base, args.run_id, args.training_commit, args.source_job)
    bound = binding(args.run_id, args.training_commit, args.source_job)
    helper = load_helper(base, args.helper_commit, args.helper_sha256, **bound)
    if args.dry_observe:
        helper.bind(base, **bound)
        state = dict(initial_state(), base=str(base), **bound)
        cycle(base, helper, state, observe_job(args.source_job, expected_job=args.source_job), dry_observe=True, **bound)
        print(json.dumps(state, sort_keys=True), flush=True); return
    result = persistent(base, helper, helper_commit=args.helper_commit,
                        helper_sha256=args.helper_sha256, poll_seconds=args.poll_seconds, **bound)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result['phase'] != 'SOURCE_JOB_COMPLETED': raise SystemExit(2)


if __name__ == '__main__': main()

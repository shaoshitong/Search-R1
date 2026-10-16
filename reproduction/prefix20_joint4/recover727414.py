"""Repeat outer1 after the saved pre-dispatch DataProto metadata failure.

Pending rollouts stay at the failed base. They are audit evidence, not a
complete model/Adam/probe-RNG checkpoint, and are never claimed as restored.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from reproduction.prefix20_joint4.assets import sha
from reproduction.prefix20_joint4.control import ASSETS, own_base, publish, verify_source, bootstrap, storage_admission
from reproduction.prefix20_joint4.recover726927 import relocate_global_storage

FAILED = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727251-20261017')
TARGET = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727414-20261017')
JOB = '727414'
RUN = '2c00ea250603418697ae7e2b3625c9d4'
OLD_COMMIT = '70b5d833012572ba670b6891a0ca1e4c3c3ab4f2'


def prepare(commit):
    os.umask(0o007)
    own_base(TARGET)
    claim = FAILED/'preparation/recovery727414-target-claim.json'
    if TARGET.exists() or claim.exists():
        raise ValueError('Recovery already attempted; inspect the existing target and submission')
    run = json.loads((FAILED/'logs/run-manifest.json').read_text())
    assert os.getuid() == 14090 and (FAILED/'job.id').read_text().strip() == JOB
    assert run['run_id'] == RUN and run['source_commit'] == OLD_COMMIT and run['completed_updates'] == 0
    terminal = subprocess.run(['sacct', '-nP', '-j', JOB, '--format=JobID,State,ExitCode,User,UID,Account'],
                              capture_output=True, text=True, timeout=25)
    rows = [line.split('|') for line in terminal.stdout.splitlines() if line.strip()]
    assert terminal.returncode == 0 and next(row for row in rows if row[0] == JOB)[1:6] == ['FAILED', '1:0', 'jxu957', '14090', 'jxu957']
    assert any(row[:3] == [JOB+'.extern', 'COMPLETED', '0:0'] for row in rows)
    cleanup = json.loads((FAILED/'logs/cleanup-status.json').read_text())
    assert cleanup['run_id'] == RUN and cleanup['cleaned'] is True
    ledger = json.loads((FAILED/'logs/process-ledger.json').read_text())
    assert ledger['run_id'] == RUN and all(row['state'] == 'STOPPED' for row in ledger['jobs'].values())
    for name in ['actual-updates.jsonl', 'actual-attempts.jsonl'] + [
            '{}-rank-{}.jsonl'.format(kind, rank) for kind in ('optimizer', 'scheduler') for rank in range(4)]:
        path = FAILED/'logs'/name
        assert not path.exists() or path.stat().st_size == 0
    assert not list((FAILED/'artifacts/question-probe/completed').glob('outer-*.json'))
    with (FAILED/'gpu-job.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source, manifest = verify_source(FAILED, commit)
    # Bound only our small failed-run namespaces, not a new full account walk.
    bases = [Path('/data/user/jxu957/experiments')/name for name in (
        'qprobe-teacher-warmup200-prefix20-joint4-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery726927-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery727000-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery727050-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery727148-20261016', FAILED.name)]
    measured = subprocess.run(['du', '-s', '-B1', *map(str, bases)], capture_output=True, text=True, timeout=60)
    assert measured.returncode == 0 and not measured.stderr
    rows = [line.split('\t') for line in measured.stdout.splitlines()]
    assert [row[1] for row in rows] == list(map(str, bases))
    projected = sum(int(row[0]) for row in rows) + sum(size for size, _ in ASSETS.values()) + 2*sum(row['bytes'] for row in manifest['files'].values()) + 64*2**20
    assert projected <= 4*2**30, 'Inspect cumulative recovery storage reserve'
    original = dict(job=JOB, run_id=RUN, source_commit=OLD_COMMIT, completed_outers=0,
        actor_optimizer_writes=0, probe_optimizer_writes=0, terminal=terminal.stdout,
        pending_rollouts_preserved=True, pending_rollouts_restored=False, restart_outer=1)
    publish(claim, dict(target=str(TARGET), source_commit=commit, original=original, checked_unix=time.time()))
    TARGET.mkdir()
    (TARGET/'preparation').mkdir()
    shutil.copytree(source, TARGET/'source'/commit)
    # This receipt binds the same pinned CPU runtime; avoid vendor rehash/refit.
    shutil.copyfile(FAILED/'preparation/probe-cpu-runtime.json', TARGET/'preparation/probe-cpu-runtime.json')
    bootstrap(TARGET, commit)
    root = TARGET/'assets/prefix20'
    root.mkdir()
    for name, (size, digest) in ASSETS.items():
        old, new = FAILED/'assets/prefix20'/name, root/name
        shutil.copyfile(old, new)
        assert new.stat().st_size == size and sha(new) == digest
    storage = relocate_global_storage(FAILED, TARGET, additional_prelaunch_reserve_bytes=4*2**30)
    value = dict(status='ZERO_UPDATE_PAYLOAD_FAILURE_FRESH_OUTER1_PREPARED', target=str(TARGET),
        source_commit=commit, original=original, projected_cumulative_prelaunch_bytes=projected,
        storage_origin=storage, storage=storage_admission(TARGET), training_submitted=False,
        old_runtime_and_rollouts_untouched=True, checked_unix=time.time())
    publish(TARGET/'preparation/recovery-from727414.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.commit)))
